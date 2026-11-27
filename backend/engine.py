import os
import json
import shutil
import inspect
import subprocess
import logging
import gc
import time
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from huggingface_hub import HfApi, login, snapshot_download, get_token
import diffusers
from diffusers import DiffusionPipeline
try:
    from diffusers.hooks import apply_group_offloading
except Exception:
    apply_group_offloading = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LampyreEngine")

MODEL_ID = "briaai/Fibo-2.0-edit-turbo"

EXPECTED_MODEL_INDEX = {
    "_class_name": "BriaFibo2Pipeline",
    "_diffusers_version": "0.41.0.dev0",
    "processor": ["transformers", "Qwen3VLProcessor"],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
    "transformer": ["diffusers", "BriaFibo2Transformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLFlux2"],
}

if os.environ.get("LAMPYRE_USER_DATA"):
    DATA_DIR = os.environ["LAMPYRE_USER_DATA"]
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

MODELS_DIR = os.path.join(DATA_DIR, "models", "Fibo-2.0-edit-turbo")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")

DEFAULT_SETTINGS = {
    "vram_profile": "auto",
    "prompt_encoder": "qwen3-vl",
    "max_edit_side": 0,
    "crop_padding": 64,
    "feather_px": 12,
    "steps": 4,
    "guidance_scale": 1.0,
    "seed": None,
    "auto_unload_idle": False,
}


def load_config() -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)
    base = {"hf_token": "", "settings": dict(DEFAULT_SETTINGS)}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    base["hf_token"] = data.get("hf_token", "") or ""
                    saved_s = data.get("settings", {})
                    if isinstance(saved_s, dict):
                        base["settings"].update({k: v for k, v in saved_s.items() if k in DEFAULT_SETTINGS})
        except Exception:
            pass
    return base


def save_config(cfg: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    merged = {
        "hf_token": (cfg.get("hf_token") or ""),
        "settings": {**DEFAULT_SETTINGS, **(cfg.get("settings") or {})},
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)


def has_nvidia_hardware() -> tuple[bool, str]:
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.check_output(
                [smi, "--query-gpu=name", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=4,
            ).strip()
            if out:
                return True, out.splitlines()[0].strip()
        except Exception:
            pass

    if os.name == "nt":
        try:
            out = subprocess.check_output(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' } | Select-Object -First 1 -ExpandProperty Name)",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            ).strip()
            if out:
                return True, out
        except Exception:
            pass

    return False, ""


class FiboTurboEngine:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.pipe = None
        self.molmo_model = None
        self.molmo_processor = None
        self.molmo_model_id = None
        self._custom_hooks = []
        self.loaded_dtype = None
        self.offload_mode = "none"
        self.last_inference_ms = None
        self.last_peak_vram_mb = None
        self.last_edit_region = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_name = self._resolve_gpu_name()
        self.total_vram_mb, self.free_vram_mb = self._query_vram_mb()

    def emit_log(self, msg: str):
        logger.info(msg)
        if self.log_callback:
            try:
                self.log_callback(msg)
            except Exception:
                pass

    def _resolve_gpu_name(self) -> str:
        if torch.cuda.is_available():
            try:
                return torch.cuda.get_device_name(0)
            except Exception:
                return "GPU NVIDIA (CUDA)"
        found, hw_name = has_nvidia_hardware()
        if found:
            return f"{hw_name} (CUDA non actif)"
        return "Processeur (CPU)"

    def _query_vram_mb(self) -> tuple[float, float]:
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                if hasattr(torch.cuda, "mem_get_info"):
                    free_b, total_b = torch.cuda.mem_get_info()
                    return round(total_b / (1024 ** 2), 1), round(free_b / (1024 ** 2), 1)
                props = torch.cuda.get_device_properties(0)
                total_mb = round(props.total_memory / (1024 ** 2), 1)
                alloc_mb = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1)
                return total_mb, max(0.0, round(total_mb - alloc_mb, 1))
            except Exception:
                pass
        return 0.0, 0.0

    def get_vram_and_model_info(self, settings: dict | None = None) -> dict:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_name = self._resolve_gpu_name()
        total_now, free_now = 0.0, 0.0
        allocated_mb = 0.0
        reserved_mb = 0.0
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                if hasattr(torch.cuda, "mem_get_info"):
                    free_b, total_b = torch.cuda.mem_get_info()
                    total_now = round(total_b / (1024 ** 2), 1)
                    free_now = round(free_b / (1024 ** 2), 1)
                else:
                    props = torch.cuda.get_device_properties(0)
                    total_now = round(props.total_memory / (1024 ** 2), 1)
                    free_now = total_now
                allocated_mb = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1)
                reserved_mb = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 1)
            except Exception:
                pass

        self.total_vram_mb, self.free_vram_mb = total_now, free_now
        profile = self._resolve_vram_profile(settings or {}, total_now)
        max_side = self._resolve_max_edit_side(settings or {}, profile)

        components = []
        if self.pipe is not None:
            for attr in ("transformer", "unet", "vae", "text_encoder", "text_encoder_2"):
                mod = getattr(self.pipe, attr, None)
                if mod is not None:
                    try:
                        param_mb = sum(p.numel() * p.element_size() for p in mod.parameters()) / (1024 ** 2)
                        if attr in ("transformer", "text_encoder") and ("group_offload" in self.offload_mode or "async_stream" in self.offload_mode):
                            dev_str = "async_stream" if "async_stream" in self.offload_mode else "group_offload"
                        else:
                            dev_str = "offload/cpu"
                            try:
                                first_p = next(mod.parameters(), None)
                                if first_p is not None:
                                    dev_str = str(first_p.device)
                            except Exception:
                                pass
                        components.append({
                            "name": attr,
                            "class": type(mod).__name__,
                            "size_mb": round(param_mb, 1),
                            "device": dev_str,
                        })
                    except Exception:
                        pass

        return {
            "model_id": MODEL_ID,
            "pipeline_class": type(self.pipe).__name__ if self.pipe is not None else "Non chargé",
            "loaded": self.pipe is not None,
            "downloaded": self.is_model_downloaded(),
            "device": self.device,
            "gpu_name": self.gpu_name,
            "dtype": str(self.loaded_dtype).replace("torch.", "") if self.loaded_dtype else "auto",
            "offload_mode": self.offload_mode,
            "active_profile": profile,
            "effective_max_edit_side": max_side,
            "vram_total_mb": total_now,
            "vram_free_mb": free_now,
            "vram_allocated_mb": allocated_mb,
            "vram_reserved_mb": reserved_mb,
            "last_inference_ms": self.last_inference_ms,
            "last_peak_vram_mb": self.last_peak_vram_mb,
            "last_edit_region": self.last_edit_region,
            "components": components,
        }

    def _resolve_vram_profile(self, settings: dict, total_vram_mb: float) -> str:
        requested = (settings.get("vram_profile") or "auto").strip().lower()
        if requested in ("6gb", "8gb", "high"):
            return requested
        if self.device != "cuda" or total_vram_mb <= 0:
            return "6gb"
        if total_vram_mb <= 6600:
            return "6gb"
        if total_vram_mb <= 9200:
            return "8gb"
        return "high"

    def _resolve_max_edit_side(self, settings: dict, profile: str) -> int:
        try:
            user_side = int(settings.get("max_edit_side") or 0)
        except Exception:
            user_side = 0
        if user_side >= 256:
            return max(256, (user_side // 16) * 16)
        if profile in ("6gb", "8gb"):
            return 768
        return 1024

    def _select_cuda_dtype(self, profile: str = "auto"):
        if self.device != "cuda":
            return torch.float32
        if profile == "6gb":
            return torch.float16
        try:
            major, _ = torch.cuda.get_device_capability(0)
            # Architecture Ampere (SM 8.0+) et plus récentes gèrent nativement bfloat16 ;
            # Pascal (GTX 10xx, SM 6.1) et Turing (RTX 20xx, SM 7.5) sont optimisés en float16.
            if major >= 8 and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16

    def flush_vram(self, reason: str = "optimisation"):
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            except Exception:
                pass
        self.emit_log(f"[VRAM] Libération mémoire cache GPU ({reason}).")

    def unload_model(self):
        for h in self._custom_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._custom_hooks.clear()
        if getattr(self, "molmo_model", None) is not None:
            self.molmo_model = None
            self.molmo_processor = None
            self.molmo_model_id = None
        if self.pipe is not None:
            self.pipe = None
            self.loaded_dtype = None
            self.offload_mode = "none"
            self.flush_vram("déchargement modèle")

    def resolve_token(self, explicit_token: str = "") -> str | None:
        if explicit_token and explicit_token.strip():
            return explicit_token.strip()
        cfg = load_config()
        if cfg.get("hf_token") and cfg["hf_token"].strip():
            return cfg["hf_token"].strip()
        if os.environ.get("HF_TOKEN") and os.environ["HF_TOKEN"].strip():
            return os.environ["HF_TOKEN"].strip()
        try:
            cached = get_token()
            if cached and cached.strip():
                return cached.strip()
        except Exception:
            pass
        return None

    def verify_token(self, token: str) -> tuple[bool, str]:
        t = (token or "").strip()
        if not t:
            return False, "Jeton vide."
        try:
            api = HfApi(token=t)
            info = api.whoami(token=t)
            user = info.get("name") or info.get("fullname") or "compte Hugging Face"
            return True, f"Authentifié en tant que {user}."
        except Exception as exc:
            return False, f"Jeton invalide ou rejeté par Hugging Face ({type(exc).__name__}: {exc})"

    def _ensure_valid_model_index(self) -> bool:
        """Restaure model_index.json s'il est corrompu alors que les 5 composants locaux sont présents."""
        index_path = os.path.join(MODELS_DIR, "model_index.json")
        required_keys = ("transformer", "scheduler", "vae", "text_encoder", "processor")
        has_all_dirs = all(os.path.isdir(os.path.join(MODELS_DIR, k)) for k in required_keys)

        needs_repair = False
        if not os.path.exists(index_path) or os.path.getsize(index_path) == 0:
            needs_repair = True
        else:
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict) or not all(k in data for k in required_keys):
                    needs_repair = True
            except Exception:
                needs_repair = True

        if needs_repair and has_all_dirs:
            try:
                with open(index_path, "w", encoding="utf-8") as f:
                    json.dump(EXPECTED_MODEL_INDEX, f, indent=2)
                self.emit_log("[CACHE] model_index.json restauré automatiquement pour BriaFibo2Pipeline.")
                return True
            except Exception as exc:
                self.emit_log(f"[CACHE AVERTISSEMENT] Impossible de réparer model_index.json : {exc}")
                return False
        return not needs_repair

    def is_model_downloaded(self) -> bool:
        if not os.path.isdir(MODELS_DIR):
            return False
        if not self._ensure_valid_model_index():
            return False

        for root, _, filenames in os.walk(MODELS_DIR):
            for fname in filenames:
                if fname.endswith((".safetensors", ".bin")):
                    fpath = os.path.join(root, fname)
                    if os.path.exists(fpath) and os.path.getsize(fpath) > 1024 * 1024:
                        return True
        return False

    def download_model(self, hf_token: str = "", progress_callback=None):
        def notify(m):
            self.emit_log(m)
            if progress_callback:
                progress_callback(m)

        if self.is_model_downloaded():
            notify(f"[CACHE OK] Modèle {MODEL_ID} déjà présent dans {MODELS_DIR} — aucun téléchargement nécessaire.")
            return

        token = self.resolve_token(hf_token)
        if not token:
            raise PermissionError(
                "Authentification requise : connectez-vous via le navigateur Hugging Face puis collez votre jeton."
            )

        try:
            login(token=token, add_to_git_credential=False)
        except Exception:
            pass

        os.makedirs(MODELS_DIR, exist_ok=True)
        notify(f"[TÉLÉCHARGEMENT] Téléchargement authentifié de {MODEL_ID} vers {MODELS_DIR}...")

        dl_kwargs = {
            "repo_id": MODEL_ID,
            "local_dir": MODELS_DIR,
            "token": token,
        }
        try:
            if "resume_download" in inspect.signature(snapshot_download).parameters:
                dl_kwargs["resume_download"] = True
        except Exception:
            pass

        snapshot_download(**dl_kwargs)
        notify(f"[TÉLÉCHARGEMENT OK] Fichiers du modèle enregistrés dans {MODELS_DIR}.")

    def _attach_layer_offload(self, mod: torch.nn.Module, cuda_dev: torch.device, cpu_dev: torch.device) -> list:
        mod.to(cpu_dev)
        params = [(p, p.data) for p in mod.parameters()]
        buffers = [(b, b.data) for b in mod.buffers()]

        def pre_hook(_m, _args):
            for p, t in params:
                p.data = t.to(cuda_dev, non_blocking=True)
            for b, t in buffers:
                b.data = t.to(cuda_dev, non_blocking=True)

        def post_hook(_m, _args, _out):
            for p, t in params:
                p.data = t
            for b, t in buffers:
                b.data = t

        return [mod.register_forward_pre_hook(pre_hook), mod.register_forward_hook(post_hook)]

    def _configure_vram_fraction(self, profile: str, notify=None):
        """Plafonne l'allocateur PyTorch dans la VRAM dédiée pour empêcher le basculement lent vers la RAM partagée Windows (WDDM)."""
        if self.device != "cuda" or not torch.cuda.is_available():
            return
        try:
            total_mb, free_mb = self._query_vram_mb()
            if total_mb <= 0:
                return
            if profile == "6gb":
                cap_mb = min(free_mb * 0.88, 5530.0)
            elif profile == "8gb":
                cap_mb = min(free_mb * 0.88, 6400.0)
            else:
                cap_mb = free_mb * 0.90
            frac = max(0.35, min(0.92, cap_mb / total_mb))
            torch.cuda.set_per_process_memory_fraction(frac, 0)
            if notify:
                notify(f"[GPU {profile.upper()}] Plafond allocateur CUDA : {cap_mb:.0f} Mo ({frac * 100:.0f}% VRAM dédiée).")
        except Exception:
            pass

    def _upcast_vae_for_fp16(self):
        vae = getattr(self.pipe, "vae", None)
        first_param = next(vae.parameters(), None) if vae is not None else None
        if first_param is not None and first_param.dtype == torch.float16:
            vae.to(dtype=torch.float32)
            self.emit_log("[GPU] VAE Flux.2 en float32 pour éviter les sorties noires en FP16.")

    def _apply_memory_optimizations(self, profile: str, notify):
        if self.device != "cuda":
            self.offload_mode = "cpu"
            notify("[CPU] Aucun GPU CUDA actif : chargement sur processeur.")
            if hasattr(self.pipe, "to"):
                self.pipe.to("cpu")
            return

        # Les crops changent de dimensions d'une retouche à l'autre : désactiver cudnn.benchmark évite les re-calibrations coûteuses.
        try:
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

        self._configure_vram_fraction(profile, notify=notify)
        self._upcast_vae_for_fp16()

        try:
            if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
                if hasattr(self.pipe.vae, "enable_tiling"):
                    self.pipe.vae.enable_tiling()
                if hasattr(self.pipe.vae, "enable_slicing"):
                    self.pipe.vae.enable_slicing()
        except Exception as e:
            notify(f"[GPU] Optimisation VAE ignorée ({e}).")

        # Fibo-2.0-edit-turbo pèse ~23.7 Go (transformer 15.1 Go + text_encoder 8.3 Go + VAE 0.3 Go).
        # Placement direct uniquement si VRAM >= 24 Go ; sinon streaming asynchrone CUDA compatible 6 Go VRAM.
        total_vram, _ = self._query_vram_mb()
        if profile == "high" and total_vram >= 23000:
            try:
                notify("[GPU HIGH >=24Go] Placement direct sur CUDA pour vitesse maximale...")
                if hasattr(self.pipe, "to"):
                    self.pipe.to("cuda")
                self.offload_mode = "cuda_direct"
                return
            except Exception:
                pass

        if apply_group_offloading is not None:
            try:
                cuda_dev = torch.device("cuda")
                cpu_dev = torch.device("cpu")

                # 1. Qwen3-VL text_encoder (8.3 Go) : offload par couche (36 blocs ~185 Mo) au lieu de 713 feuilles -> ~14x plus rapide (~4.5s).
                te = getattr(self.pipe, "text_encoder", None)
                if te is not None:
                    if hasattr(te, "model") and hasattr(te.model, "language_model") and hasattr(te.model.language_model, "layers"):
                        te.to(cpu_dev)
                        if hasattr(te.model, "visual"):
                            self._custom_hooks += self._attach_layer_offload(te.model.visual, cuda_dev, cpu_dev)
                        if hasattr(te.model.language_model, "embed_tokens"):
                            self._custom_hooks += self._attach_layer_offload(te.model.language_model.embed_tokens, cuda_dev, cpu_dev)
                        if hasattr(te.model.language_model, "norm"):
                            self._custom_hooks += self._attach_layer_offload(te.model.language_model.norm, cuda_dev, cpu_dev)
                        if hasattr(te, "lm_head"):
                            self._custom_hooks += self._attach_layer_offload(te.lm_head, cuda_dev, cpu_dev)
                        if hasattr(te.model.language_model, "rotary_emb"):
                            te.model.language_model.rotary_emb.to(cuda_dev)
                        for layer in te.model.language_model.layers:
                            self._custom_hooks += self._attach_layer_offload(layer, cuda_dev, cpu_dev)
                    else:
                        apply_group_offloading(
                            te,
                            onload_device=cuda_dev,
                            offload_device=cpu_dev,
                            offload_type="leaf_level",
                            non_blocking=True,
                            use_stream=False,
                            low_cpu_mem_usage=True,
                        )

                # 2. BriaFibo2Transformer2DModel (15.1 Go, 44 blocs) : préchargement asynchrone par flux CUDA (use_stream=True).
                # record_stream=False évite la fragmentation mémoire multi-flux tout en recouvrant transfert PCIe et calcul GPU.
                if getattr(self.pipe, "transformer", None) is not None:
                    apply_group_offloading(
                        self.pipe.transformer,
                        onload_device=cuda_dev,
                        offload_device=cpu_dev,
                        offload_type="block_level",
                        num_blocks_per_group=1,
                        non_blocking=True,
                        use_stream=True,
                        record_stream=False,
                        low_cpu_mem_usage=True,
                    )

                # 3. VAE FLUX.2 (~310 Mo) maintenu sur CUDA pour des encodages/décodages instantanés.
                if getattr(self.pipe, "vae", None) is not None:
                    self.pipe.vae.to(cuda_dev)

                self.offload_mode = f"async_stream_offload ({profile})"
                notify(
                    f"[GPU {profile.upper()}] Streaming asynchrone CUDA activé "
                    f"(text_encoder=couches, transformer=stream async, VAE sur CUDA)."
                )
                return
            except Exception as err_grp:
                notify(f"[GPU] Streaming asynchrone indisponible ({err_grp}), repli sur sequential_cpu_offload...")

        try:
            notify(f"[GPU {profile.upper()}] Activation de enable_sequential_cpu_offload()...")
            self.pipe.enable_sequential_cpu_offload()
            self.offload_mode = f"sequential_cpu_offload ({profile})"
        except Exception as err_seq:
            notify(f"[GPU] sequential_cpu_offload indisponible ({err_seq}), repli sur model_cpu_offload...")
            self.pipe.enable_model_cpu_offload()
            self.offload_mode = f"model_cpu_offload ({profile})"

    def _resolve_pipeline_classes(self, source_id: str) -> list:
        candidates = []
        seen = set()

        def add_cls(cls):
            if cls is not None and id(cls) not in seen:
                seen.add(id(cls))
                candidates.append(cls)

        index_path = os.path.join(source_id, "model_index.json")
        if os.path.exists(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if isinstance(meta, dict):
                    declared = meta.get("_class_name") or meta.get("pipeline_class")
                    if isinstance(declared, str) and hasattr(diffusers, declared):
                        add_cls(getattr(diffusers, declared))
            except Exception:
                pass

        for name in (
            "BriaFibo2Pipeline",
            "AutoPipelineForImage2Image",
            "AutoPipelineForInpainting",
            "DiffusionPipeline",
        ):
            if hasattr(diffusers, name):
                add_cls(getattr(diffusers, name))

        add_cls(DiffusionPipeline)
        return candidates

    def _call_from_pretrained_with_kwargs(self, pipe_cls, source_id: str, dtype, token: str | None, local_only: bool):
        kwargs = {
            "dtype": dtype,
            "local_files_only": local_only,
        }
        if token and not local_only:
            kwargs["token"] = token

        try:
            return pipe_cls.from_pretrained(source_id, **kwargs)
        except TypeError as exc:
            retry_kwargs = dict(kwargs)
            if "dtype" in str(exc) and "dtype" in retry_kwargs:
                retry_kwargs["torch_dtype"] = retry_kwargs.pop("dtype")
            for opt_arg in ("token", "local_files_only"):
                if opt_arg in str(exc) and opt_arg in retry_kwargs:
                    retry_kwargs.pop(opt_arg, None)
            return pipe_cls.from_pretrained(source_id, **retry_kwargs)

    def _load_pipeline_with_fallback(self, source_id: str, dtype, token: str | None, local_only: bool, notify):
        pipe_classes = self._resolve_pipeline_classes(source_id)
        last_err = None

        for idx, pipe_cls in enumerate(pipe_classes):
            cls_name = getattr(pipe_cls, "__name__", str(pipe_cls))
            if idx > 0:
                notify(f"[CHARGEMENT] Essai avec {cls_name}...")
            try:
                return self._call_from_pretrained_with_kwargs(pipe_cls, source_id, dtype, token, local_only)
            except Exception as exc:
                last_err = exc
                if dtype != torch.float16 and self.device == "cuda":
                    try:
                        notify(f"[CHARGEMENT] {cls_name} : repli sur torch.float16...")
                        return self._call_from_pretrained_with_kwargs(
                            pipe_cls, source_id, torch.float16, token, local_only
                        )
                    except Exception as exc_fp16:
                        last_err = exc_fp16
                notify(f"[CHARGEMENT] {cls_name} indisponible ({type(last_err).__name__}: {last_err})")

        if last_err is not None:
            raise last_err
        raise RuntimeError("Aucune classe de pipeline diffusers n'a pu être instanciée.")

    def load_model(self, hf_token: str = "", progress_callback=None, settings: dict | None = None):
        def notify(m):
            self.emit_log(m)
            if progress_callback:
                progress_callback(m)

        cfg_settings = settings or load_config().get("settings", DEFAULT_SETTINGS)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_name = self._resolve_gpu_name()
        total_vram, free_vram = self._query_vram_mb()
        profile = self._resolve_vram_profile(cfg_settings, total_vram)
        dtype = self._select_cuda_dtype(profile)

        if self.pipe is not None:
            if self.loaded_dtype == dtype:
                notify("[MOTEUR] Modèle déjà chargé en mémoire avec la précision optimale.")
                return
            notify(f"[MOTEUR] Rechargement du modèle ({self.loaded_dtype} -> {dtype})...")
            self.unload_model()

        vram_txt = f"{total_vram:.0f} Mo total / {free_vram:.0f} Mo libres" if total_vram > 0 else "N/A"
        notify(f"[MATÉRIEL] {self.device.upper()} ({self.gpu_name}) | VRAM : {vram_txt} | Profil : {profile.upper()}")

        local_ready = self.is_model_downloaded()
        if local_ready:
            notify(f"[CACHE OK] Modèle détecté sur disque ({MODELS_DIR}). Chargement local direct sans téléchargement.")
        else:
            self.download_model(hf_token=hf_token, progress_callback=notify)
            local_ready = self.is_model_downloaded()

        token = self.resolve_token(hf_token)
        notify(f"[CHARGEMENT] Instanciation du pipeline diffusers ({dtype})...")

        self.pipe = self._load_pipeline_with_fallback(
            source_id=MODELS_DIR if local_ready else MODEL_ID,
            dtype=dtype,
            token=token,
            local_only=local_ready,
            notify=notify,
        )
        self.loaded_dtype = getattr(getattr(self.pipe, "dtype", None), "__name__", None) or dtype
        self._apply_memory_optimizations(profile=profile, notify=notify)
        self.flush_vram("après chargement")
        notify(f"[PRÊT] Moteur FIBO 2.0 Edit Turbo opérationnel (profil {profile.upper()}, {self.loaded_dtype}).")

    def _find_mask_component_boxes(
        self,
        binary_mask_l: Image.Image,
        orig_w: int,
        orig_h: int,
    ) -> list[tuple[int, int, int, int]]:
        """Détecte les boîtes englobantes des zones masquées disjointes via une grille rapide (sans dépendance externe)."""
        grid_max = 160
        scale = min(1.0, grid_max / float(max(orig_w, orig_h)))
        gw = max(1, int(round(orig_w * scale)))
        gh = max(1, int(round(orig_h * scale)))

        small = np.array(binary_mask_l.resize((gw, gh), Image.Resampling.BOX), dtype=np.uint8) > 0
        if not np.any(small):
            return []

        visited = np.zeros_like(small, dtype=bool)
        boxes = []
        ys, xs = np.nonzero(small)
        for sy, sx in zip(ys.tolist(), xs.tolist()):
            if visited[sy, sx]:
                continue
            stack = [(sy, sx)]
            visited[sy, sx] = True
            min_y = max_y = sy
            min_x = max_x = sx
            while stack:
                cy, cx = stack.pop()
                if cy < min_y:
                    min_y = cy
                if cy > max_y:
                    max_y = cy
                if cx < min_x:
                    min_x = cx
                if cx > max_x:
                    max_x = cx
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < gh and 0 <= nx < gw and small[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            x0 = max(0, int(np.floor(min_x / scale)) - 2)
            y0 = max(0, int(np.floor(min_y / scale)) - 2)
            x1 = min(orig_w, int(np.ceil((max_x + 1) / scale)) + 2)
            y1 = min(orig_h, int(np.ceil((max_y + 1) / scale)) + 2)
            boxes.append((x0, y0, x1, y1))

        return boxes

    def _expand_mask_box(
        self,
        bbox: tuple[int, int, int, int],
        orig_w: int,
        orig_h: int,
        padding: int,
        feather_px: int,
    ) -> tuple[int, int, int, int]:
        mx0, my0, mx1, my1 = bbox
        mask_w = max(1, mx1 - mx0)
        mask_h = max(1, my1 - my0)
        pad = max(24, int(padding), int(feather_px) * 2)

        dyn_pad_x = max(pad, int(mask_w * 0.30))
        dyn_pad_y = max(pad, int(mask_h * 0.30))

        x0 = max(0, mx0 - dyn_pad_x)
        y0 = max(0, my0 - dyn_pad_y)
        x1 = min(orig_w, mx1 + dyn_pad_x)
        y1 = min(orig_h, my1 + dyn_pad_y)

        # Petite retouche locale -> 256px suffit (très rapide) ; zone moyenne -> 384px pour le contexte.
        desired_min = 256 if max(mask_w, mask_h) <= 128 else 384
        min_box = min(desired_min, orig_w, orig_h)
        cur_w = x1 - x0
        cur_h = y1 - y0
        if cur_w < min_box:
            expand = min_box - cur_w
            x0 = max(0, x0 - expand // 2)
            x1 = min(orig_w, x0 + min_box)
            x0 = max(0, x1 - min_box)
        if cur_h < min_box:
            expand = min_box - cur_h
            y0 = max(0, y0 - expand // 2)
            y1 = min(orig_h, y0 + min_box)
            y0 = max(0, y1 - min_box)

        return int(x0), int(y0), int(x1), int(y1)

    @staticmethod
    def _merge_overlapping_boxes(boxes: list[tuple[int, int, int, int]], margin: int = 16) -> list[tuple[int, int, int, int]]:
        merged = list(boxes)
        changed = True
        while changed:
            changed = False
            out = []
            while merged:
                a = merged.pop(0)
                ax0, ay0, ax1, ay1 = a
                i = 0
                while i < len(merged):
                    bx0, by0, bx1, by1 = merged[i]
                    if not (ax1 + margin < bx0 or bx1 + margin < ax0 or ay1 + margin < by0 or by1 + margin < ay0):
                        ax0 = min(ax0, bx0)
                        ay0 = min(ay0, by0)
                        ax1 = max(ax1, bx1)
                        ay1 = max(ay1, by1)
                        merged.pop(i)
                        changed = True
                    else:
                        i += 1
                out.append((ax0, ay0, ax1, ay1))
            merged = out
        return merged

    def _compute_smart_crop_boxes(
        self,
        orig_w: int,
        orig_h: int,
        mask_l: Image.Image,
        has_mask_pixels: bool,
        padding: int,
        feather_px: int = 12,
    ) -> tuple[list[tuple[int, int, int, int]], bool]:
        if not has_mask_pixels:
            return [(0, 0, orig_w, orig_h)], False

        binary_mask = mask_l.point(lambda v: 255 if v >= 128 else 0)
        union_bbox = binary_mask.getbbox()
        if not union_bbox:
            return [(0, 0, orig_w, orig_h)], False

        full_area = max(1, orig_w * orig_h)
        union_crop = self._expand_mask_box(union_bbox, orig_w, orig_h, padding, feather_px)
        union_area = max(1, (union_crop[2] - union_crop[0]) * (union_crop[3] - union_crop[1]))

        comp_boxes = self._find_mask_component_boxes(binary_mask, orig_w, orig_h)
        if 1 < len(comp_boxes) <= 4:
            expanded = [self._expand_mask_box(b, orig_w, orig_h, padding, feather_px) for b in comp_boxes]
            merged = self._merge_overlapping_boxes(expanded, margin=24)
            if 1 < len(merged) <= 4:
                sum_area = sum(max(1, (x1 - x0) * (y1 - y0)) for x0, y0, x1, y1 in merged)
                if sum_area < union_area * 0.55:
                    return merged, True

        if (union_area / full_area) > 0.92:
            return [(0, 0, orig_w, orig_h)], False

        return [union_crop], True

    def _fit_target_dimensions(self, box_w: int, box_h: int, max_side: int) -> tuple[int, int]:
        w, h = box_w, box_h
        if max_side > 0 and max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            w = max(64, int(round(w * scale)))
            h = max(64, int(round(h * scale)))
        tw = max(64, (w // 16) * 16)
        th = max(64, (h // 16) * 16)
        return tw, th

    def _build_masked_overlay(self, img_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
        arr = np.array(img_rgb, dtype=np.uint8)
        m_arr = np.array(mask_l, dtype=np.uint8) > 127
        if not np.any(m_arr):
            return img_rgb

        highlight = np.zeros_like(arr)
        highlight[..., 0] = 255
        highlight[..., 1] = 40
        highlight[..., 2] = 40

        blended = arr.copy()
        blended[m_arr] = (0.45 * arr[m_arr] + 0.55 * highlight[m_arr]).astype(np.uint8)
        out = Image.fromarray(blended, mode="RGB")

        mask_pil = Image.fromarray((m_arr.astype(np.uint8) * 255), mode="L")
        draw = ImageDraw.Draw(out)
        bbox = mask_pil.getbbox()
        if bbox:
            draw.rectangle(bbox, outline=(255, 60, 60), width=2)
        return out

    def _paste_back_seamlessly(
        self,
        full_rgb: Image.Image,
        edited_patch_rgb: Image.Image,
        crop_box: tuple[int, int, int, int],
        mask_crop_l: Image.Image,
        feather_px: int,
    ) -> Image.Image:
        x0, y0, x1, y1 = crop_box
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)

        patch = edited_patch_rgb.convert("RGB").resize((bw, bh), Image.Resampling.LANCZOS)
        base_crop = full_rgb.crop((x0, y0, x1, y1)).convert("RGB")

        m_resized = mask_crop_l.convert("L").resize((bw, bh), Image.Resampling.BILINEAR)
        m_arr = (np.array(m_resized, dtype=np.float32) > 127.0).astype(np.float32) * 255.0
        alpha_img = Image.fromarray(m_arr.astype(np.uint8), mode="L")

        blur_rad = max(1, int(feather_px))
        alpha_img = alpha_img.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(radius=blur_rad))
        alpha_norm = (np.array(alpha_img, dtype=np.float32) / 255.0)[..., None]

        base_arr = np.array(base_crop, dtype=np.float32)
        patch_arr = np.array(patch, dtype=np.float32)
        merged_arr = (patch_arr * alpha_norm + base_arr * (1.0 - alpha_norm)).clip(0, 255).astype(np.uint8)

        out_full = full_rgb.copy()
        out_full.paste(Image.fromarray(merged_arr, mode="RGB"), (x0, y0))
        return out_full

    def _extract_output_image(self, out) -> Image.Image:
        if hasattr(out, "images") and out.images:
            return out.images[0]
        if isinstance(out, (list, tuple)) and len(out) > 0:
            first = out[0]
            if isinstance(first, list) and first:
                return first[0]
            if isinstance(first, Image.Image):
                return first
        if isinstance(out, Image.Image):
            return out
        raise RuntimeError(f"Format de sortie inattendu du pipeline : {type(out).__name__}")

    def _get_pipe_call_params(self) -> set[str]:
        for target in (self.pipe.__call__, type(self.pipe).__call__):
            try:
                params = set(inspect.signature(target).parameters.keys())
                params.discard("self")
                if params and not (params == {"args", "kwargs"}):
                    return params
            except Exception:
                pass
        return {"prompt", "image", "num_inference_steps", "guidance_scale", "width", "height", "generator"}

    def _get_local_fonts(self) -> list[str]:
        fonts_dir = os.path.join(MODELS_DIR, "fonts")
        candidates = ("Aileron-Regular.otf", "DejaVuSans.ttf")
        return [
            os.path.join(fonts_dir, fname)
            for fname in candidates
            if os.path.exists(os.path.join(fonts_dir, fname))
        ]

    def _call_pipeline_adaptively(
        self,
        prompt: str,
        img_rgb: Image.Image,
        mask_l: Image.Image,
        steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        generator,
        reference_count: int = 1,
    ) -> Image.Image:
        accepted = self._get_pipe_call_params()

        masked_img = self._build_masked_overlay(img_rgb, mask_l)
        has_mask_pixels = bool(np.any(np.array(mask_l, dtype=np.uint8) > 127))

        mask_param_name = None
        for candidate in ("mask_image", "mask", "inpaint_mask", "edit_mask"):
            if candidate in accepted:
                mask_param_name = candidate
                break

        input_image = img_rgb if (mask_param_name or not has_mask_pixels) else masked_img

        kwargs = {}
        if "prompt" in accepted:
            kwargs["prompt"] = prompt

        for img_key in ("image", "input_image", "init_image", "source_image"):
            if img_key in accepted:
                count = max(1, min(4, int(reference_count)))
                kwargs[img_key] = [input_image] * count if count > 1 else input_image
                break

        # Ne passer le masque natif que si l'utilisateur a réellement peint une zone.
        # Un masque entièrement noir indiquerait au pipeline de ne rien modifier lors d'une retouche globale.
        if mask_param_name and has_mask_pixels:
            kwargs[mask_param_name] = mask_l

        if "fonts" in accepted:
            local_fonts = self._get_local_fonts()
            if local_fonts:
                kwargs["fonts"] = local_fonts

        for step_key in ("num_inference_steps", "steps", "num_steps"):
            if step_key in accepted:
                kwargs[step_key] = int(steps)
                break

        for cfg_key in ("guidance_scale", "true_cfg_scale", "cfg_scale"):
            if cfg_key in accepted:
                kwargs[cfg_key] = float(guidance_scale)
                break

        if "width" in accepted:
            kwargs["width"] = int(width)
        if "height" in accepted:
            kwargs["height"] = int(height)
        if "generator" in accepted and generator is not None:
            kwargs["generator"] = generator

        with torch.inference_mode():
            try:
                out = self.pipe(**kwargs)
            except TypeError as exc:
                retry_kwargs = dict(kwargs)
                changed = False
                if "image" in retry_kwargs and isinstance(retry_kwargs["image"], Image.Image) and "list" in str(exc).lower():
                    retry_kwargs["image"] = [retry_kwargs["image"]]
                    changed = True
                for opt_key in ("width", "height", "fonts"):
                    if opt_key in retry_kwargs and opt_key in str(exc):
                        retry_kwargs.pop(opt_key, None)
                        changed = True
                if not changed:
                    raise
                out = self.pipe(**retry_kwargs)

        return self._extract_output_image(out)

    def edit_image(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "",
        steps: int | None = None,
        seed: int | None = None,
        reference_count: int = 1,
        settings: dict | None = None,
    ) -> Image.Image:
        t0 = time.perf_counter()

        cfg_all = load_config()
        merged_settings = {**DEFAULT_SETTINGS, **(cfg_all.get("settings") or {}), **(settings or {})}

        total_vram, free_vram = self._query_vram_mb()
        profile = self._resolve_vram_profile(merged_settings, total_vram)
        max_side = self._resolve_max_edit_side(merged_settings, profile)
        crop_padding = int(merged_settings.get("crop_padding") or 64)
        feather_px = int(merged_settings.get("feather_px") or 12)
        eff_steps = max(2, min(12, int(steps if steps is not None else (merged_settings.get("steps") or 4))))
        eff_cfg = float(merged_settings.get("guidance_scale") if merged_settings.get("guidance_scale") is not None else 1.0)
        eff_seed = seed if seed is not None else merged_settings.get("seed")

        if self.pipe is None:
            self.load_model(settings=merged_settings)

        orig_w, orig_h = image.size
        full_rgb = image.convert("RGB")
        mask_full_l = mask.convert("L").resize((orig_w, orig_h), Image.Resampling.NEAREST)

        has_mask_pixels = bool(np.any(np.array(mask_full_l, dtype=np.uint8) > 127))
        clean_prompt = (prompt or "").strip()

        if not has_mask_pixels and not clean_prompt:
            raise ValueError("Veuillez colorier une zone au pinceau ou saisir une description de retouche.")

        crop_boxes, is_cropped = self._compute_smart_crop_boxes(
            orig_w=orig_w,
            orig_h=orig_h,
            mask_l=mask_full_l,
            has_mask_pixels=has_mask_pixels,
            padding=crop_padding,
            feather_px=feather_px,
        )

        # Adaptation dynamique de la résolution de calcul selon la VRAM réellement libre
        user_forced_side = int(merged_settings.get("max_edit_side") or 0)
        adaptive_max_side = max_side
        if user_forced_side < 256 and self.device == "cuda":
            _, live_free_mb = self._query_vram_mb()
            if live_free_mb > 0:
                if live_free_mb < 4200:
                    adaptive_max_side = min(adaptive_max_side, 512)
                elif live_free_mb < 5600:
                    adaptive_max_side = min(adaptive_max_side, 640)

        accepted = self._get_pipe_call_params()
        has_native_mask = any(k in accepted for k in ("mask_image", "mask", "inpaint_mask", "edit_mask"))

        if has_mask_pixels and not has_native_mask:
            if clean_prompt:
                effective_prompt = (
                    f"In the red-highlighted masked area, {clean_prompt}. "
                    "Keep the rest of the image completely unchanged and blend seamlessly."
                )
            else:
                effective_prompt = (
                    "Remove the object or content inside the red-highlighted masked area and fill in the background naturally, "
                    "matching surrounding textures, lighting, and perspective. Keep the rest of the image unchanged."
                )
        else:
            effective_prompt = clean_prompt or (
                "Remove the masked object or region cleanly and fill in the background naturally "
                "matching surrounding textures, lighting, and perspective."
            )

        if self.device == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        final_img = full_rgb.copy()
        region_summaries = []

        for box_idx, (x0, y0, x1, y1) in enumerate(crop_boxes):
            if is_cropped:
                work_img = final_img.crop((x0, y0, x1, y1))
                work_mask = mask_full_l.crop((x0, y0, x1, y1))
                region_label = f"Crop {x1-x0}x{y1-y0}px @ ({x0},{y0})"
            else:
                work_img = final_img
                work_mask = mask_full_l
                region_label = f"Image complète {orig_w}x{orig_h}px"

            target_w, target_h = self._fit_target_dimensions(work_img.width, work_img.height, max_side=adaptive_max_side)
            img_in = work_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            mask_in = work_mask.resize((target_w, target_h), Image.Resampling.NEAREST)

            # Optimisation adaptative VAE + vision Qwen3-VL selon la taille du crop envoyé à FIBO
            max_dim = max(target_w, target_h)
            if getattr(self.pipe, "vae", None) is not None:
                try:
                    if max_dim <= 512:
                        if hasattr(self.pipe.vae, "disable_tiling"):
                            self.pipe.vae.disable_tiling()
                        if hasattr(self.pipe.vae, "disable_slicing"):
                            self.pipe.vae.disable_slicing()
                    else:
                        if hasattr(self.pipe.vae, "enable_tiling"):
                            self.pipe.vae.enable_tiling()
                        if hasattr(self.pipe.vae, "enable_slicing"):
                            self.pipe.vae.enable_slicing()
                except Exception:
                    pass

            if hasattr(self.pipe, "text_encoder_image_size"):
                self.pipe.text_encoder_image_size = min(768, max(256, max_dim))

            generator = None
            if eff_seed is not None and str(eff_seed).strip() != "":
                try:
                    generator = torch.Generator(device="cpu").manual_seed(int(eff_seed) + box_idx)
                except Exception:
                    generator = None

            self.emit_log(
                f"[INFÉRENCE {box_idx + 1}/{len(crop_boxes)}] {region_label} -> {target_w}x{target_h}px | "
                f"steps={eff_steps} | cfg={eff_cfg} | références={max(1, min(4, int(reference_count)))} | "
                f"masque={'oui' if has_mask_pixels else 'non'}"
            )
            region_summaries.append(f"{region_label} -> {target_w}x{target_h}px")

            raw_out = self._call_pipeline_adaptively(
                prompt=effective_prompt,
                img_rgb=img_in,
                mask_l=mask_in,
                steps=eff_steps,
                guidance_scale=eff_cfg,
                width=target_w,
                height=target_h,
                generator=generator,
                reference_count=reference_count,
            )

            if has_mask_pixels:
                final_img = self._paste_back_seamlessly(
                    full_rgb=final_img,
                    edited_patch_rgb=raw_out,
                    crop_box=(x0, y0, x1, y1),
                    mask_crop_l=work_mask,
                    feather_px=feather_px,
                )
            else:
                final_img = raw_out.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

        self.last_edit_region = " + ".join(region_summaries)

        if self.device == "cuda" and torch.cuda.is_available():
            try:
                self.last_peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
            except Exception:
                pass

        self.last_inference_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        peak_info = f" | Pic VRAM : {self.last_peak_vram_mb} Mo" if self.last_peak_vram_mb is not None else ""
        self.emit_log(f"[INFÉRENCE OK] Terminé en {self.last_inference_ms / 1000.0:.1f}s{peak_info}.")
        self.flush_vram("fin inférence")
        if merged_settings.get("auto_unload_idle"):
            self.unload_model()
        return final_img

    def _load_molmo_encoder(self, encoder_choice: str):
        model_id = "allenai/Molmo2-8B" if encoder_choice == "molmo2-8b" else "allenai/Molmo2-4B"
        if self.molmo_model is not None and self.molmo_model_id == model_id:
            return self.molmo_processor, self.molmo_model

        if self.molmo_model is not None:
            self.molmo_model = None
            self.molmo_processor = None
            self.molmo_model_id = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Transformers ne fournit pas encore les classes Molmo2 nécessaires. "
                "Installez une version récente de transformers."
            ) from exc

        token = self.resolve_token()
        kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        if token:
            kwargs["token"] = token

        self.emit_log(f"[AIDE PROMPT] Chargement de {model_id}...")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=token)
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        except TypeError:
            kwargs.pop("dtype", None)
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)

        model.eval()
        self.molmo_processor = processor
        self.molmo_model = model
        self.molmo_model_id = model_id
        return processor, model

    def _describe_with_molmo(
        self,
        vis_img: Image.Image,
        region_label: str,
        instruction: str,
        encoder_choice: str,
        t0: float,
        max_new_tokens: int,
    ) -> dict:
        processor, model = self._load_molmo_encoder(encoder_choice)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": vis_img},
                {"type": "text", "text": instruction},
            ],
        }]

        self.emit_log(
            f"[AIDE PROMPT] Analyse Molmo2 sur {region_label} "
            f"({vis_img.width}x{vis_img.height}px)..."
        )
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        target_dev = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        inputs = {key: value.to(target_dev) for key, value in inputs.items()}

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max(12, min(96, int(max_new_tokens))),
                do_sample=False,
            )
        prompt_length = inputs["input_ids"].size(1)
        generated_tokens = generated_ids[0, prompt_length:]
        extracted = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        if len(extracted) >= 2 and extracted[0] == extracted[-1] and extracted[0] in ('"', "'"):
            extracted = extracted[1:-1].strip()

        duration_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        peak_mb = None
        if torch.cuda.is_available():
            peak_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)

        self.emit_log(f"[AIDE PROMPT OK] Extrait avec Molmo2 en {duration_ms / 1000.0:.1f}s : {extracted}")
        return {
            "extracted_prompt": extracted,
            "region": region_label,
            "duration_ms": duration_ms,
            "peak_vram_mb": peak_mb,
        }

    def describe_masked_region(
        self,
        image: Image.Image,
        mask: Image.Image | None = None,
        language: str = "fr",
        max_new_tokens: int = 48,
        settings: dict | None = None,
    ) -> dict:
        """Extrait un prompt descriptif uniquement à partir du crop de la zone masquée."""
        t0 = time.perf_counter()
        if self.pipe is None:
            self.load_model()

        orig_w, orig_h = image.size
        full_rgb = image.convert("RGB")

        bbox = None
        if mask is not None:
            mask_full_l = mask.convert("L").resize((orig_w, orig_h), Image.Resampling.NEAREST)
            binary_mask = mask_full_l.point(lambda v: 255 if v >= 128 else 0)
            bbox = binary_mask.getbbox()

        if bbox is not None:
            mx0, my0, mx1, my1 = bbox
            vis_img = full_rgb.crop((mx0, my0, mx1, my1))
            region_label = f"Crop exact {mx1 - mx0}x{my1 - my0}px @ ({mx0},{my0})"
        else:
            vis_img = full_rgb.copy()
            region_label = f"Crop envoyé {orig_w}x{orig_h}px"

        vis_img.thumbnail((384, 384), Image.Resampling.LANCZOS)

        lang = (language or "fr").strip().lower()
        if lang == "en":
            instruction = (
                "Describe concisely in one short English sentence the main subject or element in this image region "
                "(appearance, color, material, lighting), so the user can edit this description as an image-editing prompt. "
                "Output only the description, no preamble."
            )
        else:
            instruction = (
                "Décris de façon concise en une seule phrase courte en français l'objet ou l'élément principal visible "
                "dans cette zone d'image (apparence, couleur, matière, éclairage), afin que l'utilisateur puisse "
                "modifier cette description comme prompt de retouche photo. Réponds uniquement par la description."
            )

        encoder_choice = (settings or {}).get("prompt_encoder", "qwen3-vl")
        if encoder_choice.startswith("molmo2"):
            return self._describe_with_molmo(
                vis_img=vis_img,
                region_label=region_label,
                instruction=instruction,
                encoder_choice=encoder_choice,
                t0=t0,
                max_new_tokens=max_new_tokens,
            )

        te = getattr(self.pipe, "text_encoder", None)
        proc = getattr(self.pipe, "processor", None)
        if te is None or proc is None:
            raise RuntimeError("L'encodeur vision-langage (Qwen3-VL) n'est pas disponible dans le pipeline.")

        prompt_txt = (
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>"
            f"{instruction}"
            "<|im_end|>\n<|im_start|>assistant\n"
        )

        self.emit_log(f"[AIDE PROMPT] Analyse Qwen3-VL sur {region_label} ({vis_img.width}x{vis_img.height}px)...")
        target_dev = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
        if target_dev == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        inputs = proc(text=[prompt_txt], images=[vis_img], return_tensors="pt").to(target_dev)
        eos_id = getattr(proc.tokenizer, "eos_token_id", None)
        generated_ids = []

        # Vérifier si forward accepte `cache` (transformers >=5.x) ou `past_key_values`
        te_params = set()
        try:
            te_params = set(inspect.signature(te.forward).parameters.keys())
        except Exception:
            pass
        cache_kw = "cache" if "cache" in te_params else "past_key_values"

        with torch.inference_mode():
            out = te(**inputs, use_cache=True)
            past = getattr(out, "past_key_values", None)
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            for _ in range(max(12, min(96, int(max_new_tokens)))):
                tok_id = int(next_tok.item())
                if eos_id is not None and tok_id == eos_id:
                    break
                generated_ids.append(tok_id)
                step_kwargs = {"input_ids": next_tok, "use_cache": True}
                if past is not None:
                    step_kwargs[cache_kw] = past
                out = te(**step_kwargs)
                past = getattr(out, "past_key_values", None)
                next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        extracted = proc.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        # Nettoyage des guillemets éventuels autour de la phrase
        if len(extracted) >= 2 and extracted[0] == extracted[-1] and extracted[0] in ('"', "'"):
            extracted = extracted[1:-1].strip()

        duration_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        peak_mb = None
        if target_dev == "cuda":
            try:
                peak_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
            except Exception:
                pass

        self.emit_log(f"[AIDE PROMPT OK] Extrait en {duration_ms / 1000.0:.1f}s : {extracted}")
        self.flush_vram("fin extraction prompt")
        return {
            "extracted_prompt": extracted,
            "region": region_label,
            "duration_ms": duration_ms,
            "peak_vram_mb": peak_mb,
        }

