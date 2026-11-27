import io
import os
import time
import base64
import threading
import webbrowser
from collections import deque
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image
from backend.engine import (
    FiboTurboEngine,
    MODEL_ID,
    DEFAULT_SETTINGS,
    load_config,
    save_config,
)
from backend.updater import Updater

BUNDLE_DIR = os.environ.get(
    "LAMPYRE_BUNDLE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
FRONTEND_HTML = os.path.join(BUNDLE_DIR, "frontend", "index.html")
FRONTEND_LOGO = os.path.join(BUNDLE_DIR, "frontend", "logo.png")

HF_LOGIN_URL = "https://huggingface.co/login"
HF_MODEL_URL = f"https://huggingface.co/{MODEL_ID}"
HF_TOKENS_URL = "https://huggingface.co/settings/tokens"

log_buffer = deque(maxlen=300)
_log_lock = threading.Lock()


def push_log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with _log_lock:
        log_buffer.append(line)
    print(line, flush=True)


engine = FiboTurboEngine(log_callback=push_log)
updater = Updater(BUNDLE_DIR)

status_state = {
    "state": "loading",
    "processing": False,
    "message": "Serveur connecté. Vérification du cache local et de l'authentification...",
    "gpu": engine.gpu_name,
    "cuda_available": engine.device == "cuda",
    "hf_login_url": HF_LOGIN_URL,
    "hf_model_url": HF_MODEL_URL,
    "hf_tokens_url": HF_TOKENS_URL,
}

_worker_lock = threading.Lock()
_inference_lock = threading.Lock()


class EditRequest(BaseModel):
    image_b64: str
    mask_b64: str
    prompt: str = ""
    steps: int | None = None
    seed: int | None = None
    reference_count: int = 1
    settings: dict | None = None


class DescribeRequest(BaseModel):
    image_b64: str
    mask_b64: str = ""
    language: str = "fr"


class AuthRequest(BaseModel):
    hf_token: str


class SettingsRequest(BaseModel):
    settings: dict
    reload_model: bool = False


def _update_status(msg: str):
    status_state["message"] = msg
    status_state["gpu"] = engine.gpu_name
    status_state["cuda_available"] = engine.device == "cuda"


def _is_auth_error(err_text: str) -> bool:
    low = err_text.lower()
    return (
        "authentification requise" in low
        or "401" in low
        or "403" in low
        or "gated" in low
        or "access to model" in low
        or "unauthorized" in low
        or "token" in low
        or "permissionerror" in low
    )


def _init_worker(explicit_token: str = "", force_reload: bool = False):
    with _worker_lock:
        cfg = load_config()
        current_settings = cfg.get("settings", dict(DEFAULT_SETTINGS))
        status_state["state"] = "loading"

        if force_reload and engine.pipe is not None:
            engine.unload_model()

        if engine.is_model_downloaded():
            push_log("[CACHE OK] Modèle détecté sur disque : chargement local direct.")
            status_state["message"] = "Modèle détecté sur disque : chargement local..."
        else:
            resolved = engine.resolve_token(explicit_token)
            if not resolved:
                status_state["state"] = "needs_auth"
                msg = (
                    "Authentification Hugging Face requise pour télécharger le modèle. "
                    "Ouvrez Hugging Face dans votre navigateur, connectez-vous, acceptez l'accès au modèle, "
                    "puis collez votre jeton (Read) ci-dessous."
                )
                status_state["message"] = msg
                push_log(f"[AUTH REQUISE] {msg}")
                return

            ok, auth_msg = engine.verify_token(resolved)
            push_log(f"[AUTH] {auth_msg}")
            if not ok:
                status_state["state"] = "needs_auth"
                status_state["message"] = auth_msg
                return

        try:
            engine.load_model(
                hf_token=explicit_token,
                progress_callback=_update_status,
                settings=current_settings,
            )
            status_state["state"] = "ready"
            status_state["gpu"] = engine.gpu_name
            status_state["cuda_available"] = engine.device == "cuda"
            status_state["message"] = f"Prêt — {engine.gpu_name}"
            push_log(f"[INIT OK] Application prête sur {engine.gpu_name}.")
        except Exception as exc:
            err_text = f"{type(exc).__name__}: {exc}"
            push_log(f"[ERREUR] {err_text}")
            if _is_auth_error(err_text):
                status_state["state"] = "needs_auth"
                status_state["message"] = (
                    f"Accès refusé ou authentification incomplète ({err_text}). "
                    "Vérifiez votre connexion navigateur à Hugging Face, l'acceptation des conditions du modèle et votre jeton."
                )
            else:
                status_state["state"] = "error"
                status_state["message"] = f"Erreur : {err_text}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    push_log("[SERVEUR] FastAPI démarré.")
    threading.Thread(target=_init_worker, args=(cfg.get("hf_token", ""), False), daemon=True).start()
    threading.Thread(target=updater.check, daemon=True).start()
    yield
    push_log("[SERVEUR] Arrêt de FastAPI.")


app = FastAPI(title="Lampyre", lifespan=lifespan)


@app.get("/")
def serve_index():
    return FileResponse(
        FRONTEND_HTML,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/logo.png")
def serve_logo():
    return FileResponse(
        FRONTEND_LOGO,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/status")
def get_status():
    cfg = load_config()
    current_settings = cfg.get("settings", dict(DEFAULT_SETTINGS))
    telemetry = engine.get_vram_and_model_info(settings=current_settings)
    with _log_lock:
        recent_logs = list(log_buffer)
    return JSONResponse(
        content={
            **status_state,
            "settings": current_settings,
            "telemetry": telemetry,
            "logs": recent_logs,
            "update": updater.status(),
        },
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@app.get("/api/update/check")
def check_for_update():
    return updater.check()


@app.post("/api/update/install")
def install_update():
    if engine.pipe is not None:
        engine.flush_vram("mise à jour Lampyre")
    return updater.download_and_apply()


@app.post("/api/update/restart")
def restart_after_update():
    return updater.restart()


@app.post("/api/settings")
def update_settings(req: SettingsRequest):
    cfg = load_config()
    merged = {**DEFAULT_SETTINGS, **(cfg.get("settings") or {}), **(req.settings or {})}
    cfg["settings"] = merged
    save_config(cfg)
    push_log(f"[RÉGLAGES] Paramètres enregistrés : {merged}")

    if req.reload_model:
        threading.Thread(target=_init_worker, args=(cfg.get("hf_token", ""), True), daemon=True).start()

    return {
        "ok": True,
        "settings": merged,
        "telemetry": engine.get_vram_and_model_info(settings=merged),
    }


@app.post("/api/flush-vram")
def flush_vram_endpoint():
    engine.flush_vram("action utilisateur")
    cfg = load_config()
    return {
        "ok": True,
        "telemetry": engine.get_vram_and_model_info(settings=cfg.get("settings")),
    }


@app.post("/api/open-hf-browser")
def open_hf_browser():
    push_log(f"[NAVIGATEUR] Ouverture de {HF_LOGIN_URL} et {HF_MODEL_URL}")
    webbrowser.open(HF_LOGIN_URL)
    webbrowser.open(HF_MODEL_URL)
    return {"ok": True}


@app.post("/api/auth")
def submit_auth(req: AuthRequest):
    token = (req.hf_token or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="Veuillez coller votre jeton Hugging Face.")

    ok, auth_msg = engine.verify_token(token)
    push_log(f"[AUTH] {auth_msg}")
    if not ok:
        raise HTTPException(status_code=401, detail=auth_msg)

    cfg = load_config()
    cfg["hf_token"] = token
    save_config(cfg)
    push_log("[AUTH] Jeton enregistré localement. Démarrage du téléchargement...")
    threading.Thread(target=_init_worker, args=(token, False), daemon=True).start()
    return {"ok": True, "message": auth_msg}


@app.post("/api/edit")
def edit_image_endpoint(req: EditRequest):
    if status_state["state"] != "ready" and engine.pipe is None:
        push_log("[API /edit] Requête rejetée : modèle pas encore prêt.")
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore prêt.")

    try:
        push_log("[API /edit] Réception d'une demande de retouche...")
        img_part = req.image_b64.split(",", 1)[1] if "," in req.image_b64 else req.image_b64
        mask_part = req.mask_b64.split(",", 1)[1] if "," in req.mask_b64 else req.mask_b64

        image = Image.open(io.BytesIO(base64.b64decode(img_part)))
        mask = Image.open(io.BytesIO(base64.b64decode(mask_part)))

        cfg = load_config()
        merged_settings = {**DEFAULT_SETTINGS, **(cfg.get("settings") or {}), **(req.settings or {})}

        if not _inference_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Une retouche est déjà en cours. Veuillez patienter.")

        status_state["processing"] = True
        try:
            output_img = engine.edit_image(
                image=image,
                mask=mask,
                prompt=req.prompt,
                steps=req.steps,
                seed=req.seed,
                reference_count=req.reference_count,
                settings=merged_settings,
            )
            status_state["state"] = "ready"
        finally:
            status_state["processing"] = False
            _inference_lock.release()

        buf = io.BytesIO()
        output_img.save(buf, format="PNG")
        res_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        push_log("[API /edit] Réponse renvoyée au navigateur.")
        return {
            "result_b64": res_b64,
            "telemetry": engine.get_vram_and_model_info(settings=merged_settings),
        }
    except ValueError as val_exc:
        err_text = str(val_exc)
        push_log(f"[API /edit AVERTISSEMENT] {err_text}")
        raise HTTPException(status_code=400, detail=err_text)
    except Exception as exc:
        err_text = f"{type(exc).__name__}: {exc}"
        push_log(f"[API /edit ERREUR] {err_text}")
        raise HTTPException(status_code=500, detail=err_text)


@app.post("/api/describe-region")
def describe_region_endpoint(req: DescribeRequest):
    if status_state["state"] != "ready" and engine.pipe is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore prêt.")

    try:
        push_log("[API /describe-region] Extraction du prompt de la zone sélectionnée...")
        img_part = req.image_b64.split(",", 1)[1] if "," in req.image_b64 else req.image_b64
        image = Image.open(io.BytesIO(base64.b64decode(img_part)))

        mask = None
        if req.mask_b64 and req.mask_b64.strip():
            mask_part = req.mask_b64.split(",", 1)[1] if "," in req.mask_b64 else req.mask_b64
            mask = Image.open(io.BytesIO(base64.b64decode(mask_part)))

        if not _inference_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Un traitement GPU est déjà en cours. Veuillez patienter.")

        status_state["processing"] = True
        try:
            cfg = load_config()
            info = engine.describe_masked_region(
                image=image,
                mask=mask,
                language=req.language,
                settings=cfg.get("settings"),
            )
            status_state["state"] = "ready"
        finally:
            status_state["processing"] = False
            _inference_lock.release()

        cfg = load_config()
        return {
            **info,
            "telemetry": engine.get_vram_and_model_info(settings=cfg.get("settings")),
        }
    except Exception as exc:
        err_text = f"{type(exc).__name__}: {exc}"
        push_log(f"[API /describe-region ERREUR] {err_text}")
        raise HTTPException(status_code=500, detail=err_text)

