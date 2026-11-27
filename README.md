# Lampyre — Retouche Photo IA Locale

Application locale de retouche d'image par masque propulsée par **`briaai/Fibo-2.0-edit-turbo`** sur GPU NVIDIA, optimisée pour fonctionner dès **6 Go de VRAM**.

## Fonctionnalités
- **Installation 1-clic sous Windows** : environnement Python 3.11 portable, Redistributable Microsoft Visual C++ x64 et PyTorch CUDA installés automatiquement au premier lancement via `Lancer_Lampyre.bat`.
- **Optimisée 6 Go / 8 Go VRAM** : streaming asynchrone CUDA (`use_stream`), offload par couches sur Qwen3-VL, plafonnement VRAM anti-spillover WDDM et recadrage intelligent (*smart crop* mono et multi-zones).
- **Assistant Prompt intégré** : extraction automatique de la description de la zone masquée par vision IA (Qwen3-VL), modifiable et soumissible depuis une popup.
- **Interface complète** : gestion multi-versions (Originale, v1, v2…), comparateur Avant / Après entre versions, export **PNG / JPG / WEBP**, pinceau/gomme avec aperçu de taille, réglage de la teinte du masque, zoom avec déplacement au clic-molette, Annuler/Rétablir (`Ctrl+Z` / `Ctrl+Y`) et verrouillage de l'interface pendant les calculs GPU.

## Prérequis
- Windows 10 / 11 (64 bits)
- Carte graphique NVIDIA compatible CUDA (**6 Go de VRAM minimum**)
- 16 Go de RAM minimum (**32 Go recommandés**)
- ~30 Go d'espace disque libre pour le modèle local
- Compte [Hugging Face](https://huggingface.co/briaai/Fibo-2.0-edit-turbo) (conditions du modèle acceptées + jeton *Read*)

## Installation et lancement
1. Téléchargez ou clonez ce dépôt.
2. Double-cliquez sur **`Lancer_Lampyre.bat`**. Si Windows demande une autorisation pour installer Microsoft Visual C++ Redistributable, acceptez-la.
3. Au premier démarrage, l'environnement Python/CUDA s'installe automatiquement puis votre navigateur s'ouvre : collez votre jeton Hugging Face pour télécharger le modèle localement (une seule fois).

## Mise à jour automatique
- Lampyre vérifie les nouvelles GitHub Releases au démarrage.
- La mise à jour est disponible dans **Réglages & VRAM > Mise à jour Lampyre**.
- Le programme télécharge `lampyre.zip` avec Python et installe les nouveaux fichiers sans Git.
- Vos données locales (`data\`), votre environnement Python portable (`runtime\`) et votre jeton Hugging Face sont conservés.
- Après installation, utilisez **Redémarrer Lampyre** pour appliquer la mise à jour.

