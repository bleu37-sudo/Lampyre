@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
title Lampyre - Retouche Photo IA Locale

cd /d "%~dp0"
set "APP_DIR=%~dp0"
set "RUNTIME_DIR=%~dp0runtime"
set "PY_EXE=%RUNTIME_DIR%\python.exe"
set "READY_MARKER=%RUNTIME_DIR%\.lampyre_ready"
set "SETUP_ONLY=0"
if /i "%~1"=="--setup-only" set "SETUP_ONLY=1"
if "%LAMPYRE_SETUP_ONLY%"=="1" set "SETUP_ONLY=1"

if not exist "%PY_EXE%" goto :install_runtime
if not exist "%READY_MARKER%" goto :install_runtime
"%PY_EXE%" -c "import torch" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [REPARATION] Le runtime Python existe, mais PyTorch ne se charge pas.
    echo [REPARATION] Reinstallation des composants avec verification du Redistributable VC++.
    goto :install_runtime
)
goto :start_app

:install_runtime
echo ============================================================================
echo   [SETUP] Environnement local absent : preparation unique en cours
echo ============================================================================
if not exist "%RUNTIME_DIR%" mkdir "%RUNTIME_DIR%"

echo [SETUP 1/4] Telechargement de Python 3.11 portable...
set "PY_ZIP=%RUNTIME_DIR%\py311.zip"
if exist "%PY_ZIP%" del /f /q "%PY_ZIP%" 2>nul

where curl.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    curl.exe -fL --retry 3 --retry-delay 2 -o "%PY_ZIP%" "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
)
if not exist "%PY_ZIP%" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip' -OutFile $env:PY_ZIP"
)
if not exist "%PY_ZIP%" (
    echo [ERREUR] Echec du telechargement de Python 3.11 portable.
    goto :fail
)

echo [SETUP 1/4] Extraction de Python 3.11 portable...
where tar.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    tar.exe -xf "%PY_ZIP%" -C "%RUNTIME_DIR%"
)
if not exist "%PY_EXE%" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path $env:PY_ZIP -DestinationPath $env:RUNTIME_DIR -Force"
)
del /f /q "%PY_ZIP%" 2>nul

if not exist "%PY_EXE%" (
    echo [ERREUR] python.exe introuvable apres extraction dans "%RUNTIME_DIR%".
    goto :fail
)

echo [SETUP 2/4] Configuration de python311._pth et installation de pip...
set "PTH_FILE=%RUNTIME_DIR%\python311._pth"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = $env:PTH_FILE; $nl = [Environment]::NewLine; $txt = [System.IO.File]::ReadAllText($p).Replace('#import site', 'import site'); if (-not $txt.EndsWith($nl)) { $txt += $nl }; if ($txt -notmatch '(?m)^\.\.\r?$') { $txt += '..' + $nl }; [System.IO.File]::WriteAllText($p, $txt, (New-Object System.Text.UTF8Encoding($false)))"
if %ERRORLEVEL% NEQ 0 (
    echo [ERREUR] Echec de la configuration de python311._pth.
    goto :fail
)

set "GET_PIP=%RUNTIME_DIR%\get-pip.py"
if exist "%GET_PIP%" del /f /q "%GET_PIP%" 2>nul
where curl.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    curl.exe -fL --retry 3 --retry-delay 2 -o "%GET_PIP%" "https://bootstrap.pypa.io/get-pip.py"
)
if not exist "%GET_PIP%" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile $env:GET_PIP"
)
if not exist "%GET_PIP%" (
    echo [ERREUR] Echec du telechargement de get-pip.py.
    goto :fail
)

"%PY_EXE%" "%GET_PIP%" --no-warn-script-location
if %ERRORLEVEL% NEQ 0 (
    echo [ERREUR] Echec de l'installation de pip.
    goto :fail
)
del /f /q "%GET_PIP%" 2>nul

echo [SETUP 3/5] Installation du Redistributable Microsoft Visual C++ x64...
set "VC_REDIST=%RUNTIME_DIR%\vc_redist.x64.exe"
call :install_vc_redist
if %ERRORLEVEL% NEQ 0 goto :fail

echo [SETUP 4/5] Installation de PyTorch CUDA 12.4 pour GPU NVIDIA...
"%PY_EXE%" -m pip install --no-warn-script-location --index-url https://download.pytorch.org/whl/cu124 torch torchvision
if %ERRORLEVEL% NEQ 0 (
    echo [ERREUR] Echec de l'installation de PyTorch CUDA.
    goto :fail
)

echo [SETUP 5/5] Installation des bibliotheques diffusers, transformers, fastapi et uvicorn...
"%PY_EXE%" -m pip install --no-warn-script-location -r "%~dp0requirements.txt"
if %ERRORLEVEL% NEQ 0 (
    echo [ERREUR] Echec de l'installation des dependances Python.
    goto :fail
)

echo ready > "%READY_MARKER%"
echo [SETUP OK] Environnement pret.

:start_app
if "%SETUP_ONLY%"=="1" (
    echo [SETUP ONLY] Installation terminee avec succes sans lancer le serveur.
    set "EXIT_CODE=0"
    goto :finish
)

echo.
echo [DEMARRAGE] Lancement de launcher.py...
"%PY_EXE%" "%~dp0launcher.py"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERREUR] Le programme s'est arrete avec le code %ERRORLEVEL%. Consultez data\lampyre.log.
    goto :fail
)
set "EXIT_CODE=0"
goto :finish

:install_vc_redist
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -Command "$r=Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64' -ErrorAction SilentlyContinue; if ($r -and $r.Installed -eq 1) { exit 0 }; exit 1"
if %ERRORLEVEL% EQU 0 (
    echo [SETUP 3/5] Redistributable Microsoft Visual C++ x64 deja installe.
    endlocal & exit /b 0
)

if exist "%VC_REDIST%" del /f /q "%VC_REDIST%" 2>nul
echo [SETUP 3/5] Telechargement du Redistributable Microsoft Visual C++ x64...
where curl.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    curl.exe -sSfL --connect-timeout 15 --max-time 300 --retry 3 --retry-delay 2 -o "%VC_REDIST%" "https://aka.ms/vs/17/release/vc_redist.x64.exe"
)
if not exist "%VC_REDIST%" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile $env:VC_REDIST"
)
if not exist "%VC_REDIST%" (
    echo [ERREUR] Echec du telechargement du Redistributable Microsoft Visual C++.
    endlocal & exit /b 1
)

echo [SETUP 3/5] Execution du programme d'installation VC++...
"%VC_REDIST%" /install /passive /norestart
set "VC_EXIT=%ERRORLEVEL%"
if not "%VC_EXIT%"=="0" if not "%VC_EXIT%"=="3010" (
    echo [ERREUR] Echec de l'installation du Redistributable Microsoft Visual C++ ^(code %VC_EXIT%^).
    echo [ERREUR] Lancez Lancer_Lampyre.bat depuis une session Windows avec les droits administrateur.
    del /f /q "%VC_REDIST%" 2>nul
    endlocal & exit /b 1
)
del /f /q "%VC_REDIST%" 2>nul
endlocal & exit /b 0

:fail
set "EXIT_CODE=1"

:finish
if not "%LAMPYRE_NO_PAUSE%"=="1" pause
exit /b %EXIT_CODE%

