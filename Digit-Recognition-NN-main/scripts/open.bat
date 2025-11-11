@echo off
echo Tentando iniciar servidor Python...
echo.

REM Mudar para o diretório raiz do projeto (um nível acima do script)
cd /d "%~dp0\.."

REM Tentar diferentes comandos Python
python scripts\server.py 2>nul
if %errorlevel% == 0 goto :end

py scripts\server.py 2>nul
if %errorlevel% == 0 goto :end

python3 scripts\server.py 2>nul
if %errorlevel% == 0 goto :end

REM Se nenhum funcionar, tentar encontrar Python
echo Python nao encontrado. Tentando localizar...
for /f "tokens=*" %%i in ('where /r "%LOCALAPPDATA%\Microsoft\WindowsApps" python.exe 2^>nul') do (
    echo Encontrado: %%i
    cd /d "%~dp0\.."
    "%%i" scripts\server.py
    goto :end
)

echo.
echo ERRO: Python nao encontrado!
echo.
echo Por favor:
echo 1. Reinicia o terminal/PowerShell
echo 2. Ou executa manualmente: python scripts\server.py
echo 3. Ou usa: python -m http.server 8000
echo.
pause

:end

