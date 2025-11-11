@echo off
echo Compilando Java a partir da raiz do projeto...
echo.

REM Ir para o diretório raiz do projeto
cd /d "%~dp0\.."

REM Compilar todas as classes dentro de src\java
javac src\java\*.java

if %errorlevel% == 0 (
    echo.
    echo Compilacao concluida com sucesso!
    echo.
    echo Para executar o treino:
    echo   java -cp src\java Main
    echo.
) else (
    echo.
    echo ERRO na compilacao!
    echo.
)

pause

