@echo off
setlocal enabledelayedexpansion

:: Создаем папку для скриптов, если её нет
if not exist ".\py" mkdir ".\py"

for %%f in (".\notebooks\*.ipynb") do (
    set "filename=%%~nf"
    echo Конвертирую: !filename!
    ipynb-py-convert ".\notebooks\!filename!.ipynb" ".\py\!filename!.py"
)
pause