@echo off

set PYTHON=".venv\Scripts\Python.exe"
call ".venv\Scripts\activate.bat"
%PYTHON% supir-server.py --use_tile_vae --no_llava --loading_half_params

echo.
echo Launch unsuccessful. Exiting.
pause