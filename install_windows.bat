@echo off

if exist "%~dp0.venv\" (
	echo Checking .venv...
	set PYTHON=".venv\Scripts\Python.exe"
	call ".venv\Scripts\activate"
	) else (
	echo Creating virtual environment...
	python -m venv .venv
	set PYTHON=".venv\Scripts\Python.exe"
	call ".venv\Scripts\activate"
	)

echo.
echo Upgrading pip...
%PYTHON% -m pip install --upgrade pip
echo.
echo Installing requirements from file...
pip install -r requirements.txt
echo.
echo Installing empty triton from url...
pip install https://github.com/woct0rdho/triton-windows/releases/download/empty/triton-3.2.0-py3-none-any.whl
call ".venv\Scripts\deactivate"

echo.
echo Requirements installed. Run using Start_SUPIR.bat
echo. 
pause