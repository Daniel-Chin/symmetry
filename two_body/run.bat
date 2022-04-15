@echo off
call %CONDAPATH%\Scripts\activate.bat %CONDAPATH%
SET PYTHONPATH=C:\Users\iGlop\d\symmetry\Self-supervised learning via symmetry
call python %*
pause
