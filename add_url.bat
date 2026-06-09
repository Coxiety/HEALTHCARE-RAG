@echo off
setlocal

if "%~1"=="" (
    echo Vui long cung cap mot duong link bai bao hoac tu khoi tim kiem!
    echo Cach su dung: add_url.bat "red meat diabetes"
    goto end
)

echo Dang cai dat thu vien (neu chua co)...
python -m pip install beautifulsoup4 requests biopython -q

echo.
echo Dang nap du lieu vao RAG...
python scripts\add_url_to_rag.py %*

:end
echo.
pause
