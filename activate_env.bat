@echo off
REM Facial_Details_Synthesis ortam aktivasyonu
REM Bu bat dosyasını çalıştırarak doğru conda environment'ı aktive edebilirsiniz.

call C:\Users\HP\miniconda3\Scripts\activate.bat facial_details
cd /d "%~dp0released"
echo.
echo === Facial Details Synthesis Environment ===
echo Python: 3.7.16
echo PyTorch: 1.13.1 (CUDA 11.7)
echo Working Dir: %CD%
echo.
echo Kullanim:
echo   python proxyPredictor.py -i ./samples/proxy -o ./results
echo   python facialDetails.py -i ./samples/details/019615.jpg -o ./results
echo.
