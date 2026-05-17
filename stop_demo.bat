@echo off
echo Stopping MAPE-K Demo services...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq MAPE-K API*" 2>nul
taskkill /F /FI "IMAGENAME eq streamlit.exe" 2>nul
echo Done.
