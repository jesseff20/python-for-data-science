@echo off
echo ======================================
echo  NYC 311 Analysis - Setup Completo
echo ======================================
echo.

echo [1/4] Instalando dependencias...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERRO: Falha na instalacao das dependencias
    pause
    exit /b 1
)
echo ✓ Dependencias instaladas com sucesso!
echo.

echo [2/4] Criando dados de exemplo...
python src/data_acquisition.py
if %errorlevel% neq 0 (
    echo ERRO: Falha na geracao dos dados
    pause
    exit /b 1
)
echo ✓ Dados gerados com sucesso!
echo.

echo [3/4] Processando dados...
python src/data_cleaning.py
if %errorlevel% neq 0 (
    echo ERRO: Falha no processamento dos dados
    pause
    exit /b 1
)
echo ✓ Dados processados com sucesso!
echo.

echo [4/4] Iniciando dashboard Streamlit...
echo.
echo ==========================================
echo  Dashboard disponivel em: http://localhost:8501
echo  Pressione Ctrl+C para parar o servidor
echo ==========================================
echo.
streamlit run streamlit_app.py