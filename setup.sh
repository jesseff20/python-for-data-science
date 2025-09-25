#!/bin/bash
# Setup script for Unix systems (Linux/MacOS)

echo "========================================"
echo "  NYC 311 Analysis - Setup Completo"
echo "========================================"
echo

echo "[1/4] Instalando dependencias..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERRO: Falha na instalacao das dependencias"
    exit 1
fi
echo "✓ Dependencias instaladas com sucesso!"
echo

echo "[2/4] Criando dados de exemplo..."
python src/data_acquisition.py
if [ $? -ne 0 ]; then
    echo "ERRO: Falha na geracao dos dados"
    exit 1
fi
echo "✓ Dados gerados com sucesso!"
echo

echo "[3/4] Processando dados..."
python src/data_cleaning.py
if [ $? -ne 0 ]; then
    echo "ERRO: Falha no processamento dos dados"
    exit 1
fi
echo "✓ Dados processados com sucesso!"
echo

echo "[4/4] Iniciando dashboard Streamlit..."
echo
echo "=========================================="
echo "  Dashboard disponivel em: http://localhost:8501"
echo "  Pressione Ctrl+C para parar o servidor"
echo "=========================================="
echo
streamlit run streamlit_app.py