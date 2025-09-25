"""
Script de teste completo para verificar todas as funcionalidades
"""

import os
import sys
import subprocess
import time

def test_imports():
    """Testa se todas as dependências estão instaladas"""
    print("🔍 Testando imports...")
    
    try:
        import pandas as pd
        import numpy as np
        import plotly.express as px
        import streamlit as st
        import sklearn
        import requests
        print("✅ Todas as dependências importadas com sucesso!")
        return True
    except ImportError as e:
        print(f"❌ Erro ao importar dependência: {e}")
        return False

def test_data_generation():
    """Testa geração de dados sintéticos"""
    print("🔍 Testando geração de dados...")
    
    try:
        # Importar módulo
        sys.path.append('src')
        from data_acquisition import generate_synthetic_311_data, save_data
        
        # Gerar pequeno dataset de teste
        df = generate_synthetic_311_data(n_records=1000)
        save_data(df, 'test_data.csv')
        
        # Verificar se arquivo foi criado
        if os.path.exists('data/test_data.csv'):
            print("✅ Geração de dados funcionando!")
            # Limpar arquivo de teste
            os.remove('data/test_data.csv')
            return True
        else:
            print("❌ Arquivo de dados não foi criado")
            return False
            
    except Exception as e:
        print(f"❌ Erro na geração de dados: {e}")
        return False

def test_data_cleaning():
    """Testa limpeza de dados"""
    print("🔍 Testando limpeza de dados...")
    
    try:
        from data_cleaning import DataCleaner
        
        # Se não existir dados de teste, gerar primeiro
        if not os.path.exists('data/nyc_311_2023_sample.csv'):
            print("   Gerando dados de teste...")
            from data_acquisition import generate_synthetic_311_data, save_data
            df = generate_synthetic_311_data(n_records=1000)
            save_data(df)
        
        # Testar limpeza
        cleaner = DataCleaner()
        df = cleaner.load_raw_data()
        df = cleaner.normalize_dates(df)
        df = cleaner.calculate_response_time(df)
        
        print("✅ Limpeza de dados funcionando!")
        return True
        
    except Exception as e:
        print(f"❌ Erro na limpeza de dados: {e}")
        return False

def test_sql_analysis():
    """Testa análise SQL"""
    print("🔍 Testando análise SQL...")
    
    try:
        from sql_analysis import SQLAnalyzer
        
        analyzer = SQLAnalyzer()
        analyzer.connect_database('test_db.db')
        print("✅ Conexão SQL funcionando!")
        analyzer.close_connection()
        
        # Limpar arquivo de teste
        if os.path.exists('data/test_db.db'):
            os.remove('data/test_db.db')
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na análise SQL: {e}")
        return False

def test_model():
    """Testa modelo preditivo"""
    print("🔍 Testando modelo preditivo...")
    
    try:
        from predictive_model import PredictiveModel
        
        predictor = PredictiveModel()
        print("✅ Modelo preditivo carregado!")
        return True
        
    except Exception as e:
        print(f"❌ Erro no modelo preditivo: {e}")
        return False

def test_streamlit():
    """Testa se Streamlit pode ser importado"""
    print("🔍 Testando Streamlit...")
    
    try:
        import streamlit as st
        print("✅ Streamlit funcionando!")
        return True
        
    except Exception as e:
        print(f"❌ Erro no Streamlit: {e}")
        return False

def run_all_tests():
    """Executa todos os testes"""
    print("="*60)
    print("  TESTE COMPLETO DO PROJETO NYC 311 ANALYSIS")
    print("="*60)
    print()
    
    tests = [
        ("Imports das Dependências", test_imports),
        ("Geração de Dados", test_data_generation),
        ("Limpeza de Dados", test_data_cleaning),
        ("Análise SQL", test_sql_analysis),
        ("Modelo Preditivo", test_model),
        ("Streamlit", test_streamlit)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Teste: {test_name}")
        print('='*40)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erro inesperado: {e}")
            results.append((test_name, False))
        
        time.sleep(1)  # Pausa entre testes
    
    # Resumo dos resultados
    print("\n" + "="*60)
    print("  RESUMO DOS TESTES")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Resultado Final: {passed}/{total} testes passaram")
    
    if passed == total:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("✅ Projeto pronto para execução!")
        print("\n📋 Próximos passos:")
        print("   1. Execute: setup.bat (Windows) ou setup.sh (Linux/Mac)")
        print("   2. Acesse: http://localhost:8501")
    else:
        print("\n⚠️ Alguns testes falharam.")
        print("📋 Verifique as mensagens de erro acima e:")
        print("   1. Instale as dependências: pip install -r requirements.txt")
        print("   2. Verifique se Python 3.8+ está instalado")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)