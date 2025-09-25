"""
Script para aquisição de dados NYC 311 Service Requests
Gera dados sintéticos realistas baseados no padrão dos dados reais
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import os

def generate_synthetic_311_data(n_records=50000):
    """
    Gera dados sintéticos realistas para simular o dataset NYC 311
    """
    print(f"Gerando {n_records:,} registros sintéticos...")
    
    # Seed para reprodutibilidade
    np.random.seed(42)
    
    # Tipos de reclamações mais comuns em NYC (já em português)
    complaint_types = [
        'Ruído - Residencial', 'Aquecimento/Água Quente', 'Condições Insalubres',
        'Garagem Bloqueada', 'Sistema de Água', 'Condição da Rua',
        'Estacionamento Irregular', 'Semáforo', 'Pintura/Reboco',
        'Porta/Janela', 'Encanamento', 'Roedores', 'Iluminação Pública',
        'Eletrodomésticos', 'Pichação', 'Cuidado com Árvores', 'Qualidade do Ar',
        'Maus-Tratos Animais', 'Árvore Morta/Morrendo', 'Condições de Sujeira'
    ]
    
    # Boroughs de NYC (mantendo nomes originais conhecidos)
    boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
    borough_weights = [0.25, 0.30, 0.28, 0.12, 0.05]
    
    # Agencies responsáveis
    agencies = ['NYPD', 'HPD', 'DOT', 'DSNY', 'DEP', 'DPR', 'DOHMH', 'DOB']
    
    # Status dos chamados (já em português)
    status_options = ['Fechado', 'Aberto', 'Pendente', 'Em Andamento']
    status_weights = [0.75, 0.10, 0.10, 0.05]
    
    # Gerar dados
    data = []
    
    # Data base: 2023
    base_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    for i in range(n_records):
        if i % 10000 == 0:
            print(f"  Progresso: {i:,}/{n_records:,}")
            
        # Data de criação aleatória em 2023
        created_date = base_date + timedelta(
            days=np.random.randint(0, 365),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )
        
        # Selecionar tipo de reclamação
        complaint_type = np.random.choice(complaint_types)
        
        # Borough
        borough = np.random.choice(boroughs, p=borough_weights)
        
        # Status
        status = np.random.choice(status_options, p=status_weights)
        
        # Data de fechamento (apenas se status for 'Closed')
        if status == 'Closed':
            # Tempo de resposta varia por tipo de reclamação
            if 'Noise' in complaint_type:
                # Ruído: geralmente mais rápido (0-7 dias)
                response_hours = np.random.exponential(24)
            elif 'Heat' in complaint_type:
                # Aquecimento: prioritário (0-3 dias)
                response_hours = np.random.exponential(12)
            elif 'Street' in complaint_type:
                # Rua: pode ser mais demorado (0-30 dias)
                response_hours = np.random.exponential(72)
            else:
                # Outros: tempo médio (0-14 dias)
                response_hours = np.random.exponential(48)
            
            # Limitar tempo máximo a 90 dias
            response_hours = min(response_hours, 90 * 24)
            
            closed_date = created_date + timedelta(hours=response_hours)
        else:
            closed_date = None
        
        # Agency
        agency = np.random.choice(agencies)
        
        # Zip code (simulado)
        if borough == 'MANHATTAN':
            zip_code = np.random.choice(range(10001, 10282))
        elif borough == 'BROOKLYN':
            zip_code = np.random.choice(range(11201, 11256))
        elif borough == 'QUEENS':
            zip_code = np.random.choice(range(11101, 11697))
        elif borough == 'BRONX':
            zip_code = np.random.choice(range(10451, 10475))
        else:  # Staten Island
            zip_code = np.random.choice(range(10301, 10314))
        
        # Coordenadas simuladas por borough
        if borough == 'MANHATTAN':
            latitude = np.random.uniform(40.7009, 40.8820)
            longitude = np.random.uniform(-74.0479, -73.9067)
        elif borough == 'BROOKLYN':
            latitude = np.random.uniform(40.5707, 40.7395)
            longitude = np.random.uniform(-74.0446, -73.8333)
        elif borough == 'QUEENS':
            latitude = np.random.uniform(40.5456, 40.8048)
            longitude = np.random.uniform(-73.9630, -73.7004)
        elif borough == 'BRONX':
            latitude = np.random.uniform(40.7856, 40.9176)
            longitude = np.random.uniform(-73.9339, -73.7654)
        else:  # Staten Island
            latitude = np.random.uniform(40.4774, 40.6514)
            longitude = np.random.uniform(-74.2591, -74.0522)
        
        record = {
            'unique_key': f"2023{i:06d}",
            'created_date': created_date.strftime('%Y-%m-%d %H:%M:%S'),
            'closed_date': closed_date.strftime('%Y-%m-%d %H:%M:%S') if closed_date else None,
            'complaint_type': complaint_type,
            'descriptor': f"{complaint_type} - Details",
            'borough': borough,
            'zip_code': zip_code,
            'latitude': round(latitude, 6),
            'longitude': round(longitude, 6),
            'agency': agency,
            'status': status,
            'resolution_description': 'The Department investigated the complaint and took appropriate action.' if status == 'Closed' else None
        }
        
        data.append(record)
    
    print("Convertendo para DataFrame...")
    df = pd.DataFrame(data)
    
    return df

def save_data(df, filename='nyc_311_2023_sample.csv'):
    """
    Salva os dados no diretório data/
    """
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    filepath = os.path.join(data_dir, filename)
    
    print(f"Salvando dados em: {filepath}")
    df.to_csv(filepath, index=False)
    
    print(f"✓ Dados salvos com sucesso! ({len(df):,} registros)")
    print(f"✓ Arquivo: {filepath}")
    
    # Estatísticas básicas
    print(f"\n📊 Estatísticas dos dados gerados:")
    print(f"   Período: {df['created_date'].min()} a {df['created_date'].max()}")
    print(f"   Registros fechados: {df['status'].value_counts().get('Closed', 0):,}")
    print(f"   Boroughs: {df['borough'].nunique()}")
    print(f"   Tipos de reclamação: {df['complaint_type'].nunique()}")

def main():
    """
    Função principal para gerar e salvar os dados
    """
    print("=" * 50)
    print("  NYC 311 Data Generator")
    print("=" * 50)
    
    # Gerar dados sintéticos
    df = generate_synthetic_311_data(n_records=50000)
    
    # Salvar dados
    save_data(df)
    
    print("\n✅ Geração de dados concluída!")

if __name__ == "__main__":
    main()