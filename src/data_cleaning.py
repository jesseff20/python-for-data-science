"""
Módulo para limpeza e transformação dos dados NYC 311
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

class DataCleaner:
    def __init__(self):
        self.data_dir = 'data'
        
        # Mapeamentos de tradução para português brasileiro
        self.borough_translation = {
            'MANHATTAN': 'Manhattan',
            'BROOKLYN': 'Brooklyn',
            'QUEENS': 'Queens',
            'BRONX': 'Bronx',
            'STATEN ISLAND': 'Staten Island'
        }
        
        self.complaint_type_translation = {
            'Noise - Residential': 'Ruído - Residencial',
            'Noise - Street/Sidewalk': 'Ruído - Rua/Calçada',
            'Noise - Commercial': 'Ruído - Comercial',
            'Heat/Hot Water': 'Aquecimento/Água Quente',
            'PAINT/PLASTER': 'Pintura/Reboco',
            'DOOR/WINDOW': 'Porta/Janela',
            'APPLIANCE': 'Eletrodomésticos',
            'PLUMBING': 'Encanamento',
            'Street Condition': 'Condição da Rua',
            'Street Light Condition': 'Iluminação Pública',
            'Traffic Signal Condition': 'Semáforo',
            'UNSANITARY CONDITION': 'Condições Insalubres',
            'Dirty Conditions': 'Condições de Sujeira',
            'Rodent': 'Roedores',
            'Graffiti': 'Pichação',
            'Blocked Driveway': 'Garagem Bloqueada',
            'Illegal Parking': 'Estacionamento Irregular',
            'Air Quality': 'Qualidade do Ar',
            'Tree Care': 'Cuidado com Árvores',
            'Dead/Dying Tree': 'Árvore Morta/Morrendo',
            'Water System': 'Sistema de Água',
            'Animal Abuse': 'Maus-Tratos Animais'
        }
        
        self.status_translation = {
            'Open': 'Aberto',
            'Closed': 'Fechado',
            'In Progress': 'Em Andamento',
            'Pending': 'Pendente'
        }
        
        self.weekday_translation = {
            'Monday': 'Segunda-feira',
            'Tuesday': 'Terça-feira',
            'Wednesday': 'Quarta-feira',
            'Thursday': 'Quinta-feira',
            'Friday': 'Sexta-feira',
            'Saturday': 'Sábado',
            'Sunday': 'Domingo'
        }
        
        self.period_translation = {
            'Madrugada': 'Madrugada',
            'Manhã': 'Manhã',
            'Tarde': 'Tarde',
            'Noite': 'Noite'
        }
        
        self.category_translation = {
            'Noise': 'Ruído',
            'Housing': 'Habitação',
            'Streets': 'Vias Públicas',
            'Sanitation': 'Saneamento',
            'Parking': 'Estacionamento',
            'Environment': 'Meio Ambiente',
            'Water': 'Água',
            'Animal': 'Animais',
            'Other': 'Outros'
        }
    
    def load_raw_data(self, filename='nyc_311_2023_sample.csv'):
        """
        Carrega os dados brutos
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        
        print(f"Carregando dados de: {filepath}")
        df = pd.read_csv(filepath)
        
        print(f"✓ Dados carregados: {len(df):,} registros")
        return df
    
    def normalize_dates(self, df):
        """
        Normaliza as colunas de data
        """
        print("Normalizando datas...")
        
        # Converter para datetime
        df['created_date'] = pd.to_datetime(df['created_date'])
        df['closed_date'] = pd.to_datetime(df['closed_date'])
        
        # Extrair componentes úteis
        df['created_year'] = df['created_date'].dt.year
        df['created_month'] = df['created_date'].dt.month
        df['created_day'] = df['created_date'].dt.day
        df['created_weekday'] = df['created_date'].dt.day_name()
        df['created_hour'] = df['created_date'].dt.hour
        
        # Criar período do dia
        df['created_period'] = df['created_hour'].apply(self._get_period)
        
        print("✓ Datas normalizadas")
        return df
    
    def translate_data(self, df):
        """
        Traduz dados categóricos para português brasileiro
        """
        print("Traduzindo dados para português...")
        
        # Traduzir borough
        df['borough'] = df['borough'].map(self.borough_translation).fillna(df['borough'])
        
        # Traduzir complaint_type
        df['complaint_type'] = df['complaint_type'].map(self.complaint_type_translation).fillna(df['complaint_type'])
        
        # Traduzir status
        df['status'] = df['status'].map(self.status_translation).fillna(df['status'])
        
        # Traduzir dia da semana
        df['created_weekday'] = df['created_weekday'].map(self.weekday_translation).fillna(df['created_weekday'])
        
        print("✓ Dados traduzidos para português")
        return df
    
    def _get_period(self, hour):
        """
        Converte hora em período do dia
        """
        if 6 <= hour < 12:
            return 'Manhã'
        elif 12 <= hour < 18:
            return 'Tarde'
        elif 18 <= hour < 24:
            return 'Noite'
        else:
            return 'Madrugada'
    
    def calculate_response_time(self, df):
        """
        Calcula o tempo de resposta em horas e dias
        """
        print("Calculando tempo de resposta...")
        
        # Filtrar apenas registros fechados (antes da tradução, ainda é 'Closed')
        closed_mask = df['status'] == 'Closed'
        
        # Calcular tempo de resposta em horas
        df.loc[closed_mask, 'response_time_hours'] = (
            df.loc[closed_mask, 'closed_date'] - df.loc[closed_mask, 'created_date']
        ).dt.total_seconds() / 3600
        
        # Calcular tempo de resposta em dias
        df.loc[closed_mask, 'response_time_days'] = df.loc[closed_mask, 'response_time_hours'] / 24
        
        # Para registros não fechados, tempo é NaN
        df.loc[~closed_mask, 'response_time_hours'] = np.nan
        df.loc[~closed_mask, 'response_time_days'] = np.nan
        
        print(f"✓ Tempo de resposta calculado para {closed_mask.sum():,} registros fechados")
        return df
    
    def clean_outliers(self, df, max_days=90):
        """
        Remove outliers de tempo de resposta
        """
        print(f"Removendo outliers (tempo > {max_days} dias)...")
        
        # Contar outliers
        outliers_mask = df['response_time_days'] > max_days
        n_outliers = outliers_mask.sum()
        
        if n_outliers > 0:
            # Remover outliers
            df = df[~outliers_mask]
            print(f"✓ Removidos {n_outliers:,} outliers")
        else:
            print("✓ Nenhum outlier encontrado")
        
        return df
    
    def standardize_categories(self, df):
        """
        Padroniza categorias de reclamações com traduções em português
        """
        print("Padronizando categorias...")
        
        # Categorias principais usando nomes traduzidos
        category_mapping = {
            'Ruído': ['Ruído - Residencial', 'Ruído - Rua/Calçada', 'Ruído - Comercial'],
            'Habitação': ['Aquecimento/Água Quente', 'Pintura/Reboco', 'Porta/Janela', 'Eletrodomésticos', 'Encanamento'],
            'Vias Públicas': ['Condição da Rua', 'Iluminação Pública', 'Semáforo'],
            'Saneamento': ['Condições Insalubres', 'Condições de Sujeira', 'Roedores', 'Pichação'],
            'Estacionamento': ['Garagem Bloqueada', 'Estacionamento Irregular'],
            'Meio Ambiente': ['Qualidade do Ar', 'Cuidado com Árvores', 'Árvore Morta/Morrendo'],
            'Água': ['Sistema de Água'],
            'Animais': ['Maus-Tratos Animais'],
            'Outros': []
        }
        
        # Criar coluna de categoria principal
        df['complaint_category'] = 'Outros'
        
        for category, complaint_types in category_mapping.items():
            if complaint_types:  # Se não for vazia
                mask = df['complaint_type'].isin(complaint_types)
                df.loc[mask, 'complaint_category'] = category
            else:  # Para 'Outros'
                continue
        
        # Para tipos não mapeados, manter como 'Outros'
        mapped_types = []
        for types in category_mapping.values():
            mapped_types.extend(types)
        
        unmapped_mask = ~df['complaint_type'].isin(mapped_types)
        df.loc[unmapped_mask, 'complaint_category'] = 'Outros'
        
        print("✓ Categorias padronizadas")
        return df
    
    def handle_missing_values(self, df):
        """
        Trata valores nulos
        """
        print("Tratando valores nulos...")
        
        initial_nulls = df.isnull().sum().sum()
        
        # Zip codes nulos: preencher com valor padrão baseado no borough
        borough_zip_defaults = {
            'MANHATTAN': 10001,
            'BROOKLYN': 11201,
            'QUEENS': 11101,
            'BRONX': 10451,
            'STATEN ISLAND': 10301
        }
        
        for borough, default_zip in borough_zip_defaults.items():
            mask = (df['borough'] == borough) & (df['zip_code'].isnull())
            df.loc[mask, 'zip_code'] = default_zip
        
        # Coordenadas nulas: remover registros (são poucos)
        coords_null_mask = df[['latitude', 'longitude']].isnull().any(axis=1)
        if coords_null_mask.sum() > 0:
            df = df[~coords_null_mask]
            print(f"  Removidos {coords_null_mask.sum():,} registros com coordenadas nulas")
        
        final_nulls = df.isnull().sum().sum()
        print(f"✓ Valores nulos tratados: {initial_nulls} → {final_nulls}")
        
        return df
    
    def add_derived_features(self, df):
        """
        Adiciona features derivadas úteis para análise
        """
        print("Adicionando features derivadas...")
        
        # Indicador de weekend
        df['is_weekend'] = df['created_date'].dt.weekday >= 5
        
        # Quartil do ano
        df['quarter'] = df['created_date'].dt.quarter
        
        # Indicador de horário comercial (9h-17h, seg-sex)
        business_hours = (df['created_hour'].between(9, 17)) & (~df['is_weekend'])
        df['is_business_hours'] = business_hours
        
        # Prioridade baseada no tipo (usando nomes traduzidos)
        priority_types = ['Aquecimento/Água Quente', 'Sistema de Água', 'Condições Insalubres']
        df['is_priority'] = df['complaint_type'].isin(priority_types)
        
        print("✓ Features derivadas adicionadas")
        return df
    
    def save_cleaned_data(self, df, filename='nyc_311_2023_clean.csv'):
        """
        Salva os dados limpos
        """
        filepath = os.path.join(self.data_dir, filename)
        
        print(f"Salvando dados limpos em: {filepath}")
        df.to_csv(filepath, index=False)
        
        print(f"✓ Dados limpos salvos: {len(df):,} registros")
        return filepath
    
    def generate_summary_report(self, df):
        """
        Gera relatório resumo da limpeza
        """
        print("\n" + "="*50)
        print("  RELATÓRIO DE LIMPEZA DOS DADOS")
        print("="*50)
        
        print(f"📊 Total de registros: {len(df):,}")
        print(f"📅 Período: {df['created_date'].min()} a {df['created_date'].max()}")
        
        print(f"\n🔢 Status dos chamados:")
        status_counts = df['status'].value_counts()
        for status, count in status_counts.items():
            pct = (count / len(df)) * 100
            print(f"   {status}: {count:,} ({pct:.1f}%)")
        
        print(f"\n🏙️ Distribuição por Borough:")
        borough_counts = df['borough'].value_counts()
        for borough, count in borough_counts.items():
            pct = (count / len(df)) * 100
            print(f"   {borough}: {count:,} ({pct:.1f}%)")
        
        print(f"\n📋 Top 5 tipos de reclamação:")
        complaint_counts = df['complaint_type'].value_counts().head()
        for complaint, count in complaint_counts.items():
            pct = (count / len(df)) * 100
            print(f"   {complaint}: {count:,} ({pct:.1f}%)")
        
        # Estatísticas de tempo de resposta (apenas fechados)
        closed_df = df[df['status'] == 'Closed']
        if len(closed_df) > 0:
            print(f"\n⏱️ Tempo de resposta (apenas fechados - {len(closed_df):,} registros):")
            print(f"   Médio: {closed_df['response_time_days'].mean():.1f} dias")
            print(f"   Mediano: {closed_df['response_time_days'].median():.1f} dias")
            print(f"   Mínimo: {closed_df['response_time_days'].min():.1f} dias")
            print(f"   Máximo: {closed_df['response_time_days'].max():.1f} dias")
        
        print("\n✅ Limpeza concluída com sucesso!")

def main():
    """
    Função principal para executar a limpeza completa
    """
    print("=" * 50)
    print("  NYC 311 Data Cleaning")
    print("=" * 50)
    
    # Inicializar cleaner
    cleaner = DataCleaner()
    
    try:
        # 1. Carregar dados brutos
        df = cleaner.load_raw_data()
        
        # 2. Normalizar datas
        df = cleaner.normalize_dates(df)
        
        # 3. Calcular tempo de resposta (antes da tradução)
        df = cleaner.calculate_response_time(df)
        
        # 4. Traduzir dados para português
        df = cleaner.translate_data(df)
        
        # 5. Remover outliers
        df = cleaner.clean_outliers(df)
        
        # 6. Padronizar categorias
        df = cleaner.standardize_categories(df)
        
        # 7. Tratar valores nulos
        df = cleaner.handle_missing_values(df)
        
        # 8. Adicionar features derivadas
        df = cleaner.add_derived_features(df)
        
        # 9. Salvar dados limpos
        cleaner.save_cleaned_data(df)
        
        # 10. Gerar relatório
        cleaner.generate_summary_report(df)
        
    except Exception as e:
        print(f"❌ Erro durante a limpeza: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()