"""
Módulo para análise exploratória de dados (EDA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class ExploratoryAnalysis:
    def __init__(self):
        self.data_dir = 'data'
        
        # Configurar estilo dos plots
        plt.style.use('default')
        sns.set_palette("husl")
    
    def load_clean_data(self, filename='nyc_311_2023_clean.csv'):
        """
        Carrega os dados limpos
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        
        print(f"Carregando dados limpos de: {filepath}")
        df = pd.read_csv(filepath)
        
        # Converter datas
        date_columns = ['created_date', 'closed_date']
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])
        
        print(f"✓ Dados carregados: {len(df):,} registros")
        return df
    
    def plot_complaints_by_type(self, df, top_n=15):
        """
        Gráfico de distribuição por tipo de reclamação
        """
        print("Criando gráfico de distribuição por tipo...")
        
        # Top tipos de reclamação
        complaint_counts = df['complaint_type'].value_counts().head(top_n)
        
        # Plotly
        fig = px.bar(
            x=complaint_counts.values,
            y=complaint_counts.index,
            orientation='h',
            title=f'Top {top_n} Tipos de Reclamações NYC 311 (2023)',
            labels={'x': 'Número de Chamados', 'y': 'Tipo de Reclamação'},
            color=complaint_counts.values,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def plot_complaints_by_borough(self, df):
        """
        Gráfico de distribuição por borough
        """
        print("Criando gráfico de distribuição por borough...")
        
        borough_counts = df['borough'].value_counts()
        
        # Gráfico de pizza
        fig = px.pie(
            values=borough_counts.values,
            names=borough_counts.index,
            title='Distribuição de Chamados por Borough',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        
        return fig
    
    def plot_response_time_by_category(self, df):
        """
        Box plot do tempo de resposta por categoria
        """
        print("Criando box plot de tempo de resposta...")
        
        # Filtrar apenas registros fechados
        closed_df = df[df['status'] == 'Closed'].copy()
        
        if len(closed_df) == 0:
            print("⚠️ Nenhum registro fechado encontrado")
            return None
        
        fig = px.box(
            closed_df,
            x='complaint_category',
            y='response_time_days',
            title='Tempo de Resposta por Categoria de Reclamação',
            labels={
                'response_time_days': 'Tempo de Resposta (dias)',
                'complaint_category': 'Categoria'
            }
        )
        
        fig.update_layout(
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def plot_temporal_patterns(self, df):
        """
        Análise de padrões temporais
        """
        print("Criando análise temporal...")
        
        # Criar subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Chamados por Dia da Semana',
                'Chamados por Hora do Dia',
                'Chamados por Mês',
                'Chamados por Período do Dia'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # 1. Por dia da semana
        weekday_counts = df['created_weekday'].value_counts()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_counts = weekday_counts.reindex(weekday_order)
        
        fig.add_trace(
            go.Bar(x=weekday_counts.index, y=weekday_counts.values, name="Dia da Semana"),
            row=1, col=1
        )
        
        # 2. Por hora do dia
        hour_counts = df['created_hour'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=hour_counts.index, y=hour_counts.values, name="Hora"),
            row=1, col=2
        )
        
        # 3. Por mês
        month_counts = df['created_month'].value_counts().sort_index()
        months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        fig.add_trace(
            go.Bar(x=[months[i-1] for i in month_counts.index], y=month_counts.values, name="Mês"),
            row=2, col=1
        )
        
        # 4. Por período do dia
        period_counts = df['created_period'].value_counts()
        fig.add_trace(
            go.Pie(labels=period_counts.index, values=period_counts.values, name="Período"),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Padrões Temporais dos Chamados 311",
            showlegend=False
        )
        
        return fig
    
    def plot_response_time_heatmap(self, df):
        """
        Heatmap do tempo médio de resposta por borough e categoria
        """
        print("Criando heatmap de tempo de resposta...")
        
        # Filtrar apenas registros fechados
        closed_df = df[df['status'] == 'Closed'].copy()
        
        if len(closed_df) == 0:
            print("⚠️ Nenhum registro fechado encontrado")
            return None
        
        # Calcular tempo médio por borough e categoria
        heatmap_data = closed_df.groupby(['borough', 'complaint_category'])['response_time_days'].mean().unstack()
        
        # Criar heatmap com plotly
        fig = px.imshow(
            heatmap_data,
            title='Tempo Médio de Resposta por Borough e Categoria (dias)',
            labels=dict(x="Categoria", y="Borough", color="Dias"),
            aspect="auto",
            color_continuous_scale='RdYlBu_r'
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def plot_geographic_distribution(self, df, sample_size=5000):
        """
        Mapa geográfico dos chamados
        """
        print(f"Criando mapa geográfico (amostra de {sample_size:,} registros)...")
        
        # Amostrar dados para melhor performance
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df.copy()
        
        # Remover coordenadas nulas
        df_sample = df_sample.dropna(subset=['latitude', 'longitude'])
        
        if len(df_sample) == 0:
            print("⚠️ Nenhum registro com coordenadas válidas")
            return None
        
        # Mapa de densidade
        fig = px.density_mapbox(
            df_sample,
            lat='latitude',
            lon='longitude',
            z='response_time_days' if 'response_time_days' in df_sample.columns else None,
            radius=10,
            center=dict(lat=40.7128, lon=-73.9354),  # NYC center
            zoom=10,
            mapbox_style="open-street-map",
            title='Distribuição Geográfica dos Chamados 311 em NYC',
            hover_data=['borough', 'complaint_type']
        )
        
        fig.update_layout(height=600)
        
        return fig
    
    def generate_summary_stats(self, df):
        """
        Gera estatísticas resumo
        """
        print("Gerando estatísticas resumo...")
        
        stats = {}
        
        # Estatísticas gerais
        stats['total_records'] = len(df)
        stats['closed_records'] = len(df[df['status'] == 'Closed'])
        stats['open_records'] = len(df[df['status'] != 'Closed'])
        
        # Período dos dados
        stats['date_range'] = {
            'start': df['created_date'].min(),
            'end': df['created_date'].max()
        }
        
        # Top 5 tipos de reclamação
        stats['top_complaint_types'] = df['complaint_type'].value_counts().head().to_dict()
        
        # Distribuição por borough
        stats['borough_distribution'] = df['borough'].value_counts().to_dict()
        
        # Tempo de resposta (apenas fechados)
        closed_df = df[df['status'] == 'Closed']
        if len(closed_df) > 0:
            stats['response_time'] = {
                'mean_days': closed_df['response_time_days'].mean(),
                'median_days': closed_df['response_time_days'].median(),
                'min_days': closed_df['response_time_days'].min(),
                'max_days': closed_df['response_time_days'].max(),
                'std_days': closed_df['response_time_days'].std()
            }
        
        # Padrões temporais
        stats['temporal_patterns'] = {
            'busiest_weekday': df['created_weekday'].value_counts().index[0],
            'busiest_hour': df['created_hour'].value_counts().index[0],
            'busiest_month': df['created_month'].value_counts().index[0]
        }
        
        return stats
    
    def create_all_visualizations(self, df):
        """
        Cria todas as visualizações e retorna lista de figuras
        """
        print("\n" + "="*50)
        print("  CRIANDO VISUALIZAÇÕES")
        print("="*50)
        
        visualizations = {}
        
        try:
            # 1. Distribuição por tipo
            visualizations['complaints_by_type'] = self.plot_complaints_by_type(df)
            
            # 2. Distribuição por borough
            visualizations['complaints_by_borough'] = self.plot_complaints_by_borough(df)
            
            # 3. Tempo de resposta por categoria
            visualizations['response_time_by_category'] = self.plot_response_time_by_category(df)
            
            # 4. Padrões temporais
            visualizations['temporal_patterns'] = self.plot_temporal_patterns(df)
            
            # 5. Heatmap
            visualizations['response_time_heatmap'] = self.plot_response_time_heatmap(df)
            
            # 6. Mapa geográfico
            visualizations['geographic_distribution'] = self.plot_geographic_distribution(df)
            
            # 7. Estatísticas resumo
            visualizations['summary_stats'] = self.generate_summary_stats(df)
            
            print("✅ Todas as visualizações criadas com sucesso!")
            
        except Exception as e:
            print(f"❌ Erro ao criar visualizações: {e}")
            return None
        
        return visualizations

def main():
    """
    Função principal para executar a análise exploratória
    """
    print("=" * 50)
    print("  NYC 311 Exploratory Data Analysis")
    print("=" * 50)
    
    # Inicializar analisador
    analyzer = ExploratoryAnalysis()
    
    try:
        # Carregar dados limpos
        df = analyzer.load_clean_data()
        
        # Criar visualizações
        visualizations = analyzer.create_all_visualizations(df)
        
        if visualizations:
            print(f"\n📊 {len(visualizations)} visualizações criadas:")
            for name in visualizations.keys():
                print(f"   ✓ {name}")
        
    except Exception as e:
        print(f"❌ Erro durante análise: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()