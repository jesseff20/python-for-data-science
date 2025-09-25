"""
Dashboard Principal Streamlit - NYC 311 Service Requests Analysis
Análise moderna de dados com Python, SQL e Machine Learning
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import sys
import sqlite3
from datetime import datetime

# Adicionar pasta src ao path
sys.path.append('src')

# Imports dos módulos do projeto
from data_acquisition import generate_synthetic_311_data, save_data
from data_cleaning import DataCleaner
from exploratory_analysis import ExploratoryAnalysis
from sql_analysis import SQLAnalyzer
from predictive_model import PredictiveModel

# Configuração da página
st.set_page_config(
    page_title="NYC 311 Analysis",
    page_icon="🏙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache para dados
@st.cache_data
def load_data():
    """Carrega dados limpos com cache"""
    try:
        df = pd.read_csv('data/nyc_311_2023_clean.csv')
        df['created_date'] = pd.to_datetime(df['created_date'])
        df['closed_date'] = pd.to_datetime(df['closed_date'])
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def run_sql_analysis(df):
    """Executa análises SQL básicas usando pandas"""
    results = {}
    
    try:
        # 1. Tipos mais demorados
        closed_df = df[df['status'] == 'Closed'].copy()
        if len(closed_df) > 0:
            slowest = (closed_df.groupby('complaint_type')
                      .agg({'response_time_days': ['count', 'mean', 'median', 'max']})
                      .round(2))
            slowest.columns = ['total_complaints', 'avg_response_days', 'median_response_days', 'max_response_days']
            slowest = slowest[slowest['total_complaints'] >= 20].sort_values('avg_response_days', ascending=False).head(10)
            results['slowest_types'] = slowest.reset_index()
        
        # 2. Performance por borough
        borough_stats = df.groupby('borough').agg({
            'unique_key': 'count',
            'status': lambda x: (x == 'Closed').sum(),
            'response_time_days': lambda x: x.mean() if x.notna().any() else None
        }).round(2)
        borough_stats.columns = ['total_complaints', 'closed_complaints', 'avg_response_days']
        borough_stats['closure_rate_pct'] = (borough_stats['closed_complaints'] / borough_stats['total_complaints'] * 100).round(2)
        borough_stats['performance_rank'] = borough_stats['avg_response_days'].rank()
        results['borough_performance'] = borough_stats.reset_index()
        
        # 3. Percentuais de resolução
        if len(closed_df) > 0:
            resolution_stats = []
            for category in closed_df['complaint_category'].unique():
                cat_df = closed_df[closed_df['complaint_category'] == category]
                if len(cat_df) >= 30:  # Mínimo de casos
                    pct_24h = (cat_df['response_time_days'] <= 1).mean() * 100
                    pct_48h = (cat_df['response_time_days'] <= 2).mean() * 100
                    pct_7d = (cat_df['response_time_days'] <= 7).mean() * 100
                    
                    resolution_stats.append({
                        'complaint_category': category,
                        'total_closed': len(cat_df),
                        'pct_resolved_24h': round(pct_24h, 2),
                        'pct_resolved_48h': round(pct_48h, 2),
                        'pct_resolved_7d': round(pct_7d, 2)
                    })
            
            results['resolution_percentages'] = pd.DataFrame(resolution_stats)
        
        return results
        
    except Exception as e:
        st.error(f"Erro nas análises: {e}")
        return {}

def get_analysis_context(analysis_type):
    """
    Retorna o contexto, perguntas e ações para cada tipo de análise
    """
    contexts = {
        "overview": {
            "questions": [
                "🔍 Qual o volume total de solicitações de serviço público em NYC?",
                "⚡ Qual é o tempo médio de resposta da prefeitura?",
                "🏙️ Quais boroughs têm mais demanda por serviços públicos?",
                "📊 Quais são os tipos de reclamações mais frequentes?"
            ],
            "insights": [
                "**Gestão de Recursos:** Identificar boroughs com maior demanda para alocação adequada de equipes",
                "**Priorização de Serviços:** Focar nos tipos de reclamação mais comuns",
                "**Benchmarking:** Estabelecer metas de tempo de resposta baseadas nos dados históricos",
                "**Planejamento Orçamentário:** Dimensionar recursos com base no volume de demandas"
            ],
            "actions": [
                "✅ **Redistribuir equipes** para boroughs com maior volume de chamados",
                "✅ **Criar campanhas preventivas** para os tipos de reclamação mais frequentes",
                "✅ **Definir SLAs** (Service Level Agreements) realistas baseados no histórico",
                "✅ **Investir em infraestrutura** nas áreas com mais solicitações"
            ]
        },
        "temporal": {
            "questions": [
                "📅 Em que dias da semana há mais solicitações de serviço?",
                "⏰ Quais são os horários de pico para abertura de chamados?",
                "🌤️ Há padrões sazonais na demanda por serviços públicos?",
                "📈 Como a demanda varia ao longo do ano?"
            ],
            "insights": [
                "**Otimização de Plantões:** Concentrar mais funcionários nos horários e dias de pico",
                "**Manutenção Preventiva:** Realizar serviços preventivos nos períodos de menor demanda",
                "**Campanhas Educativas:** Lançar campanhas nos períodos que antecedem picos históricos",
                "**Gestão de Recursos:** Planejar férias e treinamentos nos períodos de baixa demanda"
            ],
            "actions": [
                "✅ **Ajustar escalas de trabalho** nos horários e dias de maior demanda",
                "✅ **Implementar sistema de plantão** nos finais de semana se necessário",
                "✅ **Programar manutenções preventivas** nos períodos de baixa demanda",
                "✅ **Criar alertas automáticos** para gestores em períodos de pico"
            ]
        },
        "response_time": {
            "questions": [
                "⏱️ Quais tipos de reclamação demoram mais para serem resolvidas?",
                "🏙️ Há diferenças significativas no tempo de resposta entre os boroughs?",
                "🎯 Qual percentual de chamados é resolvido em 24h, 48h e 7 dias?",
                "🚨 Quais são os gargalos no processo de atendimento?"
            ],
            "insights": [
                "**Gargalos Operacionais:** Identificar tipos de serviço que precisam de mais recursos",
                "**Equidade Territorial:** Garantir tempo de resposta similar entre boroughs",
                "**Metas de Performance:** Estabelecer KPIs realistas para cada tipo de serviço",
                "**Melhoria de Processos:** Otimizar fluxos para tipos de reclamação mais demorados"
            ],
            "actions": [
                "✅ **Contratar mais especialistas** para categorias com maior tempo de resposta",
                "✅ **Criar força-tarefa** para boroughs com performance inferior",
                "✅ **Implementar triagem automática** para priorizar chamados urgentes",
                "✅ **Revisar processos internos** das categorias mais lentas"
            ]
        },
        "predictive": {
            "questions": [
                "🔮 É possível prever o tempo de resposta de um novo chamado?",
                "📊 Quais fatores mais influenciam no tempo de resolução?",
                "⚡ Como otimizar o processo de atendimento com base nos padrões identificados?",
                "🎯 Quais chamados devem ser priorizados automaticamente?"
            ],
            "insights": [
                "**Triagem Inteligente:** Usar ML para classificar automaticamente a prioridade dos chamados",
                "**Alocação Preditiva:** Prever demanda futura para melhor distribuição de recursos",
                "**Alertas Preventivos:** Identificar chamados que podem demorar muito e intervir precocemente",
                "**Otimização de Rotas:** Predizer localização e tipo para otimizar rotas das equipes"
            ],
            "actions": [
                "✅ **Implementar sistema de triagem automática** baseado no modelo preditivo",
                "✅ **Criar dashboards preditivos** para gestores tomarem decisões proativas",
                "✅ **Desenvolver app cidadão** com estimativa de tempo de resposta",
                "✅ **Automatizar escalation** de chamados que excedem tempo previsto"
            ]
        },
        "advanced": {
            "questions": [
                "📈 Quais são as tendências de longo prazo na qualidade dos serviços públicos?",
                "🎯 Quais boroughs têm melhor performance geral?",
                "📊 Como as diferentes categorias de serviço se comparam em eficiência?",
                "💡 Onde estão as maiores oportunidades de melhoria?"
            ],
            "insights": [
                "**Benchmarking Entre Boroughs:** Identificar melhores práticas de boroughs mais eficientes",
                "**ROI de Investimentos:** Priorizar investimentos nas áreas com maior impacto",
                "**Satisfaction Score:** Correlacionar tempo de resposta com satisfação do cidadão",
                "**Continuous Improvement:** Estabelecer métricas para melhoria contínua"
            ],
            "actions": [
                "✅ **Replicar boas práticas** de boroughs eficientes nos demais",
                "✅ **Criar programa de melhoria contínua** com metas trimestrais",
                "✅ **Implementar pesquisas de satisfação** correlacionadas aos dados",
                "✅ **Desenvolver ranking interno** para incentivar melhoria entre equipes"
            ]
        }
    }
    
    return contexts.get(analysis_type, {})

def setup_data_pipeline():
    """Executa pipeline de dados se necessário"""
    if not os.path.exists('data/nyc_311_2023_clean.csv'):
        with st.spinner('Configurando dados pela primeira vez...'):
            progress_bar = st.progress(0)
            
            # 1. Gerar dados sintéticos
            st.info("Gerando dados sintéticos...")
            from data_acquisition import generate_synthetic_311_data, save_data
            df = generate_synthetic_311_data(n_records=25000)  # Otimizado
            save_data(df)
            progress_bar.progress(40)
            
            # 2. Limpar dados
            st.info("Processando e limpando dados...")
            from data_cleaning import DataCleaner
            cleaner = DataCleaner()
            df = cleaner.load_raw_data()
            df = cleaner.normalize_dates(df)
            df = cleaner.calculate_response_time(df)
            df = cleaner.clean_outliers(df)
            df = cleaner.standardize_categories(df)
            df = cleaner.handle_missing_values(df)
            df = cleaner.add_derived_features(df)
            cleaner.save_cleaned_data(df)
            progress_bar.progress(80)
            
            # 3. Treinar modelo
            st.info("Treinando modelo de ML...")
            from predictive_model import PredictiveModel
            try:
                predictor = PredictiveModel()
                X, y = predictor.prepare_features(df)
                predictor.train_random_forest(X, y)
                predictor.save_model()
            except Exception as e:
                st.warning(f"Modelo não foi treinado: {e}")
            
            progress_bar.progress(100)
            st.success("Configuração inicial concluída!")
            st.rerun()

def render_overview_page(df):
    """Página de visão geral"""
    st.header("📊 Visão Geral dos Dados")
    
    # Contexto da análise
    context = get_analysis_context("overview")
    
    with st.expander("🎯 O que esta análise responde?", expanded=False):
        st.markdown("### Perguntas Estratégicas:")
        for question in context.get("questions", []):
            st.markdown(f"- {question}")
        
        st.markdown("### 💡 Insights Principais:")
        for insight in context.get("insights", []):
            st.markdown(f"- {insight}")
        
        st.markdown("### 🚀 Ações Recomendadas:")
        for action in context.get("actions", []):
            st.markdown(f"- {action}")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Chamados", f"{len(df):,}")
    
    with col2:
        closed_count = len(df[df['status'] == 'Fechado'])
        closure_rate = (closed_count / len(df)) * 100
        st.metric("Taxa de Fechamento", f"{closure_rate:.1f}%")
    
    with col3:
        closed_df = df[df['status'] == 'Fechado']
        if len(closed_df) > 0:
            avg_response = closed_df['response_time_days'].mean()
            st.metric("Tempo Médio", f"{avg_response:.1f} dias")
        else:
            st.metric("Tempo Médio", "N/A")
    
    with col4:
        unique_types = df['complaint_type'].nunique()
        st.metric("Tipos Diferentes", unique_types)
    
    st.divider()
    
    # Textos explicativos para os gráficos
    st.markdown("""
    ### 📈 Análise da Distribuição de Chamados
    
    **O que analisamos:** A distribuição geográfica e por categoria dos chamados 311 
    nos ajuda a identificar onde e quais tipos de problemas urbanos são mais frequentes.
    
    **Por que é importante:** Esta informação é crucial para:
    - 🎯 **Alocação de Recursos**: Concentrar equipes nas áreas de maior demanda
    - 📋 **Priorização**: Focar nos problemas mais recorrentes
    - 💰 **Orçamento**: Dimensionar investimentos baseados na demanda real
    """)
    
    # Gráficos modernos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuição por Borough")
        borough_counts = df['borough'].value_counts()
        fig_borough = px.pie(
            values=borough_counts.values,
            names=borough_counts.index,
            title='Chamados por Borough',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_borough.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_borough, use_container_width=True)
    
    with col2:
        st.subheader("Top 10 Tipos de Reclamação")
        complaint_counts = df['complaint_type'].value_counts().head(10)
        fig_complaints = px.bar(
            x=complaint_counts.values,
            y=complaint_counts.index,
            orientation='h',
            title='Tipos Mais Frequentes',
            color=complaint_counts.values,
            color_continuous_scale='viridis'
        )
        fig_complaints.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        st.plotly_chart(fig_complaints, use_container_width=True)

def render_temporal_analysis_page(df):
    """Página de análise temporal"""
    st.header("⏰ Análise Temporal")
    
    # Contexto da análise
    context = get_analysis_context("temporal")
    
    with st.expander("🎯 O que esta análise responde?", expanded=False):
        st.markdown("### Perguntas Estratégicas:")
        for question in context.get("questions", []):
            st.markdown(f"- {question}")
        
        st.markdown("### 💡 Insights Principais:")
        for insight in context.get("insights", []):
            st.markdown(f"- {insight}")
        
        st.markdown("### 🚀 Ações Recomendadas:")
        for action in context.get("actions", []):
            st.markdown(f"- {action}")
    
    # Texto explicativo
    st.markdown("""
    ### 📅 Análise de Padrões Temporais
    
    **O que analisamos:** Os padrões temporais nos mostram quando os cidadãos mais 
    precisam de serviços públicos, revelando tendências por dia da semana, horário e época do ano.
    
    **Importância Estratégica:**
    - 👥 **Dimensionamento de Equipes**: Ajustar plantões conforme demanda real
    - ⚡ **Resposta Rápida**: Antecipar picos de demanda
    - 💡 **Eficiência Operacional**: Concentrar recursos nos momentos críticos
    """)
    
    # Filtros
    col1, col2 = st.columns(2)
    with col1:
        selected_borough = st.selectbox(
            "Filtrar por Borough",
            options=['Todos'] + list(df['borough'].unique())
        )
    
    with col2:
        selected_category = st.selectbox(
            "Filtrar por Categoria",
            options=['Todas'] + list(df['complaint_category'].unique())
        )
    
    # Aplicar filtros
    filtered_df = df.copy()
    if selected_borough != 'Todos':
        filtered_df = filtered_df[filtered_df['borough'] == selected_borough]
    if selected_category != 'Todas':
        filtered_df = filtered_df[filtered_df['complaint_category'] == selected_category]
    
    # Gráfico de série temporal
    st.subheader("📈 Volume de Chamados ao Longo do Tempo")
    st.markdown("**Análise:** Esta série temporal revela tendências sazonais e picos de demanda que podem estar relacionados a eventos, clima ou questões urbanas específicas.")
    
    daily_counts = filtered_df.groupby(filtered_df['created_date'].dt.date).size().reset_index()
    daily_counts.columns = ['data', 'chamados']
    
    fig_time = px.line(
        daily_counts,
        x='data',
        y='chamados',
        title='Evolução Diária do Número de Chamados',
        labels={'data': 'Data', 'chamados': 'Número de Chamados'}
    )
    fig_time.update_layout(
        title_x=0.5,
        xaxis_title="Período",
        yaxis_title="Quantidade de Chamados",
        hovermode='x unified'
    )
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Padrões por dia da semana e hora
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Padrão Semanal")
        st.markdown("**Análise:** Este gráfico mostra como a demanda varia durante a semana, revelando padrões de comportamento urbano.")
        
        # Ordem correta dos dias da semana em português
        weekday_order = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
        weekday_counts = filtered_df['created_weekday'].value_counts().reindex(weekday_order, fill_value=0)
        
        # Labels abreviados para melhor visualização
        weekday_labels = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
        
        # Corrigir possíveis valores ausentes ou inconsistentes
        weekday_counts = weekday_counts.reindex(weekday_order, fill_value=0)
        # Garantir que os labels estejam alinhados com os valores
        fig_weekday = px.bar(
            x=weekday_labels,
            y=weekday_counts.values.tolist(),
            title='Distribuição de Chamados por Dia da Semana',
            labels={'x': 'Dia da Semana', 'y': 'Número de Chamados'}
        )
        fig_weekday.update_layout(title_x=0.5)
        st.plotly_chart(fig_weekday, use_container_width=True)
    
    with col2:
        st.subheader("🕐 Padrão Horário")
        st.markdown("**Análise:** Identifica os horários de maior demanda para otimização de recursos e plantões.")
        
        hour_counts = filtered_df['created_hour'].value_counts().sort_index()
        
        fig_hour = px.bar(
            x=hour_counts.index,
            y=hour_counts.values,
            title='Distribuição de Chamados por Hora do Dia',
            labels={'x': 'Hora do Dia', 'y': 'Número de Chamados'}
        )
        fig_hour.update_layout(title_x=0.5, xaxis_title="Hora", yaxis_title="Chamados")
        st.plotly_chart(fig_hour, use_container_width=True)

def render_response_analysis_page(df, sql_results):
    """Página de análise de tempo de resposta"""
    st.header("⏱️ Análise de Tempo de Resposta")
    
    # Contexto da análise
    context = get_analysis_context("response_time")
    
    with st.expander("🎯 O que esta análise responde?", expanded=False):
        st.markdown("### Perguntas Estratégicas:")
        for question in context.get("questions", []):
            st.markdown(f"- {question}")
        
        st.markdown("### 💡 Insights Principais:")
        for insight in context.get("insights", []):
            st.markdown(f"- {insight}")
        
        st.markdown("### 🚀 Ações Recomendadas:")
        for action in context.get("actions", []):
            st.markdown(f"- {action}")
    
    # Texto explicativo
    st.markdown("""
    ### ⏱️ Análise de Performance de Atendimento
    
    **O que analisamos:** O tempo de resposta é o indicador mais crítico da qualidade 
    dos serviços públicos. Analisamos quanto tempo leva para resolver cada tipo de problema.
    
    **Por que é crucial:**
    - 🎯 **SLA Management**: Estabelecer metas realistas de atendimento
    - ⚖️ **Equidade**: Garantir que todos os boroughs recebam serviço de qualidade similar  
    - 🚨 **Identificação de Gargalos**: Encontrar onde o processo trava
    - 📊 **Benchmarking**: Comparar performance entre categorias e locais
    """)
    
    # Filtrar apenas registros fechados  
    closed_df = df[df['status'] == 'Fechado'].copy()
    
    if len(closed_df) == 0:
        st.warning("Nenhum registro fechado encontrado para análise de tempo de resposta.")
        return
    
    # Estatísticas gerais
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_response = closed_df['response_time_days'].mean()
        st.metric("Tempo Médio", f"{avg_response:.1f} dias")
    
    with col2:
        median_response = closed_df['response_time_days'].median()
        st.metric("Tempo Mediano", f"{median_response:.1f} dias")
    
    with col3:
        max_response = closed_df['response_time_days'].max()
        st.metric("Tempo Máximo", f"{max_response:.1f} dias")
    
    # Box plot por categoria
    st.subheader("📦 Variação do Tempo de Resposta por Categoria")
    st.markdown("""
    **Análise:** Este gráfico box plot revela a distribuição do tempo de resposta para cada 
    categoria de serviço. As caixas mostram a variabilidade típica, enquanto os pontos externos 
    indicam casos excepcionais que podem precisar de atenção especial.
    """)
    
    fig_box = px.box(
        closed_df,
        x='complaint_category',
        y='response_time_days',
        title='Distribuição do Tempo de Resposta por Categoria',
        labels={
            'complaint_category': 'Categoria da Reclamação',
            'response_time_days': 'Tempo de Resposta (dias)'
        }
    )
    fig_box.update_layout(
        xaxis_tickangle=-45,
        title_x=0.5,
        hovermode='closest'
    )
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Heatmap por borough e categoria
    st.subheader("🗺️ Tempo Médio de Resposta por Borough e Categoria")
    heatmap_data = closed_df.groupby(['borough', 'complaint_category'])['response_time_days'].mean().unstack()
    
    fig_heatmap = px.imshow(
        heatmap_data,
        title='Heatmap: Tempo Médio de Resposta (dias)',
        aspect="auto",
        color_continuous_scale='RdYlBu_r'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Resultados SQL se disponíveis
    if sql_results and sql_results.get('slowest_types') is not None:
        st.subheader("🐌 Tipos Mais Demorados (Análise SQL)")
        st.dataframe(sql_results['slowest_types'])
    
    if sql_results and sql_results.get('resolution_percentages') is not None:
        st.subheader("📊 Percentuais de Resolução")
        st.dataframe(sql_results['resolution_percentages'])

def render_predictive_model_page(df):
    """Página do modelo preditivo modernizada"""
    st.header("🔮 Modelo Preditivo de Machine Learning")
    
    # Contexto da análise
    context = get_analysis_context("predictive")
    
    with st.expander("🎯 O que esta análise responde?", expanded=False):
        st.markdown("### Perguntas Estratégicas:")
        for question in context.get("questions", []):
            st.markdown(f"- {question}")
        
        st.markdown("### 💡 Insights Principais:")
        for insight in context.get("insights", []):
            st.markdown(f"- {insight}")
        
        st.markdown("### 🚀 Ações Recomendadas:")
        for action in context.get("actions", []):
            st.markdown(f"- {action}")
    
    # Texto explicativo
    st.markdown("""
    ### 🤖 Predição Inteligente de Tempo de Resposta
    
    **O que faz:** Este modelo de Machine Learning usa Random Forest para prever o tempo 
    de resposta de novos chamados baseado em padrões históricos.
    
    **Features Avançadas do Modelo:**
    - 🏙️ **Interação Borough-Categoria**: Como localização e tipo se relacionam
    - ⏰ **Horário de Pico**: Identifica automaticamente períodos de alta demanda  
    - 🎊 **Indicadores Sazonais**: Detecta padrões de final de ano
    - 📊 **13 Features**: Combinação otimizada de variáveis preditivas
    
    **Aplicação Prática:** Permite triagem automática e alocação inteligente de recursos.
    """)
    
    # Verificar se modelo existe
    if not os.path.exists('models/nyc_311_response_time_model.pkl'):
        st.warning("Modelo preditivo não encontrado. Treinando modelo agora...")
        
        with st.spinner("Treinando modelo..."):
            try:
                from predictive_model import PredictiveModel
                predictor = PredictiveModel()
                X, y = predictor.prepare_features(df[df['status'] == 'Fechado'])
                results = predictor.train_random_forest(X, y)
                predictor.save_model()
                st.success("Modelo treinado com sucesso!")
            except Exception as e:
                st.error(f"Erro ao treinar modelo: {e}")
                return
    
    # Carregador do modelo
    @st.cache_resource
    def load_model():
        from predictive_model import PredictiveModel
        predictor = PredictiveModel()
        predictor.load_model()
        return predictor
    
    try:
        predictor = load_model()
        
        # Interface moderna para predição
        st.subheader("🎯 Simulador de Tempo de Resposta")
        st.markdown("**Como usar:** Selecione as características do chamado para obter uma predição do tempo de resposta esperado.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            complaint_type = st.selectbox(
                "Tipo de Reclamação",
                options=[
                    'Ruído - Residencial',
                    'Aquecimento/Água Quente', 
                    'Condição da Rua',
                    'Condições Insalubres',
                    'Garagem Bloqueada',
                    'Sistema de Água',
                    'Estacionamento Irregular',
                    'Encanamento',
                    'Porta/Janela',
                    'Roedores'
                ]
            )
            
            borough = st.selectbox(
                "Borough",
                options=['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
            )
        
        with col2:
            created_hour = st.slider("Hora do Dia", 0, 23, 14)
            
            day_options = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
            weekday = st.selectbox("Dia da Semana", options=day_options)
        
        with col3:
            is_weekend = weekday in ['Sábado', 'Domingo']
            st.write(f"Final de Semana: {'Sim' if is_weekend else 'Não'}")
            
            is_priority = complaint_type in ['Heat/Hot Water', 'Water System', 'UNSANITARY CONDITION']
            st.write(f"Prioridade: {'Alta' if is_priority else 'Normal'}")
        
        # Fazer predição
        if st.button("Prever Tempo de Resposta", type="primary"):
            try:
                predicted_time = predictor.predict_response_time(
                    complaint_type=complaint_type,
                    borough=borough,
                    created_hour=created_hour,
                    is_weekend=is_weekend,
                    weekday=weekday
                )
                
                # Resultado com interpretação
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Tempo Previsto", f"{predicted_time:.1f} dias")
                
                with col2:
                    if predicted_time <= 1:
                        st.success("Resposta Rápida (≤ 1 dia)")
                    elif predicted_time <= 7:
                        st.warning("Resposta Moderada (1-7 dias)")
                    else:
                        st.error("Resposta Demorada (> 7 dias)")
                
                # Análise contextual
                if is_priority:
                    st.info("ℹ️ Este é um tipo de chamado prioritário")
                
                if created_hour < 6 or created_hour > 22:
                    st.info("ℹ️ Chamado fora do horário comercial pode demorar mais")
                
            except Exception as e:
                st.error(f"Erro na predição: {e}")
        
        # Informações do modelo
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Informações do Modelo")
            st.write("""
            - **Algoritmo:** Random Forest Regressor
            - **Features:** Categoria, Borough, Tempo, Prioridade
            - **Dados:** Registros fechados de 2023
            - **Métrica:** MAE (Mean Absolute Error)
            """)
        
        with col2:
            st.subheader("Fatores Importantes")
            try:
                # Mostrar feature importance se disponível
                if hasattr(predictor.model, 'feature_importances_'):
                    importance = dict(zip(predictor.feature_columns[:5], 
                                        predictor.model.feature_importances_[:5]))
                    
                    fig_importance = px.bar(
                        x=list(importance.values()),
                        y=list(importance.keys()),
                        orientation='h',
                        title='Top 5 Features Mais Importantes'
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
            except:
                st.write("Feature importance não disponível")
                
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")

def render_sql_analysis_page(sql_results):
    """Página de análises avançadas"""
    st.header("� Análises Avançadas")
    
    # Contexto da análise
    context = get_analysis_context("advanced")
    
    with st.expander("🎯 O que esta análise responde?", expanded=False):
        st.markdown("### Perguntas Estratégicas:")
        for question in context.get("questions", []):
            st.markdown(f"- {question}")
        
        st.markdown("### 💡 Insights Principais:")
        for insight in context.get("insights", []):
            st.markdown(f"- {insight}")
        
        st.markdown("### 🚀 Ações Recomendadas:")
        for action in context.get("actions", []):
            st.markdown(f"- {action}")
    
    # Texto explicativo
    st.markdown("""
    ### 🎯 Análises de Performance e Benchmarking
    
    **O que analisamos:** Comparamos a eficiência entre diferentes boroughs e categorias 
    de serviço para identificar melhores práticas e oportunidades de melhoria.
    
    **Importância para Gestão:**
    - 🏆 **Benchmarking**: Identificar líderes em performance para replicar boas práticas
    - 📊 **KPIs Estratégicos**: Métricas para avaliação e melhoria contínua
    - 💰 **ROI de Investimentos**: Priorizar investimentos com maior impacto
    - ⚖️ **Equidade de Serviços**: Garantir qualidade similar em todos os boroughs
    """)
    
    if not sql_results or all(v is None for v in sql_results.values()):
        st.warning("Resultados das análises não encontrados. Execute o setup completo primeiro.")
        return
    
    # Performance por borough
    if sql_results.get('borough_performance') is not None:
        st.subheader("🏆 Ranking de Performance por Borough")
        st.markdown("""
        **Análise:** Este ranking combina taxa de fechamento e tempo médio de resposta 
        para identificar quais boroughs têm melhor performance geral no atendimento.
        """)
        
        borough_perf = sql_results['borough_performance']
        
        # Reformatar para melhor visualização
        borough_perf_display = borough_perf.copy()
        borough_perf_display['Taxa de Fechamento'] = borough_perf_display['closure_rate_pct'].apply(lambda x: f"{x}%")
        borough_perf_display['Tempo Médio (dias)'] = borough_perf_display['avg_response_days'].apply(lambda x: f"{x:.1f}")
        
        st.dataframe(
            borough_perf_display[['borough', 'total_complaints', 'Taxa de Fechamento', 'Tempo Médio (dias)', 'performance_rank']],
            column_config={
                'borough': 'Borough',
                'total_complaints': 'Total de Chamados',
                'performance_rank': 'Ranking de Performance'
            }
        )
    
    # Tipos mais demorados
    if sql_results.get('slowest_types') is not None:
        st.subheader("⏰ Tipos de Reclamação Mais Demorados")
        st.markdown("""
        **Análise:** Identifica os tipos de serviço que consistentemente demoram mais 
        para serem resolvidos, indicando onde podem ser necessários mais recursos ou 
        melhorias de processo.
        """)
        slowest = sql_results['slowest_types']
        
        fig_slowest = px.bar(
            slowest,
            x='avg_response_days',
            y='complaint_type',
            orientation='h',
            title='Tempo Médio de Resposta por Tipo',
            labels={'avg_response_days': 'Tempo Médio (dias)', 'complaint_type': 'Tipo de Reclamação'}
        )
        fig_slowest.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_slowest, use_container_width=True)
    
    # Percentuais de resolução
    if sql_results.get('resolution_percentages') is not None:
        st.subheader("📊 Percentuais de Resolução por Categoria")
        resolution = sql_results['resolution_percentages']
        
        # Gráfico de barras agrupadas
        fig_resolution = go.Figure()
        
        fig_resolution.add_trace(go.Bar(
            name='24h',
            x=resolution['complaint_category'],
            y=resolution['pct_resolved_24h']
        ))
        
        fig_resolution.add_trace(go.Bar(
            name='48h',
            x=resolution['complaint_category'],
            y=resolution['pct_resolved_48h']
        ))
        
        fig_resolution.add_trace(go.Bar(
            name='7 dias',
            x=resolution['complaint_category'],
            y=resolution['pct_resolved_7d']
        ))
        
        fig_resolution.update_layout(
            title='Percentual de Chamados Resolvidos por Tempo',
            xaxis_title='Categoria',
            yaxis_title='Percentual (%)',
            barmode='group'
        )
        
        st.plotly_chart(fig_resolution, use_container_width=True)

def render_diagnostics_page(df, sql_results):
    """Página de diagnósticos com resumo executivo"""
    st.header("🩺 Diagnósticos do Sistema 311")
    
    st.markdown("""
    ### 📋 Resumo Executivo
    
    Esta página apresenta um diagnóstico completo do sistema de atendimento público 311 de NYC, 
    consolidando os principais indicadores e fornecendo insights estratégicos baseados em dados.
    """)
    
    # Métricas principais em destaque
    col1, col2, col3, col4 = st.columns(4)
    
    closed_df = df[df['status'] == 'Fechado']
    total_records = len(df)
    closed_records = len(closed_df)
    avg_response_time = closed_df['response_time_days'].mean() if len(closed_df) > 0 else 0
    closure_rate = (closed_records / total_records) * 100
    
    with col1:
        st.metric(
            label="📊 Volume Total", 
            value=f"{total_records:,}",
            help="Total de solicitações registradas no sistema"
        )
    
    with col2:
        st.metric(
            label="✅ Taxa de Resolução", 
            value=f"{closure_rate:.1f}%",
            delta=f"{closure_rate-70:.1f}% vs meta 70%",
            delta_color="normal" if closure_rate >= 70 else "inverse"
        )
    
    with col3:
        st.metric(
            label="⏱️ Tempo Médio", 
            value=f"{avg_response_time:.1f} dias",
            delta=f"{2.0-avg_response_time:.1f} dias vs meta 2d",
            delta_color="normal" if avg_response_time <= 2.0 else "inverse"
        )
    
    with col4:
        priority_resolved = len(closed_df[closed_df['is_priority']]) if len(closed_df) > 0 else 0
        priority_total = len(df[df['is_priority']])
        priority_rate = (priority_resolved / priority_total * 100) if priority_total > 0 else 0
        st.metric(
            label="🚨 Prioridade Resolvida", 
            value=f"{priority_rate:.1f}%",
            help="% de chamados prioritários resolvidos"
        )
    
    st.divider()
    
    # Diagnóstico por Borough
    st.subheader("🏙️ Diagnóstico por Borough")
    
    borough_analysis = []
    for borough in df['borough'].unique():
        borough_df = df[df['borough'] == borough]
        borough_closed = borough_df[borough_df['status'] == 'Fechado']
        
        total_calls = len(borough_df)
        closure_rate_b = len(borough_closed) / total_calls * 100
        avg_time = borough_closed['response_time_days'].mean() if len(borough_closed) > 0 else 0
        
        # Classificação de performance
        if closure_rate_b >= 75 and avg_time <= 2.0:
            performance = "🟢 Excelente"
        elif closure_rate_b >= 70 and avg_time <= 3.0:
            performance = "🟡 Bom"
        elif closure_rate_b >= 60 and avg_time <= 5.0:
            performance = "🟠 Regular"
        else:
            performance = "🔴 Crítico"
        
        borough_analysis.append({
            'Borough': borough,
            'Total de Chamados': total_calls,
            'Taxa de Resolução (%)': f"{closure_rate_b:.1f}%",
            'Tempo Médio (dias)': f"{avg_time:.1f}",
            'Performance': performance
        })
    
    borough_df_display = pd.DataFrame(borough_analysis)
    st.dataframe(borough_df_display, use_container_width=True)
    
    # Análise de categorias críticas
    st.subheader("⚠️ Categorias que Precisam de Atenção")
    
    category_analysis = []
    for category in df['complaint_category'].unique():
        cat_df = df[df['complaint_category'] == category]
        cat_closed = cat_df[cat_df['status'] == 'Fechado']
        
        total_calls = len(cat_df)
        if total_calls < 50:  # Ignorar categorias com poucos chamados
            continue
            
        closure_rate_c = len(cat_closed) / total_calls * 100
        avg_time = cat_closed['response_time_days'].mean() if len(cat_closed) > 0 else 0
        
        # Identificar categorias problemáticas
        if closure_rate_c < 70 or avg_time > 3.0:
            urgency = "🔴 Alta" if closure_rate_c < 60 or avg_time > 5.0 else "🟠 Média"
            
            category_analysis.append({
                'Categoria': category,
                'Total de Chamados': total_calls,
                'Taxa de Resolução (%)': f"{closure_rate_c:.1f}%",
                'Tempo Médio (dias)': f"{avg_time:.1f}",
                'Urgência': urgency,
                'Principal Problema': 'Taxa baixa' if closure_rate_c < 70 else 'Tempo alto'
            })
    
    if category_analysis:
        category_df_display = pd.DataFrame(category_analysis)
        category_df_display = category_df_display.sort_values('Urgência', ascending=False)
        st.dataframe(category_df_display, use_container_width=True)
    else:
        st.success("✅ Todas as categorias principais estão dentro dos parâmetros aceitáveis!")
    
    # Insights e recomendações baseados em ML
    st.subheader("🤖 Insights Baseados em Machine Learning")
    
    st.markdown("""
    **Principais Fatores que Impactam o Tempo de Resposta:**
    
    1. **🏙️ Localização**: Diferentes boroughs apresentam padrões distintos de eficiência
    2. **📋 Tipo de Serviço**: Algumas categorias são consistentemente mais demoradas
    3. **⏰ Horário**: Chamados em horários de pico tendem a demorar mais
    4. **📅 Sazonalidade**: Final de ano apresenta tempos de resposta mais elevados
    """)
    
    # Recomendações estratégicas
    st.subheader("🎯 Recomendações Estratégicas Baseadas nos Dados")
    
    recommendations = []
    
    # Análise automática para gerar recomendações
    if closure_rate < 75:
        recommendations.append("📈 **Melhoria da Taxa de Resolução**: Implementar processo de follow-up para chamados em aberto há mais de 7 dias")
    
    if avg_response_time > 2.5:
        recommendations.append("⚡ **Otimização de Tempo**: Criar força-tarefa para categorias com tempo médio superior a 3 dias")
    
    # Análise por borough
    worst_borough = min(borough_analysis, key=lambda x: float(x['Taxa de Resolução (%)'].rstrip('%')))
    if float(worst_borough['Taxa de Resolução (%)'].rstrip('%')) < 70:
        recommendations.append(f"🏙️ **Foco em {worst_borough['Borough']}**: Borough com performance mais baixa precisa de recursos adicionais")
    
    # Se há categorias problemáticas
    if category_analysis:
        worst_category = category_analysis[0]['Categoria']
        recommendations.append(f"🔧 **Categoria Crítica**: {worst_category} precisa de revisão de processo ou recursos especializados")
    
    # Recomendação de ML
    recommendations.append("🤖 **Implementação de IA**: Usar modelo preditivo para triagem automática e alocação inteligente de recursos")
    
    # Exibir recomendações
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")
    
    # Status geral do sistema
    st.subheader("🎖️ Status Geral do Sistema")
    
    # Calcular score geral
    score = 0
    if closure_rate >= 75: score += 25
    elif closure_rate >= 70: score += 20
    elif closure_rate >= 65: score += 15
    elif closure_rate >= 60: score += 10
    
    if avg_response_time <= 1.5: score += 25
    elif avg_response_time <= 2.0: score += 20
    elif avg_response_time <= 3.0: score += 15
    elif avg_response_time <= 5.0: score += 10
    
    # Bonus por volume processado
    if total_records >= 10000: score += 25
    elif total_records >= 5000: score += 20
    elif total_records >= 1000: score += 15
    
    # Bonus por prioridades
    if priority_rate >= 80: score += 25
    elif priority_rate >= 70: score += 20
    elif priority_rate >= 60: score += 15
    
    # Classificar sistema
    if score >= 90:
        status_emoji = "🟢"
        status_text = "EXCELENTE"
        status_color = "green"
    elif score >= 75:
        status_emoji = "🟡"
        status_text = "BOM"
        status_color = "orange"
    elif score >= 60:
        status_emoji = "🟠"
        status_text = "REGULAR"
        status_color = "orange"
    else:
        status_emoji = "🔴"
        status_text = "CRÍTICO"
        status_color = "red"
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; border: 2px solid {status_color}; border-radius: 10px; background-color: rgba(255,255,255,0.1);">
            <h2>{status_emoji} Sistema: {status_text}</h2>
            <h3>Score: {score}/100</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Próximos passos
    st.subheader("📋 Próximos Passos")
    st.markdown("""
    1. **Monitoramento Contínuo**: Estabelecer dashboard de KPIs em tempo real
    2. **Metas Trimestrais**: Definir objetivos específicos para cada borough e categoria
    3. **Feedback Loop**: Implementar pesquisas de satisfação correlacionadas aos tempos
    4. **Automação**: Expandir uso de ML para predição e otimização automática
    5. **Benchmarking**: Comparar performance com outras cidades de porte similar
    """)

def main():
    """Função principal do dashboard"""
    
    # Título e descrição
    st.title("🏙️ NYC 311 Service Requests Analysis")
    st.markdown("""
    Dashboard interativo para análise dos chamados de atendimento público (311) de Nova York.
    **Objetivo:** Identificar fatores que impactam o tempo de resposta e prever tempos futuros.
    """)
    
    # Verificar se dados existem
    if not os.path.exists('data/nyc_311_2023_clean.csv'):
        st.warning("⚠️ Dados não encontrados. Executando configuração inicial...")
        setup_data_pipeline()
        return
    
    # Carregar dados
    df = load_data()
    sql_results = run_sql_analysis(df) if df is not None else {}
    
    if df is None:
        st.error("❌ Erro ao carregar dados.")
        return
    
    # Sidebar com navegação
    st.sidebar.title("Navegação")
    
    page = st.sidebar.radio(
        "Selecione uma página:",
        [
            "Visão Geral",
            "Análise Temporal", 
            "Tempo de Resposta",
            "Modelo Preditivo",
            "Análises Avançadas",
            "Diagnósticos"
        ]
    )
    
    # Informações do dataset na sidebar
    st.sidebar.divider()
    st.sidebar.subheader("Estatísticas do Dataset")
    st.sidebar.write(f"**Total de registros:** {len(df):,}")
    st.sidebar.write(f"**Período:** {df['created_date'].dt.date.min()} a {df['created_date'].dt.date.max()}")
    st.sidebar.write(f"**Boroughs:** {df['borough'].nunique()}")
    st.sidebar.write(f"**Tipos de reclamação:** {df['complaint_type'].nunique()}")
    
    closed_pct = (len(df[df['status'] == 'Fechado']) / len(df)) * 100
    st.sidebar.write(f"**% Fechados:** {closed_pct:.1f}%")
    
    # Renderizar página selecionada
    if page == "Visão Geral":
        render_overview_page(df)
    elif page == "Análise Temporal":
        render_temporal_analysis_page(df)
    elif page == "Tempo de Resposta":
        render_response_analysis_page(df, sql_results)
    elif page == "Modelo Preditivo":
        render_predictive_model_page(df)
    elif page == "Análises Avançadas":
        render_sql_analysis_page(sql_results)
    elif page == "Diagnósticos":
        render_diagnostics_page(df, sql_results)
    
    # Footer
    st.divider()
    st.markdown("""
    ---
    **NYC 311 Analysis Dashboard** | Demonstração de habilidades em Python para Ciência de Dados  
    Tecnologias: Python, Pandas, Plotly, Streamlit, DuckDB, Scikit-learn
    """)

if __name__ == "__main__":
    main()