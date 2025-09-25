# NYC 311 Service Requests Analysis

## 🎯 Objetivo do Projeto

**Pergunta Central:** "Quais fatores impactam o tempo de resposta dos chamados de atendimento público (311) em Nova York, e como podemos prever esse tempo?"

Este projeto demonstra habilidades fundamentais em ciência de dados através de uma análise completa dos dados públicos de chamados 311 da cidade de Nova York, incluindo manipulação de dados, SQL avançado, visualizações e machine learning.

## 📊 Dataset

- **Fonte:** NYC 311 Service Requests (dataset público oficial)
- **Período:** Ano de 2023 (para manter o projeto gerenciável)
- **Tamanho:** Milhões de registros
- **Variáveis principais:**
  - `created_date`: Data/hora de abertura do chamado
  - `closed_date`: Data/hora de fechamento do chamado
  - `complaint_type`: Tipo de reclamação/serviço
  - `borough`: Distrito de NYC (Manhattan, Brooklyn, Queens, Bronx, Staten Island)
  - `zip_code`: CEP
  - Coordenadas geográficas

## 🔧 Tecnologias Utilizadas

- **Python:** Pandas, NumPy, Matplotlib, Seaborn, Plotly
- **SQL:** SQLite para análises avançadas com window functions
- **Machine Learning:** Scikit-learn com Random Forest e feature engineering avançado
- **Dashboard:** Streamlit com interface moderna e interativa
- **Visualização:** Gráficos interativos e análises em tempo real

## 🚀 Setup Rápido

### **Opção 1: Execução Automática (Windows)**
```bash
# Clone o repositório e navegue até a pasta
cd python-for-data-science

# Execute o setup completo (instala dependências, gera dados, inicia dashboard)
setup.bat
```

### **Opção 2: Setup Manual**
```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Gerar dados sintéticos
python src/data_acquisition.py

# 3. Processar dados
python src/data_cleaning.py

# 4. Iniciar dashboard
streamlit run streamlit_app.py
```

Após executar, acesse: **http://localhost:8501**

## 📁 Estrutura do Projeto

```
python-for-data-science/
├── data/                          # Dados brutos e processados
├── src/                           # Scripts Python modulares
│   ├── data_acquisition.py        # Geração de dados sintéticos realistas
│   ├── data_cleaning.py          # Limpeza e transformação
│   ├── exploratory_analysis.py   # Análise exploratória
│   ├── sql_analysis.py           # Consultas SQL avançadas (SQLite)
│   └── predictive_model.py       # Modelo de ML com feature engineering
├── models/                        # Modelos treinados salvos
├── streamlit_app.py              # Dashboard principal modernizado
├── setup.bat                     # Script de setup automático (Windows)
├── setup.sh                      # Script de setup automático (Linux/Mac)
├── test_setup.py                 # Script de teste completo
└── README.md                     # Este arquivo
```

## 🎯 Features do Dashboard

### **Visão Geral**
- Métricas principais (total de chamados, tempo médio de resposta)
- Distribuição por borough e tipos de reclamação
- KPIs interativos com visualizações modernas

### **Análise Temporal**
- Séries temporais de chamados
- Padrões por dia da semana e hora
- Filtros por borough e categoria
- Análise de sazonalidade avançada

### **Tempo de Resposta**
- Box plots por categoria
- Heatmaps borough vs categoria
- Rankings de tipos mais demorados
- Percentuais de resolução (24h, 48h, 7 dias)

### **Modelo Preditivo**
- Simulador interativo de tempo de resposta
- Features avançadas: interações borough-categoria, horário de pico, final de ano
- Predições em tempo real com Random Forest otimizado
- Interpretação dos resultados

### **Análises Avançadas**
- Análises de performance utilizando pandas
- Rankings de performance por borough
- Estatísticas detalhadas com métricas avançadas
- Tabelas interativas e visualizações aprofundadas

## 🚀 Principais Análises

### 1. **Limpeza e Transformação**
- Normalização de datas e cálculo da variável `tempo_resposta`
- Tratamento de valores nulos e outliers
- Padronização de categorias de chamados

### 2. **Análise Exploratória (EDA)**
- Distribuição de chamados por tipo e distrito
- Tempo médio de resposta por categoria
- Análise temporal: sazonalidade e tendências
- Mapas de calor geograficos

### 3. **Análises com SQLite e Pandas**
- Ranking de tipos de chamado mais demorados
- Estatísticas de performance por borough
- Percentual de chamados resolvidos em prazos específicos
- Análises temporais e métricas avançadas

### 4. **Modelo Preditivo com ML Avançado**
- Random Forest Regressor com feature engineering sofisticado
- Features: interações borough-categoria, indicadores de horário de pico, final de ano
- Avaliação com cross-validation, RMSE e MAE
- Interface interativa para predições em tempo real

## 📈 Principais Insights Esperados

- Identificação dos tipos de chamado mais críticos
- Padrões temporais (dias da semana, horários)
- Diferenças entre distritos de NYC
- Fatores preditivos do tempo de resposta

## 📱 Screenshots do Dashboard

### Dashboard Principal
- Interface limpa e intuitiva com navegação lateral
- Métricas KPI em tempo real
- Gráficos interativos com Plotly

### Modelo Preditivo
- Simulador interativo para prever tempo de resposta
- Inputs customizáveis (tipo, localização, hora)
- Resultados interpretados automaticamente

## 🎓 Habilidades Demonstradas

### **Python para Dados**
- ✅ Pandas para manipulação de dados
- ✅ NumPy para computação numérica
- ✅ Matplotlib/Seaborn para visualizações
- ✅ Plotly para gráficos interativos

### **SQL Avançado**
- ✅ Window functions (média móvel, ranking)
- ✅ CTEs (Common Table Expressions)
- ✅ Aggregações complexas
- ✅ SQLite para análise estruturada

### **Machine Learning**
- ✅ Preparação de features
- ✅ Random Forest Regressor
- ✅ Validation e métricas (MAE, RMSE, R²)
- ✅ Feature importance

### **Desenvolvimento**
- ✅ Código modular e reutilizável
- ✅ Dashboard interativo com Streamlit
- ✅ Setup automatizado
- ✅ Documentação completa

## 🔧 Requisitos Técnicos

- Python 3.8+
- 8GB RAM (recomendado)
- 1GB espaço em disco
- Windows/Linux/MacOS

## 📞 Contato

- **Email:** jesseff20@gmail.com
- **LinkedIn:** /in/jesse-fernandes
- **GitHub:** /jessefernandes

---

**Desenvolvido por:** Jesse Fernandes | **Data:** Setembro 2025  
**Objetivo:** Portfolio de Ciência de Dados demonstrando habilidades completas em Python