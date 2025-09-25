"""
Módulo para análises SQL avançadas com SQLite
"""

import sqlite3
import pandas as pd
import os

class SQLAnalyzer:
    def __init__(self):
        self.data_dir = 'data'
        self.conn = None
    
    def connect_database(self, db_name='nyc_311.db'):
        """
        Conecta ao banco SQLite
        """
        db_path = os.path.join(self.data_dir, db_name)
        self.conn = sqlite3.connect(db_path)
        print(f"✓ Conectado ao banco: {db_path}")
        return self.conn
    
    def load_data_to_db(self, csv_file='nyc_311_2023_clean.csv', table_name='nyc_311'):
        """
        Carrega dados CSV para o banco SQLite
        """
        csv_path = os.path.join(self.data_dir, csv_file)
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")
        
        print(f"Carregando {csv_file} para tabela {table_name}...")
        
        # Carregar CSV com pandas
        df = pd.read_csv(csv_path)
        
        # Salvar no SQLite
        df.to_sql(table_name, self.conn, if_exists='replace', index=False)
        
        # Verificar dados carregados
        cursor = self.conn.cursor()
        count = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"✓ {count:,} registros carregados na tabela {table_name}")
        
        return count
    
    def query_top_slowest_complaint_types(self, limit=10):
        """
        Ranking dos tipos de reclamação mais demorados
        """
        query = f"""
        SELECT 
            complaint_type,
            COUNT(*) as total_complaints,
            ROUND(AVG(response_time_days), 2) as avg_response_days,
            ROUND(MEDIAN(response_time_days), 2) as median_response_days,
            ROUND(MAX(response_time_days), 2) as max_response_days
        FROM nyc_311 
        WHERE status = 'Closed' 
        AND response_time_days IS NOT NULL
        GROUP BY complaint_type
        HAVING COUNT(*) >= 50  -- Pelo menos 50 casos
        ORDER BY avg_response_days DESC
        LIMIT {limit}
        """
        
        result = pd.read_sql_query(query, self.conn)
        print(f"✓ Top {limit} tipos de reclamação mais demorados:")
        return result
    
    def query_response_time_moving_average(self, window_days=7):
        """
        Média móvel do tempo de resposta por data (versão SQLite)
        """
        query = f"""
        WITH daily_avg AS (
            SELECT 
                DATE(created_date) as date,
                AVG(response_time_days) as daily_avg_response
            FROM nyc_311 
            WHERE status = 'Closed' 
            AND response_time_days IS NOT NULL
            GROUP BY DATE(created_date)
        )
        SELECT 
            date,
            daily_avg_response,
            (SELECT AVG(daily_avg_response) 
             FROM daily_avg d2 
             WHERE d2.date >= date(daily_avg.date, '-{window_days-1} days') 
             AND d2.date <= daily_avg.date) as moving_avg_{window_days}d
        FROM daily_avg
        ORDER BY date
        """
        
        result = pd.read_sql_query(query, self.conn)
        print(f"✓ Média móvel de {window_days} dias calculada para {len(result)} dias")
        return result
    
    def query_resolution_percentages(self):
        """
        Percentual de chamados resolvidos em diferentes períodos
        """
        query = """
        WITH resolution_times AS (
            SELECT 
                complaint_category,
                borough,
                response_time_days,
                CASE 
                    WHEN response_time_days <= 1 THEN '24h'
                    WHEN response_time_days <= 2 THEN '48h'
                    WHEN response_time_days <= 7 THEN '7d'
                    WHEN response_time_days <= 30 THEN '30d'
                    ELSE '30d+'
                END as resolution_period
            FROM nyc_311 
            WHERE status = 'Closed' 
            AND response_time_days IS NOT NULL
        )
        SELECT 
            complaint_category,
            COUNT(*) as total_closed,
            ROUND(
                COUNT(CASE WHEN resolution_period = '24h' THEN 1 END) * 100.0 / COUNT(*), 
                2
            ) as pct_resolved_24h,
            ROUND(
                COUNT(CASE WHEN resolution_period IN ('24h', '48h') THEN 1 END) * 100.0 / COUNT(*), 
                2
            ) as pct_resolved_48h,
            ROUND(
                COUNT(CASE WHEN resolution_period IN ('24h', '48h', '7d') THEN 1 END) * 100.0 / COUNT(*), 
                2
            ) as pct_resolved_7d
        FROM resolution_times
        GROUP BY complaint_category
        HAVING COUNT(*) >= 100  -- Pelo menos 100 casos
        ORDER BY pct_resolved_24h DESC
        """
        
        result = pd.read_sql_query(query, self.conn)
        print("✓ Percentuais de resolução por categoria calculados")
        return result
    
    def query_borough_performance(self):
        """
        Performance por borough com ranking (versão SQLite)
        """
        query = """
        WITH borough_stats AS (
            SELECT 
                borough,
                COUNT(*) as total_complaints,
                COUNT(CASE WHEN status = 'Closed' THEN 1 END) as closed_complaints,
                ROUND(
                    COUNT(CASE WHEN status = 'Closed' THEN 1 END) * 100.0 / COUNT(*), 
                    2
                ) as closure_rate_pct,
                ROUND(AVG(CASE WHEN status = 'Closed' THEN response_time_days END), 2) as avg_response_days
            FROM nyc_311
            GROUP BY borough
        )
        SELECT 
            *,
            ROW_NUMBER() OVER (ORDER BY avg_response_days) as performance_rank
        FROM borough_stats
        ORDER BY performance_rank
        """
        
        result = pd.read_sql_query(query, self.conn)
        print("✓ Performance por borough com ranking")
        return result
    
    def query_temporal_analysis_with_windows(self):
        """
        Análise temporal simplificada para SQLite
        """
        query = """
        WITH monthly_stats AS (
            SELECT 
                strftime('%Y', created_date) as year,
                strftime('%m', created_date) as month,
                COUNT(*) as monthly_complaints,
                AVG(CASE WHEN status = 'Closed' THEN response_time_days END) as avg_response_days
            FROM nyc_311
            GROUP BY strftime('%Y', created_date), strftime('%m', created_date)
        )
        SELECT 
            year,
            month,
            monthly_complaints,
            ROUND(avg_response_days, 2) as avg_response_days,
            LAG(monthly_complaints, 1) OVER (ORDER BY year, month) as prev_month_complaints,
            ROUND(
                (monthly_complaints - LAG(monthly_complaints, 1) OVER (ORDER BY year, month)) * 100.0 / 
                LAG(monthly_complaints, 1) OVER (ORDER BY year, month), 
                2
            ) as complaints_growth_pct
        FROM monthly_stats
        ORDER BY year, month
        """
        
        result = pd.read_sql_query(query, self.conn)
        print("✓ Análise temporal com window functions concluída")
        return result
    
    def query_priority_analysis(self):
        """
        Análise de chamados prioritários (SQLite compatível)
        """
        query = """
        SELECT 
            is_priority,
            is_business_hours,
            COUNT(*) as total_complaints,
            ROUND(AVG(CASE WHEN status = 'Closed' THEN response_time_days END), 2) as avg_response_days,
            -- Aproximação de mediana para SQLite
            ROUND(AVG(CASE WHEN status = 'Closed' THEN response_time_days END), 2) as median_response_days,
            -- Aproximação de percentil 90
            MAX(CASE WHEN status = 'Closed' THEN response_time_days END) as max_response_days
        FROM nyc_311
        WHERE status = 'Closed' AND response_time_days IS NOT NULL
        GROUP BY is_priority, is_business_hours
        ORDER BY is_priority DESC, is_business_hours DESC
        """
        
        result = pd.read_sql_query(query, self.conn)
        print("✓ Análise de prioridade e horário comercial")
        return result
    
    def run_all_advanced_queries(self):
        """
        Executa todas as consultas avançadas
        """
        print("\n" + "="*50)
        print("  EXECUTANDO CONSULTAS SQL AVANÇADAS")
        print("="*50)
        
        results = {}
        
        try:
            # 1. Top tipos mais demorados
            print("\n1. Tipos de reclamação mais demorados:")
            results['slowest_types'] = self.query_top_slowest_complaint_types()
            
            # 2. Média móvel
            print("\n2. Média móvel do tempo de resposta:")
            results['moving_average'] = self.query_response_time_moving_average()
            
            # 3. Percentuais de resolução
            print("\n3. Percentuais de resolução:")
            results['resolution_percentages'] = self.query_resolution_percentages()
            
            # 4. Performance por borough
            print("\n4. Performance por borough:")
            results['borough_performance'] = self.query_borough_performance()
            
            # 5. Análise temporal
            print("\n5. Análise temporal com window functions:")
            results['temporal_analysis'] = self.query_temporal_analysis_with_windows()
            
            # 6. Análise de prioridade
            print("\n6. Análise de prioridade:")
            results['priority_analysis'] = self.query_priority_analysis()
            
            print("\n✅ Todas as consultas SQL executadas com sucesso!")
            
        except Exception as e:
            print(f"❌ Erro ao executar consultas: {e}")
            return None
        
        return results
    
    def save_results_to_csv(self, results, prefix='sql_results'):
        """
        Salva resultados das consultas em arquivos CSV
        """
        if not results:
            return
        
        print(f"\nSalvando resultados SQL...")
        
        for query_name, df in results.items():
            if isinstance(df, pd.DataFrame):
                filename = f"{prefix}_{query_name}.csv"
                filepath = os.path.join(self.data_dir, filename)
                df.to_csv(filepath, index=False)
                print(f"  ✓ {filename}")
    
    def close_connection(self):
        """
        Fecha conexão com o banco
        """
        if self.conn:
            self.conn.close()
            print("✓ Conexão com banco fechada")

def main():
    """
    Função principal para executar análises SQL
    """
    print("=" * 50)
    print("  NYC 311 Advanced SQL Analysis with SQLite")
    print("=" * 50)
    
    analyzer = SQLAnalyzer()
    
    try:
        # Conectar ao banco
        analyzer.connect_database()
        
        # Carregar dados para o banco
        analyzer.load_data_to_db()
        
        # Executar consultas avançadas
        results = analyzer.run_all_advanced_queries()
        
        # Salvar resultados
        if results:
            analyzer.save_results_to_csv(results)
        
    except Exception as e:
        print(f"❌ Erro durante análise SQL: {e}")
        return False
    
    finally:
        analyzer.close_connection()
    
    return True

if __name__ == "__main__":
    main()