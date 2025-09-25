"""
Módulo para modelo preditivo moderno do tempo de resposta
Utiliza técnicas avançadas de ML e feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class PredictiveModel:
    def __init__(self):
        self.data_dir = 'data'
        self.models_dir = 'models'
        self.model = None
        self.label_encoders = {}
        self.scaler = None
        self.feature_columns = None
        
        # Criar diretório de modelos
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def load_data(self, filename='nyc_311_2023_clean.csv'):
        """
        Carrega dados para modelagem
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        
        print(f"Carregando dados de: {filepath}")
        df = pd.read_csv(filepath)
        
        # Converter datas
        df['created_date'] = pd.to_datetime(df['created_date'])
        df['closed_date'] = pd.to_datetime(df['closed_date'])
        
        # Filtrar apenas registros fechados com tempo de resposta válido
        model_df = df[
            (df['status'] == 'Closed') & 
            (df['response_time_days'].notna()) &
            (df['response_time_days'] > 0) &
            (df['response_time_days'] <= 90)  # Remover outliers extremos
        ].copy()
        
        print(f"✓ Dados carregados para modelagem: {len(model_df):,} registros")
        return model_df
    
    def prepare_features(self, df):
        """
        Prepara features avançadas para o modelo
        """
        print("Preparando features avançadas...")
        
        # Features base
        feature_columns = [
            'complaint_category', 'borough', 'created_hour', 'created_weekday',
            'created_month', 'quarter', 'is_weekend', 'is_business_hours', 
            'is_priority', 'created_period'
        ]
        
        # Verificar se todas as colunas existem
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            # Criar colunas faltantes com valores padrão
            for col in missing_cols:
                if col == 'quarter':
                    df[col] = df['created_month'].apply(lambda x: (x-1)//3 + 1)
                elif col == 'is_weekend':
                    df[col] = df['created_weekday'].isin(['Saturday', 'Sunday'])
                elif col == 'is_business_hours':
                    df[col] = (df['created_hour'].between(9, 17)) & (~df.get('is_weekend', False))
                elif col == 'is_priority':
                    priority_types = ['Heat/Hot Water', 'Water System', 'UNSANITARY CONDITION']
                    df[col] = df['complaint_type'].isin(priority_types)
                elif col == 'created_period':
                    df[col] = df['created_hour'].apply(self._get_period)
                else:
                    df[col] = 'Unknown'
        
        # Criar DataFrame de features
        X = df[feature_columns].copy()
        y = df['response_time_days'].copy()
        
        # Features adicionais (feature engineering)
        # 1. Interação borough x categoria
        X['borough_category'] = X['borough'].astype(str) + '_' + X['complaint_category'].astype(str)
        
        # 2. Indicador de hora de pico
        X['is_peak_hour'] = X['created_hour'].isin([8, 9, 10, 17, 18, 19])
        
        # 3. Indicador de final de ano
        X['is_year_end'] = X['created_month'].isin([11, 12])
        
        # Encoding de variáveis categóricas
        categorical_columns = [
            'complaint_category', 'borough', 'created_weekday', 
            'created_period', 'borough_category'
        ]
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                # Para novos valores não vistos no treino
                try:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
                except ValueError:
                    # Se houver valores não vistos, usar o primeiro valor conhecido
                    unknown_mask = ~X[col].astype(str).isin(self.label_encoders[col].classes_)
                    X.loc[unknown_mask, col] = self.label_encoders[col].classes_[0]
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Converter booleanos para int
        boolean_columns = ['is_weekend', 'is_business_hours', 'is_priority', 'is_peak_hour', 'is_year_end']
        for col in boolean_columns:
            X[col] = X[col].astype(int)
        
        # Atualizar lista de features
        self.feature_columns = list(X.columns)
        
        print(f"Features preparadas: {X.shape[1]} features, {len(X):,} amostras")
        return X, y
    
    def _get_period(self, hour):
        """Converte hora em período do dia"""
        if 6 <= hour < 12:
            return 'Manhã'
        elif 12 <= hour < 18:
            return 'Tarde'
        elif 18 <= hour < 24:
            return 'Noite'
        else:
            return 'Madrugada'
    
    def train_random_forest(self, X, y, test_size=0.2, random_state=42):
        """
        Treina modelo Random Forest
        """
        print("Treinando modelo Random Forest...")
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Treinar modelo
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Fazer predições
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calcular métricas
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        test_metrics = self._calculate_metrics(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        
        results = {
            'model_type': 'Random Forest',
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_mae': cv_mae,
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_)),
            'data_split': {'train_size': len(X_train), 'test_size': len(X_test)}
        }
        
        print("✓ Modelo Random Forest treinado com sucesso!")
        return results
    
    def train_linear_regression(self, X, y, test_size=0.2, random_state=42):
        """
        Treina modelo de regressão linear (baseline)
        """
        print("Treinando modelo Linear Regression (baseline)...")
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Padronizar features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Treinar modelo
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        
        # Fazer predições
        y_train_pred = lr_model.predict(X_train_scaled)
        y_test_pred = lr_model.predict(X_test_scaled)
        
        # Calcular métricas
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        test_metrics = self._calculate_metrics(y_test, y_test_pred)
        
        results = {
            'model_type': 'Linear Regression',
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'model': lr_model,
            'data_split': {'train_size': len(X_train), 'test_size': len(X_test)}
        }
        
        print("✓ Modelo Linear Regression treinado com sucesso!")
        return results
    
    def _calculate_metrics(self, y_true, y_pred):
        """
        Calcula métricas de avaliação
        """
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
    
    def save_model(self, filename='nyc_311_response_time_model.pkl'):
        """
        Salva o modelo treinado
        """
        if self.model is None:
            raise ValueError("Nenhum modelo treinado para salvar")
        
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        filepath = os.path.join(self.models_dir, filename)
        joblib.dump(model_data, filepath)
        
        print(f"✓ Modelo salvo em: {filepath}")
        return filepath
    
    def load_model(self, filename='nyc_311_response_time_model.pkl'):
        """
        Carrega modelo salvo
        """
        filepath = os.path.join(self.models_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Modelo não encontrado: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.scaler = model_data.get('scaler')
        self.feature_columns = model_data['feature_columns']
        
        print(f"✓ Modelo carregado de: {filepath}")
        return True
    
    def predict_response_time(self, complaint_type='Noise - Residential', borough='MANHATTAN', 
                            created_hour=14, is_weekend=False, weekday='Monday'):
        """
        Prediz tempo de resposta para um chamado específico
        """
        if self.model is None:
            raise ValueError("Nenhum modelo carregado para predição")
        
        # Mapear para categorias conhecidas
        category_mapping = {
            'Noise - Residential': 'Noise',
            'Heat/Hot Water': 'Housing',
            'Street Condition': 'Streets',
            'UNSANITARY CONDITION': 'Sanitation',
            'Blocked Driveway': 'Parking',
            'Water System': 'Water',
            'Illegal Parking': 'Parking'
        }
        
        complaint_category = category_mapping.get(complaint_type, 'Other')
        
        # Determinar outras features
        is_business_hours = 9 <= created_hour <= 17 and not is_weekend
        
        created_period = self._get_period(created_hour)
        
        # Determinar se é prioritário
        priority_types = ['Heat/Hot Water', 'Water System', 'UNSANITARY CONDITION']
        is_priority = complaint_type in priority_types
        
        # Criar sample para predição com todas as features necessárias
        sample_data = {
            'complaint_category': complaint_category,
            'borough': borough,
            'created_hour': created_hour,
            'created_weekday': weekday,
            'created_month': 6,  # Junho (médio do ano)
            'quarter': 2,  # Q2
            'is_weekend': is_weekend,
            'is_business_hours': is_business_hours,
            'is_priority': is_priority,
            'created_period': created_period,
            'borough_category': f"{borough}_{complaint_category}",
            'is_peak_hour': created_hour in [8, 9, 10, 17, 18, 19],
            'is_year_end': False
        }
        
        # Converter para DataFrame
        sample_df = pd.DataFrame([sample_data])
        
        # Aplicar encodings para variáveis categóricas
        categorical_columns = ['complaint_category', 'borough', 'created_weekday', 
                              'created_period', 'borough_category']
        
        for col in categorical_columns:
            if col in self.label_encoders:
                try:
                    sample_df[col] = self.label_encoders[col].transform(sample_df[col].astype(str))
                except ValueError:
                    # Se valor não existe no encoder, usar o primeiro valor conhecido
                    sample_df[col] = 0
            else:
                sample_df[col] = 0
        
        # Converter booleanos para int
        boolean_columns = ['is_weekend', 'is_business_hours', 'is_priority', 
                          'is_peak_hour', 'is_year_end']
        for col in boolean_columns:
            sample_df[col] = sample_df[col].astype(int)
        
        # Garantir que todas as colunas do modelo estejam presentes
        for col in self.feature_columns:
            if col not in sample_df.columns:
                sample_df[col] = 0
        
        # Reordenar colunas para corresponder ao treinamento
        sample_df = sample_df[self.feature_columns]
        
        # Fazer predição
        try:
            prediction = self.model.predict(sample_df)[0]
            return max(0.1, prediction)  # Mínimo de 0.1 dias
        except Exception as e:
            print(f"Erro na predição: {e}")
            return 3.0  # Valor padrão
    
    def generate_model_report(self, results):
        """
        Gera relatório do modelo
        """
        print("\n" + "="*60)
        print("  RELATÓRIO DO MODELO PREDITIVO")
        print("="*60)
        
        print(f"\n📊 Modelo: {results['model_type']}")
        print(f"   Dados de treino: {results['data_split']['train_size']:,}")
        print(f"   Dados de teste: {results['data_split']['test_size']:,}")
        
        print(f"\n🎯 Métricas de Treino:")
        train = results['train_metrics']
        print(f"   MAE:  {train['mae']:.2f} dias")
        print(f"   RMSE: {train['rmse']:.2f} dias")
        print(f"   R²:   {train['r2']:.3f}")
        
        print(f"\n🎯 Métricas de Teste:")
        test = results['test_metrics']
        print(f"   MAE:  {test['mae']:.2f} dias")
        print(f"   RMSE: {test['rmse']:.2f} dias")
        print(f"   R²:   {test['r2']:.3f}")
        
        if 'cv_mae' in results:
            print(f"\n🔄 Cross-Validation MAE: {results['cv_mae']:.2f} dias")
        
        if 'feature_importance' in results:
            print(f"\n📈 Top 5 Features Mais Importantes:")
            importance = results['feature_importance']
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for feature, importance_score in top_features:
                print(f"   {feature}: {importance_score:.3f}")

def main():
    """
    Função principal para treinar e avaliar modelo
    """
    print("=" * 50)
    print("  NYC 311 Predictive Model")
    print("=" * 50)
    
    predictor = PredictiveModel()
    
    try:
        # Carregar dados
        df = predictor.load_data()
        
        # Preparar features
        X, y = predictor.prepare_features(df)
        
        # Treinar modelo Random Forest
        rf_results = predictor.train_random_forest(X, y)
        
        # Salvar modelo
        predictor.save_model()
        
        # Gerar relatório
        predictor.generate_model_report(rf_results)
        
        # Exemplo de predição
        print(f"\n🔮 Exemplo de Predição:")
        pred_time = predictor.predict_response_time(
            complaint_type='Noise - Residential',
            borough='MANHATTAN',
            created_hour=22,  # 22h
            is_weekend=True
        )
        print(f"   Ruído residencial em Manhattan, 22h, final de semana")
        print(f"   Tempo previsto: {pred_time:.1f} dias")
        
        return rf_results
        
    except Exception as e:
        print(f"❌ Erro durante modelagem: {e}")
        return None

if __name__ == "__main__":
    main()