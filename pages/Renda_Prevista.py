import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
import shap
import streamlit as st

st.set_page_config(layout="wide")
st.title("Previsão de Renda com LightGBM")

# Carregando os dados
df = pd.read_csv("./test/input/previsao_de_renda.csv")
df = df.drop_duplicates()

# Removendo colunas não úteis
X = df.drop(columns=['Unnamed: 0', 'id_cliente', 'data_ref', 'renda'])
y = df['renda']

# Transformação logarítmica na variável target
y_log = np.log1p(y)

# Separando tipos de colunas
numeric_features = ['idade', 'tempo_emprego', 'qt_pessoas_residencia', 'qtd_filhos']
categorical_features = ['sexo', 'posse_de_veiculo', 'posse_de_imovel', 'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia']

# Define pipelines de pré-processamento
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Pipeline com LightGBM
model = LGBMRegressor(n_estimators=2000, learning_rate=0.02, subsample=0.8, random_state=42)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Treinamento do modelo
pipeline.fit(X_train, y_train)

# Recupera nomes das features após transformação
preprocessor2 = pipeline.named_steps['preprocessor']
ohe = preprocessor2.named_transformers_['cat'].named_steps['onehot']
categorical_feature_names = ohe.get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(categorical_feature_names)

# Calculando SHAP values
explainer = shap.Explainer(pipeline.named_steps['model'])
X_transformed = preprocessor.transform(X_test)
shap_values = explainer(X_transformed)
shap_values.feature_names = all_feature_names

# Gráfico SHAP
st.subheader("Importância das Features (SHAP)")
fig_shap, ax = plt.subplots(figsize=(10, 8))
shap.plots.beeswarm(shap_values, max_display=15, show=False)
st.pyplot(fig_shap)

# Predições
y_pred_log = pipeline.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

# Métricas
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

st.markdown(f"**MAE (Erro Absoluto Médio):** R$ {mae:.2f}")
st.markdown(f"**RMSE (Raiz do Erro Quadrático Médio):** R$ {rmse:.2f}")

# Gráfico de resíduos
st.subheader("Distribuição dos Resíduos")
residuals = y_true - y_pred
fig_residuals, ax = plt.subplots(figsize=(10, 6))
sns.histplot(residuals, bins=50, kde=True, ax=ax)
ax.set_title("Distribuição dos Resíduos")
ax.set_xlabel("Erro de Previsão (R$)")
st.pyplot(fig_residuals)