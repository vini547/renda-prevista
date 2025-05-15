import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Carregar dados (substitua com seu caminho real)
df = pd.read_csv("./test/input/previsao_de_renda.csv")

# Página
st.set_page_config(
    page_title="Relatório de Predição de Renda",
    page_icon="📊",
    layout="wide"
)

# Estilo e introdução
st.markdown("""
<style>
.main {
    background-color: #f7f9fa;
    padding: 3rem 6rem;
    font-family: 'Segoe UI', sans-serif;
}
.title {
    font-size: 2.5rem;
    color: #0a0a0a;
    font-weight: 700;
    margin-bottom: 1rem;
}
.subtitle {
    font-size: 1.25rem;
    color: #444444;
    margin-bottom: 2rem;
}
.body-text {
    font-size: 1.1rem;
    line-height: 1.6;
    color: #333333;
}
ul {
    padding-left: 1.5rem;
    margin-bottom: 1.5rem;
}
</style>

<div class="main">
<div class="title">📈 Metodologia CRISP-DM para Predição de Renda</div>
<div class="subtitle">Análise preditiva com método científico </div>
<div class="body-text">
<p>Bem-vindo ao <strong>Relatório de Pesquisa em Predição de Renda</strong>.
Este estudo aplica técnicas modernas de aprendizado de máquina para prever a renda individual com base em atributos demográficos e socioeconômicos.
O objetivo é identificar os sinais preditivos que impulsionam a dinâmica da renda.</p>

<ul>
<li><strong>Modelo:</strong> Regressor LightGBM</li>
<li><strong>Transformação do alvo:</strong> Log1p</li>
<li><strong>Base de dados:</strong> dados anonimizados de solicitações financeiras</li>
</ul>

<p>
  Navegue pelas seções no menu lateral para explorar a análise exploratória, os resultados do modelo e os insights interpretativos.
  <br>
  Acesse o notebook no Kaggle para mais detalhes sobre a metodologia CRISP-DM através deste <a href="https://www.kaggle.com/code/viniciuscoimbra547/previs-o-de-renda-crisp-dm" target="_blank">link</a>.
</p>
</div>
</div>
""", unsafe_allow_html=True)

# ====================
# Gráfico 1: Distribuição da Renda (KDE)
# ====================
st.subheader("Distribuição da Renda")

df_filtrado = df[df['renda'] <= 25000]
fig1, ax1 = plt.subplots(figsize=(12, 4))
sns.histplot(df_filtrado['renda'], kde=True, ax=ax1, color='skyblue')
ax1.set_title('Distribuição das observações de renda com o kernel density estimate (KDE) plot.')
ax1.set_xlabel('Renda')
ax1.set_ylabel('Frequência')
ax1.set_xticks(np.arange(0, 25000, 2000))
st.pyplot(fig1)


