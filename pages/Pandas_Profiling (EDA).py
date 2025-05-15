import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit.components.v1 import html

st.title("Análise Exploratória - YData Profiling")

# Função cacheada para gerar o relatório
@st.cache_data
def gerar_relatorio(dataframe):
    profile = ProfileReport(
        dataframe,
        explorative=True,
        correlations={"auto": {"calculate": False}}  # desativa autocorrelação
    )
    return profile.to_html()

# Carrega os dados
df = pd.read_csv("./test/input/previsao_de_renda.csv")
df['qt_pessoas_residencia'] = df['qt_pessoas_residencia'].astype(int)
df.drop(columns=['id_cliente','Unnamed: 0','data_ref'], inplace=True)

# Gera o HTML do relatório
with st.spinner("Gerando relatório... isso pode levar alguns segundos."):
    profile_html = gerar_relatorio(df)

# Exibe o relatório no Streamlit
html(profile_html, height=1000, scrolling=True)