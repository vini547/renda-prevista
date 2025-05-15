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
# Filter only numeric columns
numeric_df = df.select_dtypes(include=['float', 'int'])

# Create the pairplot
pairplot_fig = sns.pairplot(numeric_df)
pairplot_fig.fig.suptitle("Pair Plot das features numéricas", y=1.02, fontsize=16)

# Display the plot in Streamlit
st.pyplot(pairplot_fig)