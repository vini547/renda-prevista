import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

sns.set_theme(context='paper', style='ticks')

st.set_page_config(
     page_title="Análise Exploratória",
     page_icon=":bar_chart:",
     layout="wide",
)

# Cache data loading with st.cache_data
@st.cache_data
def load_data():
    return pd.read_csv("./test/input/previsao_de_renda.csv")

renda = load_data()

# Move st.write() and st.pyplot() outside of the cached function
st.write('# Plots de Bivariadas')

# Cache plotting function for efficiency
@st.cache_data
def plot_graphs():
    
    fig_bivar, ax_bivar = plt.subplots(7, 1, figsize=(10, 50))
    sns.barplot(x='posse_de_imovel', y='renda', data=renda, ax=ax_bivar[0])
    sns.barplot(x='posse_de_veiculo', y='renda', data=renda, ax=ax_bivar[1])
    sns.barplot(x='qtd_filhos', y='renda', data=renda, ax=ax_bivar[2])
    sns.barplot(x='tipo_renda', y='renda', data=renda, ax=ax_bivar[3])
    sns.barplot(x='educacao', y='renda', data=renda, ax=ax_bivar[4])
    sns.barplot(x='estado_civil', y='renda', data=renda, ax=ax_bivar[5])
    sns.barplot(x='tipo_residencia', y='renda', data=renda, ax=ax_bivar[6])
    sns.despine()
    
    return fig_bivar

# Plot graphs after caching the data and the plot functions
fig_bivar = plot_graphs()


st.pyplot(fig_bivar)