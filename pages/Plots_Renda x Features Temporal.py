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
st.write('# Plots de Renda x Features ao Longo do Tempo')

# Cache plotting function for efficiency
@st.cache_data
def plot_graphs():
    fig, ax = plt.subplots(8, 1, figsize=(10, 70))
    renda[['posse_de_imovel', 'renda']].plot(kind='hist', ax=ax[0])

    sns.lineplot(x='data_ref', y='renda', hue='posse_de_imovel', data=renda, ax=ax[1])
    ax[1].tick_params(axis='x', rotation=45)
    sns.lineplot(x='data_ref', y='renda', hue='posse_de_veiculo', data=renda, ax=ax[2])
    ax[2].tick_params(axis='x', rotation=45)
    sns.lineplot(x='data_ref', y='renda', hue='qtd_filhos', data=renda, ax=ax[3])
    ax[3].tick_params(axis='x', rotation=45)
    sns.lineplot(x='data_ref', y='renda', hue='tipo_renda', data=renda, ax=ax[4])
    ax[4].tick_params(axis='x', rotation=45)
    sns.lineplot(x='data_ref', y='renda', hue='educacao', data=renda, ax=ax[5])
    ax[5].tick_params(axis='x', rotation=45)
    sns.lineplot(x='data_ref', y='renda', hue='estado_civil', data=renda, ax=ax[6])
    ax[6].tick_params(axis='x', rotation=45)
    sns.lineplot(x='data_ref', y='renda', hue='tipo_residencia', data=renda, ax=ax[7])
    ax[7].tick_params(axis='x', rotation=45)
    sns.despine()   
    
    return fig

# Plot graphs after caching the data and the plot functions
fig = plot_graphs()

st.pyplot(fig)
