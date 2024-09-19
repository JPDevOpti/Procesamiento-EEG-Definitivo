import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

def calcular_conectividad_y_graficar(df, metodo='funcional',canales_seleccionados=None):
    """
    Calcula la conectividad entre los canales seleccionados o todos los canales si no se especifican,
    y visualiza los resultados en un mapa de calor.

    Parameters:
    - df: DataFrame con señales EEG, donde cada columna es un canal.
    - canales_seleccionados: Lista con los nombres de los canales que se desean analizar. Si es None, se usan todos los canales.
    - metodo: Método de conectividad a calcular ('funcional', 'efectiva', 'estructural').
    """
    if canales_seleccionados is None:
        canales_seleccionados = df.columns
    
    df_seleccionado = df[canales_seleccionados]
    
    if metodo == 'funcional':
        conectividades = df_seleccionado.corr()
        titulo = 'Mapa de Calor de la Conectividad (Funcional)'
    elif metodo == 'efectiva':
        n_canales = df_seleccionado.shape[1]
        conectividades = pd.DataFrame(np.zeros((n_canales, n_canales)), index=canales_seleccionados, columns=canales_seleccionados)

        for i in range(n_canales):
            for j in range(n_canales):
                if i != j:
                    test_result = grangercausalitytests(df_seleccionado[[canales_seleccionados[i], canales_seleccionados[j]]], maxlag=1)
                    conectividades.loc[canales_seleccionados[i], canales_seleccionados[j]] = test_result[1][0][0]['ssr_ftest'][0]

        titulo = 'Mapa de Calor de la Conectividad (Efectiva)'
        
    elif metodo == 'estructural':
        conectividades = df_seleccionado.corr()  
        titulo = 'Mapa de Calor de la Conectividad (Estructural)'
        
    else:
        raise ValueError("Método no soportado. Usa 'funcional', 'efectiva' o 'estructural'.")

    plt.figure(figsize=(10, 8))
    sns.heatmap(conectividades, annot=True, cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title(titulo)
    plt.show()


def calcular_conectividad_y_graficar_sin_numeros(df, metodo='funcional', canales_seleccionados=None):
    """
    Calcula la conectividad entre los canales seleccionados o todos los canales si no se especifican,
    y visualiza los resultados en un mapa de calor sin números en cada cuadro.

    Parameters:
    - df: DataFrame con señales EEG, donde cada columna es un canal.
    - canales_seleccionados: Lista con los nombres de los canales que se desean analizar. Si es None, se usan todos los canales.
    - metodo: Método de conectividad a calcular ('funcional', 'efectiva', 'estructural').
    """
    if canales_seleccionados is None:
        canales_seleccionados = df.columns
    
    df_seleccionado = df[canales_seleccionados]
    
    if metodo == 'funcional':
        conectividades = df_seleccionado.corr()
        titulo = 'Mapa de Calor de la Conectividad (Funcional)'
    elif metodo == 'efectiva':
        n_canales = df_seleccionado.shape[1]
        conectividades = pd.DataFrame(np.zeros((n_canales, n_canales)), index=canales_seleccionados, columns=canales_seleccionados)

        for i in range(n_canales):
            for j in range(n_canales):
                if i != j:
                    test_result = grangercausalitytests(df_seleccionado[[canales_seleccionados[i], canales_seleccionados[j]]], maxlag=1)
                    conectividades.loc[canales_seleccionados[i], canales_seleccionados[j]] = test_result[1][0][0]['ssr_ftest'][0]

        titulo = 'Mapa de Calor de la Conectividad (Efectiva)'
    elif metodo == 'estructural':
        conectividades = df_seleccionado.corr()  # Placeholder para conectividad estructural
        titulo = 'Mapa de Calor de la Conectividad (Estructural)'
    else:
        raise ValueError("Método no soportado. Usa 'funcional', 'efectiva' o 'estructural'.")

    plt.figure(figsize=(10, 8))
    sns.heatmap(conectividades, annot=False, cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title(titulo)
    plt.show()

