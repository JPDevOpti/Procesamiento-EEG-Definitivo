import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
import tkinter as tk
from tkinter import filedialog

def potencia_banda(frecuencias, psd, banda):
    idx_banda = np.logical_and(frecuencias >= banda[0], frecuencias <= banda[1])
    return np.mean(psd[idx_banda])

def calcular_potencias_banda(df, fs, bandas, canales_especificos=None):
    if canales_especificos:
        df = df[canales_especificos]
    
    lista_potencias_banda = []
    
    for columna in df.columns:
        datos_canal = df[columna].values
        frecuencias, psd = welch(datos_canal, fs, nperseg=4096)
        potencias_banda = {banda: potencia_banda(frecuencias, psd, bandas[banda]) for banda in bandas}
        lista_potencias_banda.append(potencias_banda)
    
    df_potencias_banda = pd.DataFrame(lista_potencias_banda, index=df.columns)
    
    return df_potencias_banda

def graficar_histograma_combinado(df_potencias_banda, num_canales=3, canales_seleccionados=None):
    if canales_seleccionados:
        df_potencias_banda = df_potencias_banda.loc[canales_seleccionados]

    num_canales = min(num_canales, df_potencias_banda.shape[0])
    canales_seleccionados = df_potencias_banda.index[:num_canales]
    nombres_bandas = df_potencias_banda.columns
    ancho_barra = 0.1
    indice = np.arange(len(nombres_bandas))
    plt.figure(figsize=(10, 6))
    colores = plt.cm.get_cmap('tab10', num_canales)
    
    for i, canal in enumerate(canales_seleccionados):
        valores_banda = df_potencias_banda.loc[canal]
        plt.bar(indice + i * ancho_barra, valores_banda, ancho_barra, label=canal, color=colores(i))
    
    plt.xlabel('Bandas de Frecuencia')
    plt.ylabel('Potencia Promedio (uV²/Hz)')
    plt.title('Histograma de Bandas de Frecuencia por Canal')
    plt.xticks(indice + ancho_barra * (num_canales - 1) / 2, nombres_bandas, rotation=45)
    plt.legend(title='Canales', loc='upper right')
    plt.tight_layout()
    plt.show()

def graficar_histograma_promedio(df_potencias_banda):
    valores_promedio_banda = df_potencias_banda.mean(axis=0)
    nombres_bandas = df_potencias_banda.columns
    
    plt.figure(figsize=(10, 6))
    plt.bar(nombres_bandas, valores_promedio_banda, color='red', alpha=0.7)
    plt.xlabel('Bandas de Frecuencia')
    plt.ylabel('Potencia Promedio (uV²/Hz)')
    plt.title('Promedio de Bandas de Frecuencia sobre Todos los Canales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def guardar_csv(df):
    """
    Abre una ventana para seleccionar la ubicación y nombre del archivo CSV,
    y la mantiene al frente de todas las ventanas para mayor visibilidad.

    Parameters:
    - df: DataFrame que se desea guardar.
    """
    root = tk.Tk()
    root.withdraw() 
    root.attributes('-topmost', True) 
    nombre_archivo = filedialog.asksaveasfilename(defaultextension=".csv",
                                                  filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    
    if nombre_archivo:
        df.to_csv(nombre_archivo, index=True)  
        print(f"Archivo guardado con éxito en: {nombre_archivo}")
    else:
        print("No se seleccionó ninguna ubicación.")
    
    root.destroy()