#Librarias
import pandas as pd
import matplotlib.pyplot as plt
import mne
import tkinter as tk
from tkinter import filedialog,Tk, messagebox
import os
import numpy as np
from contextlib import redirect_stdout
from scipy.signal import firwin, lfilter, iirnotch, sosfilt, welch, filtfilt
import pywt
import logging

def cargar_archivo_eeg():
    """
    Abre un cuadro de diálogo para seleccionar un archivo .cnt,
    carga el archivo en un objeto MNE Raw, convierte los datos en un DataFrame,
    guarda la metadata en una variable y la retorna junto con el DataFrame.
    Imprime un aviso de éxito o error en lugar de información detallada.
    """
    # Desactivar los mensajes de registro de MNE
    logging.getLogger('mne').setLevel(logging.ERROR)
    root = Tk()
    root.withdraw()  
    root.attributes('-topmost', True)  #
    archivo_eeg = filedialog.askopenfilename(
        title="Selecciona un archivo CNT",
        filetypes=[("Archivos CNT", "*.cnt")]
    )
    root.destroy()
    
    if archivo_eeg:
        try:
            eeg_data = mne.io.read_raw_cnt(archivo_eeg, preload=True)
            data, _ = eeg_data[:, :]
            df = pd.DataFrame(data.T, columns=eeg_data.info['ch_names'])
            info = eeg_data.info  
            print("Archivo cargado correctamente.")
            return df, info
        
        except Exception as e:
            print(f"Error al cargar el archivo: {str(e)}")
            return None, None
    else:
        print("No se seleccionó ningún archivo.")
        return None, None


def imprimir_metadata_eeg(info=None):
    """
    Imprime la metadata más importante del archivo EEG de forma organizada,
    incluyendo la lista de nombres de canales en una línea horizontal.
    """
    if info is None:
        print("No se ha proporcionado información del archivo EEG.")
        return

    print("=== Información IMPORTANTE del Archivo EEG === \n")
    print(f"Frecuencia de muestreo:            {info['sfreq']} Hz")
    print(f"Total de canales:                         {info['nchan']}")
    print(f"Número de electrodos:               {len(info['dig'])}")
    canales = info['ch_names']
    canales_str = ', '.join(canales)
    print(f"Lista de Canales:                        {canales_str}")
    print(f"Bads (canales malos):                 {', '.join(info['bads']) if info['bads'] else 'Ninguno'}")

    
    
def graficar_eeg_sujeto(df_sujeto, frecuencia_muestreo=1000, canales=4, duracion=10, offset=3000, canales_especificos=None):
    """
    Grafica los canales del DataFrame de EEG en un periodo de tiempo definido,
    desplazando cada señal con un offset para mejorar la visualización.
    
    :param df_sujeto: DataFrame que contiene los datos de EEG.
    :param frecuencia_muestreo: Frecuencia de muestreo en Hz (default 1000).
    :param canales: Número de canales a graficar (default 4). Ignorado si se especifican canales_especificos.
    :param duracion: Duración de la señal a graficar en segundos.
    :param offset: Desplazamiento vertical entre señales (default 3000 microvoltios).
    :param canales_especificos: Lista de nombres de canales específicos a graficar (default None).
    """
    
    if canales_especificos:
        columnas = df_sujeto[canales_especificos]
    else:
        columnas = df_sujeto.iloc[:, :canales]

    if duracion is not None:
        muestras_duracion = int(duracion * frecuencia_muestreo)
        columnas = columnas.iloc[:muestras_duracion, :]
    else:
        muestras_duracion = len(columnas)
    
    tiempo = df_sujeto.index[:muestras_duracion] / frecuencia_muestreo

    plt.figure(figsize=(20, 8))
    
    for i, col in enumerate(columnas.columns):
        plt.plot(tiempo, columnas[col] / 0.0000001 + i * offset, label=col)
        
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud (Microvoltios)')
    plt.title(f'Canales del EEG con Offset (Duración: {duracion if duracion else "Completa"} s)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    