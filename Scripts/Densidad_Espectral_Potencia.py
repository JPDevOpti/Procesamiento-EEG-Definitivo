#Librarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch


def compute_average_psd(df, fs=1000, nperseg=3072):
    """
    Calcula el espectro de densidad de potencia (PSD) promedio de todas las columnas de un DataFrame.
    
    Parameters:
    - df: pandas.DataFrame con los datos de señal.
    - fs: frecuencia de muestreo en Hz.
    - nperseg: longitud de cada segmento para el cálculo de Welch.
    
    Returns:
    - freqs: Array de frecuencias.
    - avg_psd: PSD promedio de todas las columnas.
    """
    
    psd_list = []
    freqs = None
    
    for col in df.columns:
        f, Pxx = welch(df[col], fs=fs, nperseg=nperseg)
        psd_list.append(Pxx)
        if freqs is None:
            freqs = f
    
    avg_psd = np.mean(psd_list, axis=0)
    
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, avg_psd, color='red')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Densidad espectral de potencia (PSD)')
    plt.yscale('log')
    plt.xlim([-2, 60])
    plt.ylim([1e-15, 1e-9])
    plt.title('PSD Promedio de Todas las Columnas')
    plt.grid(True)
    plt.show()
    
    return freqs, avg_psd