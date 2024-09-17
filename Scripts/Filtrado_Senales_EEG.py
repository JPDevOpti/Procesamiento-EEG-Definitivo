#Librarias
import pandas as pd
import matplotlib.pyplot as plt
import mne
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import os
import numpy as np
from contextlib import redirect_stdout
from scipy.signal import firwin, lfilter, iirnotch, sosfilt, welch, filtfilt
from tkinter import Tk, filedialog
import pywt


def wnoisest(coeff):
    """Estima la desviación estándar del ruido para cada nivel de coeficiente de wavelet."""
    stdc = np.zeros(len(coeff))
    for i in range(1, len(coeff)):
        stdc[i] = np.median(np.absolute(coeff[i])) / 0.6745
    return stdc

def thselect(signal):
    """Selecciona el valor del umbral basado en el umbral universal."""
    num_samples = sum(s.shape[0] for s in signal)
    thr = np.sqrt(2 * np.log(num_samples))
    return thr

def wthresh(coeff, thr):
    """Función de umbral para los coeficientes de wavelet."""
    y = []
    for i in range(len(coeff)):
        y.append(np.where(np.abs(coeff[i]) > thr, coeff[i], 0))
    return y

def apply_wavelet_filter(df, wavelet='db1', level=1):
    """
    Aplica un filtro de wavelet a cada columna en un DataFrame utilizando umbralización.

    Parámetros:
    - df: DataFrame de pandas con datos de señal donde cada columna es una señal separada.
    - wavelet: Nombre del wavelet a usar (por defecto es 'db1').
    - level: Nivel de descomposición (por defecto es 1).

    Retorna:
    - filtered_df: DataFrame con las señales filtradas por wavelet.
    """
    filtered_data = {}
    
    for column in df.columns:
        
        signal = df[column].values
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        stdc = wnoisest(coeffs)
        thr = thselect(coeffs)
        coeffs_thresholded = wthresh(coeffs, thr)
        filtered_signal = pywt.waverec(coeffs_thresholded, wavelet)
        
        if len(filtered_signal) > len(signal):
            filtered_signal = filtered_signal[:len(signal)]
            
        elif len(filtered_signal) < len(signal):
            filtered_signal = np.pad(filtered_signal, (0, len(signal) - len(filtered_signal)), mode='constant')
        
        filtered_data[column] = filtered_signal
    
    filtered_df = pd.DataFrame(filtered_data, index=df.index)
    return filtered_df


def apply_filters(df, low_cutoff=1, high_cutoff=60, notch_freq=None, filter_order=200, fs=1000, window='hamming', quality_factor=50):
    """
    Aplica un filtro FIR pasa banda y opcionalmente un filtro notch a todas las columnas de un DataFrame.
    
    Parameters:
    - df: pandas.DataFrame con los datos a filtrar.
    - low_cutoff: frecuencia de corte inferior del filtro pasa banda en Hz.
    - high_cutoff: frecuencia de corte superior del filtro pasa banda en Hz.
    - notch_freq: frecuencia a eliminar con el filtro notch en Hz. Si es None, no se aplica filtro notch.
    - filter_order: orden del filtro FIR pasa banda (debe ser impar).
    - fs: frecuencia de muestreo en Hz.
    - window: tipo de ventana para el diseño del filtro FIR pasa banda (ej. 'hamming', 'blackman').
    - quality_factor: factor de calidad del filtro notch.
    
    Returns:
    - pandas.DataFrame con los datos filtrados.
    """
    nyquist = 0.5 * fs
    normal_low_cutoff = low_cutoff / nyquist
    normal_high_cutoff = high_cutoff / nyquist
    
    b_bandpass = firwin(filter_order, [normal_low_cutoff, normal_high_cutoff], pass_zero=False, window=window)
    
    df_filtered_bandpass = pd.DataFrame()
    
    for column in df.columns:
        df_filtered_bandpass[column] = filtfilt(b_bandpass, [1.0], df[column])
    
    if notch_freq is not None:
        notch_b, notch_a = iirnotch(notch_freq / nyquist, quality_factor)
        df_filtered = pd.DataFrame()
        for column in df_filtered_bandpass.columns:
            df_filtered[column] = filtfilt(notch_b, notch_a, df_filtered_bandpass[column])
    else:
        df_filtered = df_filtered_bandpass
    
    return df_filtered