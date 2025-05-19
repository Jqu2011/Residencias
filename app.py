
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import os

# Título
st.title("Consulta y Predicción de Residencias de Adulto Mayor")
st.markdown("Desarrollado por Julia | Proyecto ConVive+")

# Cargar datos
@st.cache_data
def cargar_datos():
    df = pd.read_excel("residencias_unificadas_v3.xlsx")
    return df

data = cargar_datos()

# Filtros
st.sidebar.header("Filtros")
provincia = st.sidebar.selectbox("Selecciona una provincia", data["provincia"].unique())
tipo_titularidad = st.sidebar.multiselect("Selecciona tipo de titularidad", data["Titularidad"].unique(), default=data["Titularidad"].unique())

# Aplicar filtros
df_filtrado = data[(data["provincia"] == provincia) & (data["Titularidad"].isin(tipo_titularidad))]

# Mostrar resumen
st.subheader(f"Resumen de residencias en {provincia}")
st.write(f"Total de residencias: {df_filtrado.shape[0]}")

if "precio_mensual" in df_filtrado.columns:
    st.write(f"Precio promedio mensual: {round(df_filtrado['precio_mensual'].mean(), 2)} €")
else:
    st.warning("No se encontró la columna 'precio_mensual' para calcular el promedio.")

# Mostrar tabla con direcciones
st.dataframe(df_filtrado[["Denominación", "direccion_completa", "Titularidad", "Plazas"]])

# Visualización: número de camas
if "Plazas" in df_filtrado.columns:
    fig1 = px.histogram(df_filtrado, x="Plazas", nbins=20, title="Distribución de Número de Plazas")
    st.plotly_chart(fig1)

# Visualización: tipo de titularidad
fig2 = px.pie(df_filtrado, names="Titularidad", title="Distribución por Tipo de Titularidad")
st.plotly_chart(fig2)

# Cargar modelo entrenado
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_entrenado.pkl")

modelo = cargar_modelo()

st.subheader("Predicción del Precio de Residencia")

# Campos de entrada para predicción (Ejemplo básico: número de camas y tipo de titularidad codificada)
num_camas_input = st.number_input("Número de plazas (camas)", min_value=10, max_value=500, step=10)
titularidad_input = st.selectbox("Titularidad", sorted(data["Titularidad"].unique()))

# Codificación simple (debe coincidir con el modelo entrenado)
titularidad_dict = {name: i for i, name in enumerate(data["Titularidad"].unique())}
titularidad_cod = titularidad_dict.get(titularidad_input, 0)

# Predicción
input_modelo = np.array([[num_camas_input, titularidad_cod]])
if st.button("Predecir precio mensual"):
    pred = modelo.predict(input_modelo)
    st.success(f"Precio mensual estimado: {round(pred[0], 2)} €")
