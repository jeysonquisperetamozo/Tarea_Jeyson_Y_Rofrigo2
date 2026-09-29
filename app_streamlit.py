import streamlit as st
import pandas as pd
from joblib import load

# -------------------------PROCESO DE DESPLIEGUE------------------------------
# En consola:
# pip install streamlit pandas scikit-learn joblib

# 01 --------------------------Cargar modelo y datos auxiliares-------------------------------------------
pipeline = load('Modelopipeline_zomato.joblib')
ciudades_disponibles = load('ciudades_disponibles.joblib')

# 02---------------- Variables globales para los campos del formulario-----------------------
price_range_options = [1, 2, 3, 4]
si_no_options = ["Yes", "No"]

price_range = 1
average_cost = 0.0
has_table_booking = "No"
has_online_delivery = "No"
votes = 0
city = ciudades_disponibles[0]

# 03 Reseteo------------- Flag para trackear error---------------------------------------
error_flag = False

def reset_inputs():
    global price_range, average_cost, has_table_booking, has_online_delivery, votes, city, error_flag
    price_range = 1
    average_cost = 0.0
    has_table_booking = "No"
    has_online_delivery = "No"
    votes = 0
    city = ciudades_disponibles[0]
    error_flag = False

reset_inputs()
# -----------------------------------------------------------------------------------------------

# ------------------------Título centrado-------------------------------------------------
st.title("Modelo Predictivo de Restaurantes (Zomato) con Random Forest Regressor")
st.markdown("Este modelo predice la calificación que tendría un restaurante en base a diferentes características.")
st.markdown("---")
st.markdown("Realizado por:")
st.markdown("*Jeyson Quispe Retamozo")
st.markdown("*Rodrigo Zuñiga Suma")


# ----------------------- Función para validar los campos del formulario----------------------------
def validate_inputs():
    global error_flag
    if any(val < 0 for val in [average_cost, votes]):
        st.error("No se permiten valores negativos. Por favor, ingrese valores válidos en todos los campos.")
        error_flag = True
    else:
        error_flag = False

# ------------------------------------ Formulario en dos columnas------------------------------------
with st.form("rating_form"):
    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("**Ciudad**", ciudades_disponibles)
        price_range = st.selectbox("**Rango de precio (1 a 4)**", price_range_options)
        average_cost = st.number_input("**Costo promedio para dos personas**", min_value=0.0, value=average_cost, step=10.0)

    with col2:
        has_table_booking = st.selectbox("**¿Tiene reserva de mesa?**", si_no_options)
        has_online_delivery = st.selectbox("**¿Tiene delivery online?**", si_no_options)
        votes = st.number_input("**Número de votos**", min_value=0, value=votes, step=1)

    predict_button = st.form_submit_button("Predecir")

# Validar antes de predecir
if predict_button:
    validate_inputs()

if predict_button and error_flag:
    st.stop()

if predict_button and not error_flag:
    # Crear DataFrame con el mismo orden/nombres de columnas usados al entrenar
    data = {
        "Price range": [price_range],
        "Average Cost for two": [average_cost],
        "Has Table booking": [has_table_booking],
        "Has Online delivery": [has_online_delivery],
        "Votes": [votes],
        "City": [city]
    }
    df = pd.DataFrame(data)

    # Realizar predicción (regresión -> un número, no una clase)
    rating_predicho = pipeline.predict(df)[0]
    rating_predicho = max(0, min(5, rating_predicho))  # acotar entre 0 y 5

    # Color según qué tan bueno sea el rating
    if rating_predicho >= 4:
        style_result = 'background-color: lightgreen; font-size: larger;'
    elif rating_predicho >= 2.5:
        style_result = 'background-color: khaki; font-size: larger;'
    else:
        style_result = 'background-color: lightcoral; font-size: larger;'

    result_html = f"<div style='{style_result}'>Rating predicho: <b>{rating_predicho:.2f}</b> / 5</div>"
    st.markdown(result_html, unsafe_allow_html=True)

# --------------------------- Botón de Resetear-------------------------------------
if st.button("Resetear"):
    reset_inputs()

# streamlit run app_streamlit.py       en la consola

# pip freeze > requirements.txt