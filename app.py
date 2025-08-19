import streamlit as st
import pandas as pd
import joblib

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Predictor de Velocidad MLB", page_icon="⚾")
st.title("⚾ Predictor de Velocidad del Primer Lanzamiento")
st.write("Esta aplicación utiliza un modelo de XGBoost para predecir la velocidad del primer lanzamiento de un pitcher en la primera entrada, asumiendo que será una recta de 4 costuras (FF).")

# --- CARGA DE DATOS Y MODELO (CON CACHÉ PARA RAPIDEZ) ---
@st.cache_data
def cargar_recursos():
    """Carga el modelo, los mapeos y los datos históricos una sola vez."""
    modelo = joblib.load('modelo_final_simplificado.pkl')
    mapeos = joblib.load('mapeos_simplificado.pkl')
    df_historico = pd.read_csv('datos_historicos.csv')
    #df_historico = pd.read_parquet('datos_historicos.parquet')
    return modelo, mapeos, df_historico

modelo, mapeos, df_historico = cargar_recursos()
features = list(mapeos.keys()) # Obtenemos la lista de features desde los mapeos

# --- FUNCIONES AUXILIARES ---
def buscar_pitcher_localmente(nombre_completo, df):
    info_jugador = df[df['player_name'] == nombre_completo]
    if info_jugador.empty:
        return None, f"No se encontró a '{nombre_completo}' en los datos históricos."
    data = info_jugador.iloc[0]
    return {"id": str(int(data['pitcher'])), "lanza": data['p_throws']}, None

# --- INTERFAZ DE USUARIO ---
st.header("Realizar una Nueva Predicción")

# Usamos columnas para organizar la entrada
col1, col2 = st.columns(2)
with col1:
    apellido = st.text_input("Apellido del Pitcher:", "Cole")
with col2:
    nombre = st.text_input("Nombre del Pitcher:", "Gerrit")

nombre_completo_pitcher = f"{apellido}, {nombre}"

if st.button("Predecir Velocidad", type="primary"):
    if not apellido or not nombre:
        st.error("Por favor, introduce el nombre y apellido del pitcher.")
    else:
        with st.spinner('Buscando pitcher y realizando predicción...'):
            # --- LÓGICA DE PREDICCIÓN ---
            pitcher_info, error = buscar_pitcher_localmente(nombre_completo_pitcher, df_historico)

            if error:
                st.error(error)
            else:
                # Datos para la predicción
                datos_partido = {
                    'pitcher': pitcher_info['id'],
                    'p_throws': pitcher_info['lanza'],
                    'inning': '1', 'at_bat_number': '1',
                    'pitch_number': '1', 'pitch_type': 'FF'
                }
                
                # Mapeo a números
                datos_numericos = {}
                error_mapeo = False
                for feature, value in datos_partido.items():
                    if value not in mapeos[feature]:
                        st.error(f"Error: El valor '{value}' para '{feature}' es desconocido.")
                        error_mapeo = True
                        break
                    datos_numericos[feature] = mapeos[feature][value]

                if not error_mapeo:
                    # Predicción
                    fila_a_predecir = pd.DataFrame([datos_numericos], columns=features)
                    velocidad_predicha = modelo.predict(fila_a_predecir)[0]

                    # Búsqueda de historial
                    historial = df_historico[df_historico['pitcher'] == float(pitcher_info['id'])]
                    primeros_lanzamientos = historial[
                        (historial['inning'] == 1) & (historial['at_bat_number'] == 1) & (historial['pitch_number'] == 1)
                    ].head(5)
                    historial_display = primeros_lanzamientos[['game_date', 'release_speed', 'pitch_type']].rename(
                        columns={'game_date': 'Fecha', 'release_speed': 'Velocidad (mph)', 'pitch_type': 'Tipo'}
                    )
                    historial_display['Fecha'] = pd.to_datetime(historial_display['Fecha']).dt.strftime('%Y-%m-%d')

                    # --- MOSTRAR RESULTADOS ---
                    st.success("¡Predicción completada!")
                    st.metric(label=f"Velocidad Predicha para {nombre_completo_pitcher}", value=f"{velocidad_predicha:.2f} mph")
                    
                    st.subheader("Referencia Histórica")
                    st.write("Últimos 5 primeros lanzamientos del partido registrados:")
                    st.dataframe(historial_display, use_container_width=True)