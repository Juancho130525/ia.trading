import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import ta
import altair as alt
import requests

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Predictor Crypto IDX", layout="wide")
st.title("IA para Crypto IDX (Conectado a Kraken)")
st.write("Esta IA analiza el mercado real de Bitcoin para predecir la dirección de la próxima vela.")

# --- INICIALIZAR CONTADORES (DIARIO DE TRADING) ---
if 'ganadas' not in st.session_state:
    st.session_state.ganadas = 0
if 'perdidas' not in st.session_state:
    st.session_state.perdidas = 0

# --- BARRA LATERAL ---
st.sidebar.header("⚙️ Configuración")
temporalidad = st.sidebar.selectbox(
    "Selecciona la temporalidad:",
    options=['1m', '5m', '15m', '1h'],
    index=2 
)

# --- CONFIGURACIÓN DE TELEGRAM ---
st.sidebar.divider()
st.sidebar.subheader("📱 Alertas de Telegram")
telegram_token = st.sidebar.text_input("Token del Bot", type="password")
telegram_chat_id = st.sidebar.text_input("Chat ID", type="password")

# Botón de prueba de Telegram
if st.sidebar.button("Probar mensaje de Telegram"):
    if telegram_token and telegram_chat_id:
        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        payload = {"chat_id": telegram_chat_id, "text": "✅ ¡Hola! Este es un mensaje de prueba de mi IA de Trading."}
        try:
            respuesta = requests.post(url, data=payload, timeout=5)
            if respuesta.status_code == 200:
                st.sidebar.success("¡Mensaje enviado! Revisa tu Telegram.")
            else:
                st.sidebar.error("Error: Revisa el Token y el Chat ID.")
        except:
            st.sidebar.error("Error de conexión al enviar el mensaje.")
    else:
        st.sidebar.warning("Falta el Token o el Chat ID.")

# --- PANEL DE ESTADÍSTICAS ---
st.sidebar.divider()
st.sidebar.subheader("📊 Tu Diario de Trading")
total_ops = st.session_state.ganadas + st.session_state.perdidas
if total_ops > 0:
    win_rate = (st.session_state.ganadas / total_ops) * 100
    st.sidebar.metric("Win Rate (Acierto)", f"{win_rate:.1f}%")
    st.sidebar.write(f"✅ Ganadas: {st.session_state.ganadas}")
    st.sidebar.write(f"❌ Perdidas: {st.session_state.perdidas}")
else:
    st.sidebar.write("Aún no hay operaciones registradas.")

if st.sidebar.button("Borrar Historial"):
    st.session_state.ganadas = 0
    st.session_state.perdidas = 0
    st.rerun()

# --- CONEXIÓN CON KRAKEN ---
@st.cache_resource
def conectar_kraken():
    return ccxt.kraken()

exchange = conectar_kraken()

def obtener_datos_reales(timeframe):
    velas = exchange.fetch_ohlcv('BTC/USD', timeframe=timeframe, limit=500)
    df = pd.DataFrame(velas, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# --- EL CEREBRO DE LA IA ---
try:
    df = obtener_datos_reales(temporalidad)
    
    # Indicadores
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['EMA_9'] = ta.trend.EMAIndicator(df['Close'], window=9).ema_indicator()
    df['EMA_21'] = ta.trend.EMAIndicator(df['Close'], window=21).ema_indicator()
    
    # Target (1 si subió, 0 si bajó)
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df_ml = df.dropna()
    
    # Entrenamiento
    X = df_ml[['RSI', 'EMA_9', 'EMA_21', 'Close']]
    y = df_ml['Target']
    modelo = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    modelo.fit(X, y)
    
    # Predicción
    ultima_vela = df.iloc[-1:]
    ultimo_rsi = ultima_vela['RSI'].values[0]
    ultimo_close = ultima_vela['Close'].values[0]
    ultima_ema9 = ultima_vela['EMA_9'].values[0]
    ultima_ema21 = ultima_vela['EMA_21'].values[0]
    
    datos_prediccion = pd.DataFrame(
        [[ultimo_rsi, ultima_ema9, ultima_ema21, ultimo_close]], 
        columns=['RSI', 'EMA_9', 'EMA_21', 'Close']
    )
    
    probabilidades = modelo.predict_proba(datos_prediccion)[0]
    prob_compra = probabilidades[1] * 100
    prob_venta = probabilidades[0] * 100
    
    # --- MOSTRAR EN PANTALLA ---
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precio BTC Real", f"${ultimo_close:,.2f}")
    col2.metric("RSI (14)", f"{ultimo_rsi:.1f}")
    col3.metric("EMA Rápida (9)", f"${ultima_ema9:,.2f}")
    col4.metric("EMA Lenta (21)", f"${ultima_ema21:,.2f}")
    
    st.divider()
    st.subheader(f"Predicción para la próxima vela de {temporalidad}")
    
    # Lógica de Señal
    UMBRAL_CERTEZA = 60.0
    señal_emitida = False
    mensaje_alerta = ""

    if prob_compra >= UMBRAL_CERTEZA:
        st.success(f"🟢 SEÑAL DE COMPRA (CALL) - Probabilidad: {prob_compra:.1f}%")
        señal_emitida = True
        mensaje_alerta = f"🟢 SEÑAL DE COMPRA (CALL) en {temporalidad}\nProbabilidad: {prob_compra:.1f}%\nPrecio: ${ultimo_close:,.2f}"
    elif prob_venta >= UMBRAL_CERTEZA:
        st.error(f"🔴 SEÑAL DE VENTA (PUT) - Probabilidad: {prob_venta:.1f}%")
        señal_emitida = True
        mensaje_alerta = f"🔴 SEÑAL DE VENTA (PUT) en {temporalidad}\nProbabilidad: {prob_venta:.1f}%\nPrecio: ${ultimo_close:,.2f}"
    else:
        st.warning(f"⚪ NO OPERAR - Mercado indeciso (Compra: {prob_compra:.1f}% / Venta: {prob_venta:.1f}%)")

    # --- ENVIAR ALERTA A TELEGRAM AUTOMÁTICA ---
    # Solo envía si hay señal y la probabilidad es mayor a 65% para no spamear
    if señal_emitida and telegram_token and telegram_chat_id:
        if prob_compra >= 65.0 or prob_venta >= 65.0:
            url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
            payload = {"chat_id": telegram_chat_id, "text": mensaje_alerta}
            try:
                requests.post(url, data=payload, timeout=5)
            except:
                pass 

    # --- BOTONES DEL DIARIO DE TRADING ---
    st.write("---")
    st.write("**Registra el resultado de tu operación:**")
    col_b1, col_b2, col_b3 = st.columns([1, 1, 2])
    if col_b1.button("✅ Gané"):
        st.session_state.ganadas += 1
        st.rerun()
    if col_b2.button("❌ Perdí"):
        st.session_state.perdidas += 1
        st.rerun()

    # --- GRÁFICA ---
    st.subheader("Gráfica del Mercado Real (Últimas 30 velas)")
    chart_data = df[['Close']].tail(30).reset_index()
    chart = alt.Chart(chart_data).mark_line(color="#00FF00").encode(
        x=alt.X('timestamp:T', title='Hora'),
        y=alt.Y('Close:Q', scale=alt.Scale(zero=False), title='Precio BTC')
    ).properties(height=350)
    st.altair_chart(chart, width='stretch')
    
    # --- BOTÓN DE ACTUALIZACIÓN MANUAL ---
    if st.button("🔄 Actualizar Análisis Ahora"):
        st.rerun()

except Exception as e:
    st.error(f"Error de conexión: {e}")
