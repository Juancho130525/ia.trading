import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import ta
import altair as alt

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Predictor Crypto IDX", layout="wide")
st.title("IA para Crypto IDX (Conectado a Binance)")
st.write("Esta IA analiza el mercado real de Bitcoin para predecir la dirección de la próxima vela.")

# --- BARRA LATERAL ---
st.sidebar.header("Configuración")
temporalidad = st.sidebar.selectbox(
    "Selecciona la temporalidad:",
    options=['1m', '5m', '15m', '1h'],
    index=2 # Por defecto 15m
)

# --- CONEXIÓN CON BINANCE ---
@st.cache_resource
def conectar_bybit():
    return ccxt.bybit()

exchange = conectar_bybit()

def obtener_datos_reales(timeframe):
    velas = exchange.fetch_ohlcv('BTC/USDT', timeframe=timeframe, limit=500)
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
    
    # CORRECCIÓN 1: Pasamos los datos como un DataFrame con nombres de columnas
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
    
    # FILTRO DE SEGURIDAD
    UMBRAL_CERTEZA = 60.0
    
    if prob_compra >= UMBRAL_CERTEZA:
        st.success(f"🟢 SEÑAL DE COMPRA (CALL) - Probabilidad: {prob_compra:.1f}%")
    elif prob_venta >= UMBRAL_CERTEZA:
        st.error(f"🔴 SEÑAL DE VENTA (PUT) - Probabilidad: {prob_venta:.1f}%")
    else:
        st.warning(f"⚪ NO OPERAR - Mercado indeciso (Compra: {prob_compra:.1f}% / Venta: {prob_venta:.1f}%). Espera una mejor oportunidad.")

    # Gráfica mejorada con Zoom
    st.subheader("Gráfica del Mercado Real (Últimas 30 velas)")
    
    chart_data = df[['Close']].tail(30).reset_index()
    
    chart = alt.Chart(chart_data).mark_line(color="#00FF00").encode(
        x=alt.X('timestamp:T', title='Hora'),
        y=alt.Y('Close:Q', scale=alt.Scale(zero=False), title='Precio BTC')
    ).properties(height=350)
    
    # CORRECCIÓN 2: Reemplazamos use_container_width por width='stretch'
    st.altair_chart(chart, width='stretch')
    
    st.button("🔄 Actualizar Análisis")

except Exception as e:
    st.error(f"Error de conexión: {e}")
