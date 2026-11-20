import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import ta
import requests
import plotly.graph_objects as go
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Quantum Crypto AI", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# Ocultar menús de Streamlit para que se vea como una App real
# Ocultar menús de Streamlit sin dañar la flechita de la barra lateral
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            /* Esto hace que la flechita sea visible y verde neón */
            [data-testid="collapsedControl"] {
                display: block;
                color: #00FF00;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Auto-refrescar cada 30 segundos para que el mercado esté en vivo
st_autorefresh(interval=30000, key="datarefresh")

# --- TÍTULO PRINCIPAL (Con estilo) ---
st.markdown("<h1 style='text-align: center; color: #00FF00;'>⚡ QUANTUM CRYPTO AI ⚡</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Análisis algorítmico en tiempo real conectado a Kraken</p>", unsafe_allow_html=True)

# --- ESTADO DE LA SESIÓN ---
if 'ganadas' not in st.session_state: st.session_state.ganadas = 0
if 'perdidas' not in st.session_state: st.session_state.perdidas = 0

# --- BARRA LATERAL (Controles) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2933/2933245.png", width=80) # Icono de gráfico
st.sidebar.header("Configuración del Sistema")
temporalidad = st.sidebar.selectbox("Temporalidad de Velas", options=['1m', '5m', '15m', '1h'], index=2)

st.sidebar.divider()
st.sidebar.subheader("📱 Alertas Push (Telegram)")
telegram_token = st.sidebar.text_input("Token del Bot", type="password")
telegram_chat_id = st.sidebar.text_input("Chat ID", type="password")

if st.sidebar.button("Probar mensaje de Telegram"):
    if telegram_token and telegram_chat_id:
        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        try:
            requests.post(url, data={"chat_id": telegram_chat_id, "text": "⚡ ALERTA: Sistema Quantum AI conectado correctamente."}, timeout=5)
            st.sidebar.success("¡Mensaje enviado! Revisa tu Telegram.")
        except:
            st.sidebar.error("Error de conexión.")
    else:
        st.sidebar.warning("Falta Token o Chat ID.")

st.sidebar.divider()
st.sidebar.subheader("📊 Rendimiento de tu Cuenta")
total_ops = st.session_state.ganadas + st.session_state.perdidas
if total_ops > 0:
    win_rate = (st.session_state.ganadas / total_ops) * 100
    st.sidebar.markdown(f"### Win Rate: <span style='color:#00FF00;'>{win_rate:.1f}%</span>", unsafe_allow_html=True)
    st.sidebar.write(f"✅ Ganadas: **{st.session_state.ganadas}**")
    st.sidebar.write(f"❌ Perdidas: **{st.session_state.perdidas}**")
else:
    st.sidebar.info("Esperando primera operación...")

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

# --- CEREBRO DE LA IA ---
try:
    with st.spinner('Conectando al mercado y analizando patrones...'):
        df = obtener_datos_reales(temporalidad)
        
        # Indicadores
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['EMA_9'] = ta.trend.EMAIndicator(df['Close'], window=9).ema_indicator()
        df['EMA_21'] = ta.trend.EMAIndicator(df['Close'], window=21).ema_indicator()
        
        # Target y Entrenamiento
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        df_ml = df.dropna()
        
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
        
        datos_pred = pd.DataFrame([[ultimo_rsi, ultima_ema9, ultima_ema21, ultimo_close]], columns=['RSI', 'EMA_9', 'EMA_21', 'Close'])
        probabilidades = modelo.predict_proba(datos_pred)[0]
        prob_compra = probabilidades[1] * 100
        prob_venta = probabilidades[0] * 100

        # --- TARJETAS DE DATOS (Métricas) ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Precio BTC", f"${ultimo_close:,.2f}")
        col2.metric("RSI (14)", f"{ultimo_rsi:.1f}", "Sobreventa" if ultimo_rsi < 30 else ("Sobrecompra" if ultimo_rsi > 70 else "Neutral"))
        col3.metric("EMA Rápida", f"${ultima_ema9:,.2f}")
        col4.metric("EMA Lenta", f"${ultima_ema21:,.2f}")

        st.divider()

        # --- PANEL DE SEÑAL ---
        st.markdown("### 🧠 Veredicto de la Inteligencia Artificial")
        
        UMBRAL = 60.0
        señal_emitida = False
        
        if prob_compra >= UMBRAL:
            st.markdown(f"<div style='background-color: rgba(0, 255, 0, 0.2); padding: 20px; border-radius: 10px; border-left: 5px solid #00FF00;'><h2 style='color: #00FF00; margin:0;'>🟢 COMPRA (CALL) - {prob_compra:.1f}% de Probabilidad</h2></div>", unsafe_allow_html=True)
            señal_emitida = True
            mensaje_alerta = f"🟢 SEÑAL DE COMPRA (CALL) en {temporalidad}\nProbabilidad: {prob_compra:.1f}%"
        elif prob_venta >= UMBRAL:
            st.markdown(f"<div style='background-color: rgba(255, 0, 0, 0.2); padding: 20px; border-radius: 10px; border-left: 5px solid #FF0000;'><h2 style='color: #FF0000; margin:0;'>🔴 VENTA (PUT) - {prob_venta:.1f}% de Probabilidad</h2></div>", unsafe_allow_html=True)
            señal_emitida = True
            mensaje_alerta = f"🔴 SEÑAL DE VENTA (PUT) en {temporalidad}\nProbabilidad: {prob_venta:.1f}%"
        else:
            st.markdown(f"<div style='background-color: rgba(255, 255, 0, 0.1); padding: 20px; border-radius: 10px; border-left: 5px solid #FFD700;'><h3 style='color: #FFD700; margin:0;'>⚪ MERCADO INDECISO - NO OPERAR</h3><p style='margin:0; color: gray;'>Probabilidad insuficiente (Compra: {prob_compra:.1f}% / Venta: {prob_venta:.1f}%)</p></div>", unsafe_allow_html=True)

        # Enviar Telegram
        if señal_emitida and telegram_token and telegram_chat_id:
            if prob_compra >= 65.0 or prob_venta >= 65.0:
                url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
                try: requests.post(url, data={"chat_id": telegram_chat_id, "text": mensaje_alerta}, timeout=3)
                except: pass

        # Botones de registro
        st.write("")
        c1, c2, _ = st.columns([1, 1, 3])
        if c1.button("✅ GANÉ", use_container_width=True):
            st.session_state.ganadas += 1
            st.rerun()
        if c2.button("❌ PERDÍ", use_container_width=True):
            st.session_state.perdidas += 1
            st.rerun()

        st.divider()

        # --- GRÁFICA DE VELAS JAPONESAS (PRO) ---
        st.markdown("### 📈 Gráfico de Velas en Tiempo Real")
        
        df_plot = df.tail(40) # Últimas 40 velas para que se vea bien
        
        fig = go.Figure()
        
        # Velas Japonesas
        fig.add_trace(go.Candlestick(
            x=df_plot.index, open=df_plot['Open'], high=df_plot['High'],
            low=df_plot['Low'], close=df_plot['Close'],
            increasing_line_color='#00FF00', decreasing_line_color='#FF0000',
            name='Precio BTC'
        ))
        
        # Líneas de EMAs
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA_9'], line=dict(color='#00BFFF', width=1.5), name='EMA 9'))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA_21'], line=dict(color='#FF00FF', width=1.5), name='EMA 21'))

        # Estilo visual del gráfico
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_rangeslider_visible=False,
            height=500,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error de conexión con el mercado: {e}")
