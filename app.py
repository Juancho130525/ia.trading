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

# --- ESTILOS CSS PROFESIONALES (Glassmorphism y Ocultar menús) ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            
            /* Efecto Cristal para las métricas */
            [data-testid="stMetricValue"] {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
                padding: 10px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                color: #00FF00;
            }
            /* Estilo de las pestañas */
            .stTabs [data-baseweb="tab-list"] {
                gap: 24px;
            }
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                white-space: pre-wrap;
                background-color: #262730;
                border-radius: 4px 4px 0px 0px;
                gap: 1px;
                padding-top: 10px;
                padding-bottom: 10px;
            }
            .stTabs [aria-selected="true"] {
                background-color: #00FF00;
                color: #0E1117;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Auto-refrescar cada 30 segundos
st_autorefresh(interval=30000, key="datarefresh")

# --- TÍTULO PRINCIPAL ---
st.markdown("<h1 style='text-align: center; color: #00FF00;'>⚡ QUANTUM CRYPTO AI ⚡</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Terminal de Análisis Algorítmico en Tiempo Real</p>", unsafe_allow_html=True)

# --- ESTADO DE LA SESIÓN ---
if 'ganadas' not in st.session_state: st.session_state.ganadas = 0
if 'perdidas' not in st.session_state: st.session_state.perdidas = 0

# --- BARRA LATERAL ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2933/2933245.png", width=80)
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
            st.sidebar.success("¡Mensaje enviado!")
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

        # ==========================================
        # AQUÍ EMPIEZA LA MAGIA DE LAS PESTAÑAS
        # ==========================================
        tab_terminal, tab_estadisticas, tab_config = st.tabs(["📈 Terminal de Trading", "📊 Estadísticas", "⚙️ Info & Riesgo"])

        # --- PESTAÑA 1: TERMINAL DE TRADING ---
        with tab_terminal:
            # Tarjetas de datos
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Precio BTC", f"${ultimo_close:,.2f}")
            col2.metric("RSI (14)", f"{ultimo_rsi:.1f}", "Sobreventa" if ultimo_rsi < 30 else ("Sobrecompra" if ultimo_rsi > 70 else "Neutral"))
            col3.metric("EMA Rápida", f"${ultima_ema9:,.2f}")
            col4.metric("EMA Lenta", f"${ultima_ema21:,.2f}")

            st.divider()

            # Panel de Señal
            st.markdown("### 🧠 Veredicto de la Inteligencia Artificial")
            UMBRAL = 60.0
            señal_emitida = False
            
            if prob_compra >= UMBRAL:
                st.markdown(f"<div style='background-color: rgba(0, 255, 0, 0.2); padding: 20px; border-radius: 10px; border-left: 5px solid #00FF00;'><h2 style='color: #00FF00; margin:0;'>🟢 COMPRA (CALL) - {prob_compra:.1f}%</h2></div>", unsafe_allow_html=True)
                señal_emitida = True
                mensaje_alerta = f"🟢 SEÑAL DE COMPRA (CALL) en {temporalidad}\nProbabilidad: {prob_compra:.1f}%"
            elif prob_venta >= UMBRAL:
                st.markdown(f"<div style='background-color: rgba(255, 0, 0, 0.2); padding: 20px; border-radius: 10px; border-left: 5px solid #FF0000;'><h2 style='color: #FF0000; margin:0;'>🔴 VENTA (PUT) - {prob_venta:.1f}%</h2></div>", unsafe_allow_html=True)
                señal_emitida = True
                mensaje_alerta = f"🔴 SEÑAL DE VENTA (PUT) en {temporalidad}\nProbabilidad: {prob_venta:.1f}%"
            else:
                st.markdown(f"<div style='background-color: rgba(255, 255, 0, 0.1); padding: 20px; border-radius: 10px; border-left: 5px solid #FFD700;'><h3 style='color: #FFD700; margin:0;'>⚪ MERCADO INDECISO - NO OPERAR</h3><p style='margin:0; color: gray;'>Probabilidad insuficiente (C: {prob_compra:.1f}% / V: {prob_venta:.1f}%)</p></div>", unsafe_allow_html=True)

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

            # Gráfica de Velas
            st.markdown("### 📈 Gráfico de Velas en Tiempo Real")
            df_plot = df.tail(40)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], increasing_line_color='#00FF00', decreasing_line_color='#FF0000', name='Precio BTC'))
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA_9'], line=dict(color='#00BFFF', width=1.5), name='EMA 9'))
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA_21'], line=dict(color='#FF00FF', width=1.5), name='EMA 21'))
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False, height=450, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)


        # --- PESTAÑA 2: ESTADÍSTICAS ---
        with tab_estadisticas:
            st.markdown("### 📊 Análisis de Rendimiento")
            if total_ops > 0:
                st.write(f"Basado en **{total_ops}** operaciones registradas.")
                
                # Curva de capital simulada
                st.markdown("#### Curva de Crecimiento Simulada (Base $100)")
                np.random.seed(42) 
                winrate_decimal = (st.session_state.ganadas / total_ops)
                # Simulamos 50 operaciones futuras basadas en tu winrate real
                movimientos = np.random.choice([1.82, -1.0], size=50, p=[winrate_decimal, 1-winrate_decimal])
                curva = 100 + np.cumsum(movimientos * 2) 
                
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(y=curva, mode='lines', fill='tozeroy', line=dict(color='#00FF00', width=3)))
                fig_equity.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_equity, use_container_width=True)
                st.info("Nota: Esta curva es una proyección matemática basada en tu porcentaje de acierto actual, no son ganancias reales.")
            else:
                st.warning("Aún no has registrado operaciones. Usa los botones 'Ganar' o 'Perder' en la pestaña del Terminal para empezar a construir tus estadísticas.")


        # --- PESTAÑA 3: INFO & RIESGO ---
        with tab_config:
            st.markdown("### 🌐 Confluencia de Tendencias")
            st.write("Análisis rápido de la fuerza del mercado actual:")
            
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                st.metric("Tendencia Corto Plazo", "ALCISTA 🟢" if ultimo_close > ultima_ema9 else "BAJISTA 🔴")
            with col_t2:
                st.metric("Fuerza del RSI", "Fuerte 💪" if ultimo_rsi > 50 else "Débil 📉")
            with col_t3:
                st.metric("Tendencia Largo Plazo", "ALCISTA 🟢" if ultima_ema9 > ultima_ema21 else "BAJISTA 🔴")

            st.divider()
            st.markdown("### ⚠️ Aviso Legal y de Riesgo")
            st.markdown(
                """
                <div style='background-color: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 10px; border-left: 5px solid #FFD700;'>
                    <p><b>Quantum Crypto AI © 2026</b> | Plataforma de Análisis Algorítmico</p>
                    <p style='color: gray; font-size: 14px;'>
                    El trading de opciones binarias y criptomonedas implica un alto nivel de riesgo de pérdida de capital. 
                    Esta herramienta es un asistente estadístico basado en Machine Learning y <b>no garantiza ganancias</b>. 
                    Los mercados son impredecibles y los resultados pasados no aseguran resultados futuros. 
                    Opere siempre con gestión de riesgo y bajo su propia responsabilidad.
                    </p>
                </div>
                """, 
                unsafe_allow_html=True
            )

except Exception as e:
    st.error(f"Error de conexión con el mercado: {e}")
