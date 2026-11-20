# --- CEREBRO DE LA IA (CORREGIDO Y BALANCEADO) ---
try:
    with st.spinner('Conectando al mercado y analizando patrones...'):
        df = obtener_datos_reales(temporalidad)
        
        # Indicadores
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['EMA_9'] = ta.trend.EMAIndicator(df['Close'], window=9).ema_indicator()
        df['EMA_21'] = ta.trend.EMAIndicator(df['Close'], window=21).ema_indicator()
        
        # 1. Calculamos la "Distancia porcentual" a las EMAs (Esto ayuda a la IA a entender la tendencia)
        df['Dist_EMA9'] = (df['Close'] - df['EMA_9']) / df['EMA_9'] * 100
        df['Dist_EMA21'] = (df['Close'] - df['EMA_21']) / df['EMA_21'] * 100
        
        # Target: 1 si subió, 0 si bajó
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        df_ml = df.dropna()
        
        # 2. Entrenamiento Balanceado
        # Ya no usamos 'Close' crudo, usamos las distancias y el RSI
        X = df_ml[['RSI', 'Dist_EMA9', 'Dist_EMA21']]
        y = df_ml['Target']
        
        # class_weight='balanced' soluciona el problema de las señales de venta excesivas
        modelo = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced')
        modelo.fit(X, y)
        
        # 3. Predicción con la última vela
        ultima_vela = df.iloc[-1:]
        ultimo_rsi = ultima_vela['RSI'].values[0]
        ultimo_close = ultima_vela['Close'].values[0]
        ultima_ema9 = ultima_vela['EMA_9'].values[0]
        ultima_ema21 = ultima_vela['EMA_21'].values[0]
        
        ultimo_dist9 = (ultimo_close - ultima_ema9) / ultima_ema9 * 100
        ultimo_dist21 = (ultimo_close - ultima_ema21) / ultima_ema21 * 100
        
        datos_pred = pd.DataFrame([[ultimo_rsi, ultimo_dist9, ultimo_dist21]], columns=['RSI', 'Dist_EMA9', 'Dist_EMA21'])
        probabilidades = modelo.predict_proba(datos_pred)[0]
        prob_compra = probabilidades[1] * 100
        prob_venta = probabilidades[0] * 100

        # ==========================================
        # SISTEMA DE PESTAÑAS (4 PESTAÑAS)
        # ==========================================
        tab_terminal, tab_estadisticas, tab_noticias, tab_config = st.tabs(["📈 Terminal", "📊 Estadísticas", "📰 Noticias", "⚙️ Info & Riesgo"])

        # --- PESTAÑA 1: TERMINAL ---
        with tab_terminal:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Precio BTC", f"${ultimo_close:,.2f}")
            col2.metric("RSI (14)", f"{ultimo_rsi:.1f}", "Sobreventa" if ultimo_rsi < 30 else ("Sobrecompra" if ultimo_rsi > 70 else "Neutral"))
            col3.metric("EMA Rápida", f"${ultima_ema9:,.2f}")
            col4.metric("EMA Lenta", f"${ultima_ema21:,.2f}")

            st.divider()
            st.markdown("### 🧠 Veredicto de la Inteligencia Artificial")
            UMBRAL = 60.0
            señal_emitida = False
            
            if prob_compra >= UMBRAL:
                st.markdown(f"<div style='background-color: rgba(0, 255, 0, 0.2); padding: 20px; border-radius: 10px; border-left: 5px solid #00FF00;'><h2 style='color: #00FF00; margin:0;'>🟢 COMPRA (CALL) - {prob_compra:.1f}%</h2></div>", unsafe_allow_html=True)
                señal_emitida = True
                mensaje_alerta = f"🟢 SEÑAL DE COMPRA en {temporalidad} ({prob_compra:.1f}%)"
            elif prob_venta >= UMBRAL:
                st.markdown(f"<div style='background-color: rgba(255, 0, 0, 0.2); padding: 20px; border-radius: 10px; border-left: 5px solid #FF0000;'><h2 style='color: #FF0000; margin:0;'>🔴 VENTA (PUT) - {prob_venta:.1f}%</h2></div>", unsafe_allow_html=True)
                señal_emitida = True
                mensaje_alerta = f"🔴 SEÑAL DE VENTA en {temporalidad} ({prob_venta:.1f}%)"
            else:
                st.markdown(f"<div style='background-color: rgba(255, 255, 0, 0.1); padding: 20px; border-radius: 10px; border-left: 5px solid #FFD700;'><h3 style='color: #FFD700; margin:0;'>⚪ MERCADO INDECISO - NO OPERAR</h3><p style='margin:0; color: gray;'>Probabilidad insuficiente (C: {prob_compra:.1f}% / V: {prob_venta:.1f}%)</p></div>", unsafe_allow_html=True)

            if señal_emitida and telegram_token and telegram_chat_id:
                if prob_compra >= 65.0 or prob_venta >= 65.0:
                    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
                    try: requests.post(url, data={"chat_id": telegram_chat_id, "text": mensaje_alerta}, timeout=3)
                    except: pass

            st.write("")
            c1, c2, _ = st.columns([1, 1, 3])
            if c1.button("✅ GANÉ", use_container_width=True):
                st.session_state.ganadas += 1
                st.rerun()
            if c2.button("❌ PERDÍ", use_container_width=True):
                st.session_state.perdidas += 1
                st.rerun()

            st.divider()
            st.markdown("### 📈 Terminal de Trading en Tiempo Real")
            df_plot = df.tail(40)
            
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2], subplot_titles=("", "Volumen", "RSI (14)"))
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], increasing_line_color='#00FF00', decreasing_line_color='#FF0000', name='Precio BTC'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA_9'], line=dict(color='#00BFFF', width=1.5), name='EMA 9'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA_21'], line=dict(color='#FF00FF', width=1.5), name='EMA 21'), row=1, col=1)
            colors = ['#00FF00' if close > open else '#FF0000' for close, open in zip(df_plot['Close'], df_plot['Open'])]
            fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], marker_color=colors, name='Volumen'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], line=dict(color='#FFD700', width=2), name='RSI'), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False, height=700, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
