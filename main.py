import os
import sys
import io
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import html

# ==========================================
# TELEGRAM NOTIFICATION SETUP
# ==========================================
def send_telegram_message(message):
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        print("⚠️ Telegram credentials not found.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    # Menggunakan HTML parse mode
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'HTML'
    }
    
    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Failed: {e}")

# ==========================================
# MAIN LOGIC
# ==========================================
def run_analysis():
    target_list = [
        "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", "ASII.JK",
        "MEDC.JK", "AKRA.JK", "PGAS.JK", "ADRO.JK", "PTBA.JK", "ITMG.JK",
        "UNTR.JK", "ANTM.JK", "MDKA.JK", "BRMS.JK", "ICBP.JK", "UNVR.JK", "AMRT.JK"
    ]

    macro_tickers = {
        "^VIX": "VIX (Panic)", "GC=F": "Gold", "CL=F": "Oil", 
        "^DJI": "Dow Jones", "IDR=X": "USD/IDR", "JPY=X": "USD/JPY"
    }

    CORR_WINDOW = 60
    VOL_WINDOW = 20
    PENALTY_FACTOR = 0.5

    all_tickers = list(macro_tickers.keys()) + target_list
    try:
        data_full = yf.download(all_tickers, period="2y", progress=False)
        data_close = data_full['Close'].ffill()
        data_volume = data_full['Volume'].ffill()
    except Exception as e:
        print(f"❌ Error Download: {e}"); return

    # Helper Functions
    def get_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/period, adjust=False).mean()
        return 100 - (100 / (1 + (gain/loss)))

    # Macro processing
    macro_returns = data_close[list(macro_tickers.keys())].pct_change().dropna()
    macro_scaled = pd.DataFrame(RobustScaler().fit_transform(macro_returns), 
                                index=macro_returns.index, columns=macro_returns.columns)

    report_html = "<b>🚀 MARKET INTELLIGENCE REPORT</b>\n"
    report_html += f"<i>Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</i>\n\n"

    final_results = []
    for stock in target_list:
        try:
            stock_price = data_close[stock].dropna()
            stock_vol = data_volume[stock].dropna()
            stock_ret = stock_price.pct_change()
            common = macro_scaled.index.intersection(stock_ret.index)
            
            df_ind = pd.concat([macro_scaled.loc[common], stock_ret.loc[common].shift(-1)], axis=1).dropna()
            if len(df_ind) < CORR_WINDOW: continue
            
            corr_matrix = df_ind.iloc[-CORR_WINDOW:].corr()
            stock_corr = corr_matrix[stock].drop(stock, errors='ignore')
            
            signals = macro_scaled.iloc[-1] * stock_corr
            raw_score = signals.sum()
            agreement = (abs(raw_score) / signals.abs().sum()) if signals.abs().sum() != 0 else 0
            
            vol_z = ((stock_vol - stock_vol.rolling(VOL_WINDOW).mean()) / stock_vol.rolling(VOL_WINDOW).std()).iloc[-1]
            rsi = get_rsi(stock_price).iloc[-1]
            
            final_score = raw_score * PENALTY_FACTOR if (raw_score > 0.1 and vol_z < 0) else raw_score
            driver = stock_corr.abs().idxmax()
            conf = (stock_corr.abs().max() * 100) * (1.0 if (raw_score <= 0.1 or vol_z >= 0) else PENALTY_FACTOR)

            pred = "⚪ Neutral"
            if final_score > 0.35 and agreement > 0.6: pred = "🚀 BUY SIGNAL"
            elif final_score > 0.1: pred = "📈 Slight Uptick"
            elif final_score < -0.35 and agreement > 0.6: pred = "🔻 SELL SIGNAL"
            elif final_score < -0.1: pred = "📉 Slight Drop"
            
            if vol_z < 0 and raw_score > 0.1: pred += " (Low Vol)"

            final_results.append({
                "stock": stock, "pred": pred, "score": final_score, "conf": conf,
                "driver": macro_tickers.get(driver, driver), "agree": agreement * 100,
                "rsi": rsi, "vol": vol_z, "price": stock_price.iloc[-1]
            })
        except: continue

    # Urutkan berdasarkan Score tertinggi
    sorted_results = sorted(final_results, key=lambda x: x['score'], reverse=True)

    # Bangun pesan dalam format Card agar muat semua data
    for item in sorted_results:
        k_icon = "💎" if item['agree'] > 80 else "⚠️" if item['agree'] < 40 else ""
        
        card = (
            f"<b>{item['stock']}</b> | {item['pred']}\n"
            f"<code>Price : {item['price']:,.0f} (RSI: {item['rsi']:.1f})</code>\n"
            f"<code>Driver: {item['driver']:<12} | Conf: {item['conf']:.0f}%</code>\n"
            f"<code>Agree : {item['agree']:.0f}% {k_icon:<1} | Vol Z: {item['vol']:.1f}</code>\n"
            f"────────────────────\n"
        )
        report_html += card

    send_telegram_message(report_html)

if __name__ == "__main__":
    run_analysis()