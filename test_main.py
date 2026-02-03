import os
import logging
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import matplotlib
from datetime import datetime
import pytz
from sklearn.preprocessing import RobustScaler

matplotlib.use('Agg')

# ======================
# CONFIG & LOGGING
# ======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

TARGET_LIST = [
    "BBCA.JK","BBRI.JK","BMRI.JK","BBNI.JK","TLKM.JK","ASII.JK",
    "MEDC.JK","AKRA.JK","PGAS.JK","ADRO.JK","PTBA.JK","ITMG.JK",
    "UNTR.JK","ANTM.JK","MDKA.JK","BRMS.JK","ICBP.JK","UNVR.JK","AMRT.JK"
]

MACRO_TICKERS = {
    "^VIX":"VIX", "GC=F":"Gold", "CL=F":"Oil",
    "^DJI":"Dow Jones", "IDR=X":"USDIDR", "JPY=X":"USDJPY"
}

# Params Institutional (Versi 2)
INST_CONFIG = {
    "CORR_WINDOW": 60,
    "VOL_WINDOW": 20,
    "PENALTY_FACTOR": 0.5,
    "BUY_TH": 0.35,
    "SELL_TH": -0.35,
    "AGREEMENT_MIN": 0.6
}

# ======================
# TELEGRAM HELPERS
# ======================
def send_telegram_photo(photo_buffer, caption):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    photo_buffer.seek(0)
    try:
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'HTML'},
                      files={'photo': ('report.jpg', photo_buffer, 'image/jpeg')}, timeout=30)
    except Exception as e: logging.warning(f"Telegram failed: {e}")

def send_telegram_message(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try: requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': text, 'parse_mode': 'HTML'}, timeout=10)
    except Exception as e: logging.warning(f"Telegram failed: {e}")

# ======================
# CORE MATH FUNCTIONS
# ======================
def get_wilders_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def generate_table_img(df, title, run_time):
    fig_height = 2.5 + len(df) * 0.4
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    plt.title(title, fontsize=14, weight='bold')
    plt.figtext(0.5, 0.02, f"Run at: {run_time}", ha="center", fontsize=8)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', bbox_inches='tight', dpi=150)
    plt.close(fig)
    return buf

# ======================
# ANALYSIS ENGINE
# ======================
def run_combined_analysis():
    tz = pytz.timezone('Asia/Jakarta')
    run_time = datetime.now(tz).strftime('%d %b %Y, %H:%M WIB')
    
    logging.info("🧪 TEST MODE: Running analysis without time validation")
    
    # 1. Download Data
    all_tickers = list(MACRO_TICKERS.keys()) + TARGET_LIST
    data = yf.download(all_tickers, period="2y", progress=False)
    
    close = data['Close'].ffill()
    volume = data['Volume'].ffill()
    
    # 2. Macro Pre-processing (Robust Scaling)
    macro_rets = close[list(MACRO_TICKERS.keys())].pct_change().dropna()
    scaler = RobustScaler()
    macro_scaled = pd.DataFrame(scaler.fit_transform(macro_rets), 
                                index=macro_rets.index, columns=macro_rets.columns)

    inst_results = []
    
    # 3. Institutional Analysis Loop (Versi 2)
    for stock in TARGET_LIST:
        try:
            s_price = close[stock].dropna()
            s_vol = volume[stock].dropna()
            s_ret = s_price.pct_change()
            
            # Lead-Lag: Macro(T) vs Stock(T+1)
            df_sync = pd.concat([macro_scaled, s_ret.shift(-1)], axis=1).dropna()
            if len(df_sync) < INST_CONFIG["CORR_WINDOW"]: continue
            
            recent = df_sync.iloc[-INST_CONFIG["CORR_WINDOW"]:]
            corr_matrix = recent.corr()
            stock_corr = corr_matrix[stock].drop(stock)
            
            # Signal Calculation
            latest_macro = macro_scaled.iloc[-1]
            indiv_signals = latest_macro * stock_corr
            raw_score = indiv_signals.sum()
            agreement = (abs(raw_score) / indiv_signals.abs().sum()) if indiv_signals.abs().sum() != 0 else 0
            
            # Volume & RSI
            vol_z = (s_vol.iloc[-1] - s_vol.rolling(20).mean().iloc[-1]) / s_vol.rolling(20).std().iloc[-1]
            rsi = get_wilders_rsi(s_price).iloc[-1]
            
            final_score = raw_score
            penalty_tag = ""
            if raw_score > 0.1 and vol_z < 0:
                final_score *= INST_CONFIG["PENALTY_FACTOR"]
                penalty_tag = " (Low Vol)"
            
            # Classification
            conf = (stock_corr.abs().max() * 100) * (1.0 if penalty_tag == "" else 0.5)
            driver = MACRO_TICKERS.get(stock_corr.abs().idxmax(), "N/A")
            
            signal = "Neutral"
            if final_score > INST_CONFIG["BUY_TH"] and agreement > INST_CONFIG["AGREEMENT_MIN"]: signal = "🚀 BUY"
            elif final_score > 0.1: signal = "📈 Uptick"
            elif final_score < INST_CONFIG["SELL_TH"] and agreement > INST_CONFIG["AGREEMENT_MIN"]: signal = "🔻 SELL"
            elif final_score < -0.1: signal = "📉 Drop"
            
            if signal != "Neutral":
                inst_results.append([stock.replace(".JK",""), signal + penalty_tag, f"{conf:.1f}%", driver, f"{rsi:.1f}", f"{s_price.iloc[-1]:,.0f}"])

        except Exception as e:
            logging.error(f"Error analyzing {stock}: {e}")

    # 4. Report Generation
    if inst_results:
        df_inst = pd.DataFrame(inst_results, columns=["TICKER","SIGNAL","CONF","DRIVER","RSI","PRICE"])
        img_table = generate_table_img(df_inst, "🏦 INSTITUTIONAL SIGNALS (Robust Macro) - TEST MODE", run_time)
        
        caption = (f"<b>🧪 TEST MODE - INSTITUTIONAL REPORT</b>\n"
                   f"Strategy: Robust Scaling + Lead-Lag (T+1)\n"
                   f"Analisis ini lebih sensitif terhadap pergerakan Komoditas Dunia.")
        send_telegram_photo(img_table, caption)
        logging.info(f"✅ Report sent with {len(inst_results)} signals")
    else:
        send_telegram_message(f"🏦 <b>INSTITUTIONAL (TEST)</b>: No clear signals detected at {run_time}")
        logging.info("ℹ️ No signals to report")

if __name__ == "__main__":
    logging.info("="*60)
    logging.info("🧪 RUNNING IN TEST MODE - NO TIME VALIDATION")
    logging.info("="*60)
    run_combined_analysis()
    logging.info("✅ Test completed")
