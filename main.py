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

matplotlib.use('Agg')

# ==========================================
# LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ==========================================
# CONFIG
# ==========================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

CORR_WINDOW = 60
VOL_WINDOW = 20
EWMA_LAMBDA = 0.94
MIN_AVG_VOL = 1_000_000
TSTAT_THRESHOLD = 1.5

target_list = [
    "BBCA.JK","BBRI.JK","BMRI.JK","BBNI.JK","TLKM.JK","ASII.JK",
    "MEDC.JK","AKRA.JK","PGAS.JK","ADRO.JK","PTBA.JK","ITMG.JK",
    "UNTR.JK","ANTM.JK","MDKA.JK","BRMS.JK","ICBP.JK","UNVR.JK","AMRT.JK"
]

macro_tickers = {
    "^VIX":"VIX","GC=F":"Gold","CL=F":"Oil",
    "YM=F":"DowFut","IDR=X":"USDIDR","JPY=X":"USDJPY"
}

# ==========================================
# TELEGRAM FUNCTIONS
# ==========================================
def send_telegram_photo(photo_buffer, caption):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram not configured.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    photo_buffer.seek(0)
    requests.post(url,
        data={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'HTML'},
        files={'photo': ('report.jpg', photo_buffer, 'image/jpeg')}
    )

def send_telegram_message(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram not configured.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={
        'chat_id': TELEGRAM_CHAT_ID,
        'text': text,
        'parse_mode': 'HTML'
    })

# ==========================================
# UTIL
# ==========================================
def ewma_weights(n, lam=0.94):
    w = np.array([(1-lam)*(lam**i) for i in range(n)][::-1])
    return w / w.sum()

def rolling_zscore(series, window):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def get_market_context():
    tz = pytz.timezone('Asia/Jakarta')
    now = datetime.now(tz)
    return now.strftime('%d %b %Y, %H:%M WIB')

# ==========================================
# FACTOR MODEL
# ==========================================
def rolling_factor_model(stock_ret, macro_ret):
    df = pd.concat([stock_ret, macro_ret], axis=1).dropna()
    if len(df) < CORR_WINDOW:
        return None

    window = df.iloc[-CORR_WINDOW:]
    y = window.iloc[:, 0].values
    X = window.iloc[:, 1:].values
    X = np.column_stack([np.ones(len(X)), X])

    w = ewma_weights(len(y), EWMA_LAMBDA)
    W = np.diag(w)

    beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
    residuals = y - X @ beta
    sigma2 = (w * residuals**2).sum() / (len(y)-X.shape[1])
    var_beta = sigma2 * np.linalg.inv(X.T @ W @ X)
    tstats = beta / np.sqrt(np.diag(var_beta))

    return beta[1:], tstats[1:], window.columns[1:]

# ==========================================
# IMAGE: TABLE
# ==========================================
def generate_table(df, run_time):
    fig_height = 3 + len(df)*0.45
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center',
        edges='closed'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    plt.title("INSTITUTIONAL MACRO FACTOR SIGNAL", fontsize=18, weight='bold')
    plt.figtext(0.5, 0.04, f"Generated: {run_time}", ha="center", fontsize=10)

    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', bbox_inches='tight', dpi=200)
    plt.close(fig)
    return buf

# ==========================================
# IMAGE: EXPLANATION
# ==========================================
def generate_explanation_image(df, run_time):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    top = df.iloc[0]

    explanation = f"""
📊 MARKET SIGNAL SUMMARY

🕒 {run_time}

🔥 Saham Terkuat: {top['TICKER']}
➡️ Signal: {top['SIGNAL']}
🎯 Confidence: {top['CONFIDENCE']}
🌍 Driver Utama: {top['MACRO DRIVER']}

📌 Artinya:
Model membaca tekanan dana besar dari faktor global
seperti Dollar, Oil, Gold, atau VIX.

BUY  = tekanan naik dominan
SELL = tekanan turun dominan
WAIT = belum ada arah kuat

⚠️ Bukan signal scalping.
Ini model arus dana & tekanan makro (swing).
"""
    plt.text(0.05, 0.95, explanation, fontsize=12, va='top')

    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', bbox_inches='tight', dpi=200)
    plt.close(fig)
    return buf

# ==========================================
# MAIN ENGINE
# ==========================================
def run_analysis():
    logging.info("Downloading market data...")
    tickers = list(macro_tickers.keys()) + target_list
    data = yf.download(tickers, period="2y", progress=False)

    close = data['Close'].ffill()
    volume = data['Volume'].ffill()

    macro_ret = close[list(macro_tickers.keys())].pct_change()
    results = []

    for stock in target_list:
        try:
            price = close[stock]
            vol = volume[stock]
            if vol.rolling(20).mean().iloc[-1] < MIN_AVG_VOL:
                continue

            stock_ret = price.pct_change()
            model = rolling_factor_model(stock_ret, macro_ret)
            if model is None:
                continue

            betas, tstats, drivers = model
            sig_mask = np.abs(tstats) > TSTAT_THRESHOLD
            if not sig_mask.any():
                continue

            latest_macro = macro_ret.iloc[-1]
            score = np.sum(betas[sig_mask] * latest_macro[sig_mask])

            vol_z = rolling_zscore(vol, VOL_WINDOW).iloc[-1]
            if vol_z < 0:
                score *= 0.7

            signal = "WAIT"
            if score > 0.01: signal = "BUY"
            if score < -0.01: signal = "SELL"

            confidence = min(95, np.mean(np.abs(tstats[sig_mask]))*20)
            main_driver = macro_tickers[drivers[np.argmax(np.abs(betas))]]

            results.append([
                stock.replace(".JK",""), signal,
                f"{price.iloc[-1]:,.0f}",
                f"{confidence:.1f}%",
                "LOW" if vol_z < 0 else "NORMAL",
                main_driver
            ])

        except Exception as e:
            logging.warning(f"Skip {stock}: {e}")

    if not results:
        logging.warning("No signals generated.")
        return

    df = pd.DataFrame(results,
        columns=["TICKER","SIGNAL","PRICE","CONFIDENCE","VOLUME","MACRO DRIVER"]
    ).sort_values("CONFIDENCE", ascending=False)

    run_time = get_market_context()

    # 1️⃣ TABLE
    img_table = generate_table(df, run_time)
    send_telegram_photo(img_table,
        "<b>📊 MACRO FACTOR SIGNAL TABLE</b>\nModel: Institutional Flow Detection"
    )

    # 2️⃣ EXPLANATION
    img_explain = generate_explanation_image(df, run_time)
    send_telegram_photo(img_explain,
        "<b>🧠 HOW TO READ THIS SIGNAL</b>\nPenjelasan sederhana"
    )

    # 3️⃣ TEXT MESSAGE
    send_telegram_message(
        "<b>Catatan:</b> Sistem ini membaca tekanan dana besar (macro flow), "
        "bukan candle pattern atau scalping."
    )

    logging.info("Dual report sent.")

# ==========================================
if __name__ == "__main__":
    run_analysis()
