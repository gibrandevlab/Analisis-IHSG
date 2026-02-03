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

# ======================
# LOGGING
# ======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ======================
# CONFIG
# ======================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Shared
VOL_WINDOW = 20
MIN_AVG_VOL = 200_000        # adjustable liquidity filter for intraday / daily
DEFAULT_PERIOD_YEARS = "2y"

# Institutional (slow, low-noise)
INSTITUTIONAL = {
    "CORR_WINDOW": 60,
    "EWMA_LAMBDA": 0.94,
    "TSTAT_THRESHOLD": 1.5,
    "SCORE_TH": 0.01
}

# Scalping (fast, high sensitivity)
SCALP = {
    "CORR_WINDOW": 20,
    "EWMA_LAMBDA": 0.85,
    "TSTAT_THRESHOLD": 1.0,
    "SCORE_TH": 0.005
}

target_list = [
    "BBCA.JK","BBRI.JK","BMRI.JK","BBNI.JK","TLKM.JK","ASII.JK",
    "MEDC.JK","AKRA.JK","PGAS.JK","ADRO.JK","PTBA.JK","ITMG.JK",
    "UNTR.JK","ANTM.JK","MDKA.JK","BRMS.JK","ICBP.JK","UNVR.JK","AMRT.JK"
]

macro_tickers = {
    "^VIX":"VIX","GC=F":"Gold","CL=F":"Oil",
    "YM=F":"DowFut","IDR=X":"USDIDR","JPY=X":"USDJPY"
}

# ======================
# TELEGRAM HELPERS
# ======================
def send_telegram_photo(photo_buffer, caption):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram not configured; skipping send.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    photo_buffer.seek(0)
    try:
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'HTML'},
                      files={'photo': ('report.jpg', photo_buffer, 'image/jpeg')}, timeout=30)
    except Exception as e:
        logging.warning(f"Failed sending photo: {e}")

def send_telegram_message(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram not configured; skipping send.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': text, 'parse_mode': 'HTML'}, timeout=10)
    except Exception as e:
        logging.warning(f"Failed sending message: {e}")

# ======================
# UTIL
# ======================
def get_market_context():
    tz = pytz.timezone('Asia/Jakarta')
    now = datetime.now(tz)
    return now.strftime('%d %b %Y, %H:%M WIB')

def ewma_weights(n, lam=0.94):
    w = np.array([(1-lam)*(lam**i) for i in range(n)][::-1])
    return w / w.sum()

def rolling_zscore(series, window):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

# ======================
# FACTOR MODEL (PARAMETERIZED)
# ======================
def rolling_factor_model(stock_ret, macro_ret, window_size, lam):
    df = pd.concat([stock_ret, macro_ret], axis=1).dropna()
    if len(df) < window_size:
        return None
    window = df.iloc[-window_size:]
    y = window.iloc[:, 0].values
    X = window.iloc[:, 1:].values
    X = np.column_stack([np.ones(len(X)), X])

    w = ewma_weights(len(y), lam)
    W = np.diag(w)

    # Regularized inversion guard (small ridge)
    XtW = X.T @ W
    try:
        beta = np.linalg.inv(XtW @ X) @ (XtW @ y)
    except np.linalg.LinAlgError:
        # add tiny ridge
        ridge = 1e-6 * np.eye(X.shape[1])
        beta = np.linalg.inv(XtW @ X + ridge) @ (XtW @ y)

    residuals = y - X @ beta
    sigma2 = (w * residuals**2).sum() / max(1, (len(y)-X.shape[1]))
    try:
        var_beta = sigma2 * np.linalg.inv(XtW @ X)
    except np.linalg.LinAlgError:
        ridge = 1e-6 * np.eye(X.shape[1])
        var_beta = sigma2 * np.linalg.inv(XtW @ X + ridge)

    tstats = beta / np.sqrt(np.abs(np.diag(var_beta)) + 1e-12)
    return beta[1:], tstats[1:], window.columns[1:]

# ======================
# IMAGE GENERATORS
# ======================
def generate_table(df, title, run_time):
    fig_height = 2.8 + max(1, len(df))*0.45
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', edges='closed')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    plt.title(title, fontsize=16, weight='bold')
    plt.figtext(0.5, 0.04, f"Generated: {run_time}", ha="center", fontsize=9)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', bbox_inches='tight', dpi=180)
    plt.close(fig)
    return buf

def generate_explanation_image(mode_name, top_rows, run_time):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    header = f"{mode_name} — Summary (Top 3)"
    lines = [header, f"Generated: {run_time}", ""]
    if top_rows is None or len(top_rows) == 0:
        lines.append("No signals found for this mode.")
    else:
        for r in top_rows[:3]:
            # r: [TICKER, SIGNAL, PRICE, CONFIDENCE, DRIVER]
            lines.append(f"{r[0]} | {r[1]} | {r[3]} | Driver: {r[4]}")
    lines += ["", "How to read:", "BUY = pressure naik, SELL = pressure turun, WAIT = no clear flow", "Scalping = high sensitivity (fast)", "Institutional = low noise (slow)"]
    text = "\n".join(lines)
    plt.text(0.02, 0.98, text, fontsize=11, va='top', wrap=True)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', bbox_inches='tight', dpi=180)
    plt.close(fig)
    return buf

# ======================
# SCALPING SCANNER (INTRADAY OPEN=LOW)
# ======================
def calc_vwap(df):
    pv = (df['Close'] * df['Volume']).cumsum()
    volcum = df['Volume'].cumsum().replace(0, np.nan)
    return pv / volcum

def analyze_scalp(stock):
    """
    Returns list [TICKER, SIGNAL, PRICE, CONFIDENCE, DRIVER] or None
    Uses 5m intraday yfinance (period=1d). Prototype-level: yfinance has latency.
    """
    try:
        intr = yf.download(stock, interval="5m", period="1d", progress=False)
        if intr is None or len(intr) < 3:
            return None

        intr = intr.dropna(subset=['Open','High','Low','Close','Volume'])
        intr['VWAP'] = calc_vwap(intr)
        open_price = intr.iloc[0]['Open']
        low_price = intr.iloc[0]['Low']
        first_vol = intr.iloc[0]['Volume']
        avg_vol_20 = intr['Volume'].rolling(20, min_periods=1).mean().iloc[-1]

        # gap relative to previous close (use today's first row Open vs prev day Close)
        # yfinance intraday doesn't include prev day easily; approximate via history 2 days:
        hist = yf.download(stock, period="2d", progress=False, interval="1d")
        prev_close = None
        if hist is not None and 'Close' in hist.columns and len(hist['Close'].dropna()) >= 2:
            prev_close = hist['Close'].dropna().iloc[-2]
        else:
            prev_close = intr.iloc[0]['Open']  # fallback (gap=0)

        gap_pct = 0.0
        try:
            gap_pct = (open_price - prev_close) / (prev_close if prev_close != 0 else open_price) * 100.0
        except Exception:
            gap_pct = 0.0

        cond_open_low = abs(open_price - low_price) <= (open_price * 0.0015)  # practically equal
        cond_vol_spike = first_vol > max(1, avg_vol_20) * 1.5
        cond_vwap_reclaim = intr['Close'].iloc[-1] > intr['VWAP'].iloc[-1]
        cond_gap = abs(gap_pct) < 3.0

        # ATR proxy (5m) for range sanity
        intr['H-L'] = intr['High'] - intr['Low']
        atr5 = intr['H-L'].rolling(6, min_periods=1).mean().iloc[-1]  # last ~30min avg
        range_ok = atr5 > 0

        if cond_open_low and cond_vol_spike and cond_vwap_reclaim and cond_gap and range_ok:
            strength = (intr['Close'].iloc[-1] - open_price) / (open_price if open_price!=0 else 1) * 100
            signal = "BUY" if strength > 0 else "WAIT"
            confidence = min(95, max(10, abs(strength)*10))
            driver = "Intraday Flow"
            return [stock.replace(".JK",""), signal, f"{intr['Close'].iloc[-1]:,.0f}", f"{confidence:.1f}%", driver]
    except Exception as e:
        logging.debug(f"Scalp analyze error {stock}: {e}")
    return None

# ======================
# ANALYZE MODE (COMMON)
# ======================
def analyze_mode_config(stock, price_series, volume_series, macro_ret, config):
    """
    Run rolling factor model with mode-specific config.
    Return [signal, confidence, driver] or None
    """
    stock_ret = price_series.pct_change()
    model = rolling_factor_model(stock_ret, macro_ret, config["CORR_WINDOW"], config["EWMA_LAMBDA"])
    if model is None:
        return None
    betas, tstats, drivers = model
    sig_mask = np.abs(tstats) > config["TSTAT_THRESHOLD"]
    if not sig_mask.any():
        return None
    latest_macro = macro_ret.iloc[-1]
    # ensure indices align: drivers is Index of macro names
    # betas, tstats order corresponds to drivers
    try:
        score = np.sum(betas[sig_mask] * latest_macro[sig_mask])
    except Exception:
        # fallback safer alignment by name
        betas_s = pd.Series(betas, index=drivers)
        relevant = betas_s[betas_s.index.isin(latest_macro.index)]
        score = (relevant * latest_macro[relevant.index]).sum()

    vol_z = rolling_zscore(volume_series, VOL_WINDOW).iloc[-1] if len(volume_series) >= VOL_WINDOW else 0
    if vol_z < 0:
        score *= 0.8

    signal = "WAIT"
    if score > config["SCORE_TH"]:
        signal = "BUY"
    elif score < -config["SCORE_TH"]:
        signal = "SELL"

    confidence = min(95, max(5, np.mean(np.abs(tstats[sig_mask]))*20))
    main_driver = "N/A"
    try:
        main_driver = macro_tickers[drivers[np.argmax(np.abs(betas))]]
    except Exception:
        main_driver = "N/A"

    return [signal, confidence, main_driver]

# ======================
# MAIN ORCHESTRATOR
# ======================
def run_analysis():
    run_time = get_market_context()
    logging.info("Downloading daily market data (macro + targets)...")
    all_tickers = list(macro_tickers.keys()) + target_list
    try:
        data = yf.download(all_tickers, period=DEFAULT_PERIOD_YEARS, progress=False)
    except Exception as e:
        logging.error(f"Failed downloading market data: {e}")
        return

    if 'Close' not in data:
        logging.error("Downloaded data missing Close.")
        return

    close = data['Close'].ffill()
    volume = data['Volume'].ffill()

    macro_symbols = list(macro_tickers.keys())
    macro_ret = close[macro_symbols].pct_change().dropna()
    if macro_ret.empty:
        logging.error("Macro returns empty; abort.")
        return

    scalp_results = []
    inst_results = []

    logging.info("Running per-stock analysis...")
    for stock in target_list:
        try:
            price_series = close[stock].dropna()
            vol_series = volume[stock].dropna()
            if price_series.empty or vol_series.empty:
                continue
            # liquidity filter (daily)
            if vol_series.rolling(20, min_periods=1).mean().iloc[-1] < MIN_AVG_VOL:
                continue

            # Scalping: run intraday scanner (separate, uses yfinance intraday)
            scalp = analyze_scalp(stock)
            if scalp:
                scalp_results.append(scalp)

            # Institutional: slower factor model
            inst = analyze_mode_config(stock, price_series, vol_series, macro_ret, INSTITUTIONAL)
            if inst:
                inst_results.append([stock.replace(".JK",""), inst[0], f"{price_series.iloc[-1]:,.0f}", f"{inst[1]:.1f}%", inst[2]])

        except Exception as e:
            logging.warning(f"Skip {stock}: {e}")

    # Prepare and send outputs
    # 1) Scalping Table + Explanation
    if scalp_results:
        df_scalp = pd.DataFrame(scalp_results, columns=["TICKER","SIGNAL","PRICE","CONFIDENCE","DRIVER"]).sort_values("CONFIDENCE", ascending=False)
        img_scalp_table = generate_table(df_scalp, "⚡ SCALPING MODE SIGNALS (Open=Low Scanner)", run_time)
        send_telegram_photo(img_scalp_table, "<b>⚡ SCALPING MODE SIGNALS</b>\nHigh sensitivity intraday scanner (open=low)")
        img_scalp_expl = generate_explanation_image("SCALPING MODE", scalp_results, run_time)
        send_telegram_photo(img_scalp_expl, "<b>🧾 SCALPING - HOW TO READ</b>\nSimple explanation")
    else:
        logging.info("No scalp signals found.")
        send_telegram_message(f"⚡ SCALPING MODE: No setups found at {run_time}")

    # 2) Institutional Table + Explanation
    if inst_results:
        df_inst = pd.DataFrame(inst_results, columns=["TICKER","SIGNAL","PRICE","CONFIDENCE","DRIVER"]).sort_values("CONFIDENCE", ascending=False)
        img_inst_table = generate_table(df_inst, "🏦 INSTITUTIONAL MODE SIGNALS (Macro Flow)", run_time)
        send_telegram_photo(img_inst_table, "<b>🏦 INSTITUTIONAL MODE SIGNALS</b>\nLow-noise macro flow detection")
        inst_top = [list(x) for x in df_inst.head(3).itertuples(index=False, name=None)]
        img_inst_expl = generate_explanation_image("INSTITUTIONAL MODE", inst_top, run_time)
        send_telegram_photo(img_inst_expl, "<b>🧾 INSTITUTIONAL - HOW TO READ</b>\nSimple explanation")
    else:
        logging.info("No institutional signals found.")
        send_telegram_message(f"🏦 INSTITUTIONAL MODE: No signals found at {run_time}")

    # 3) final short reminder message
    send_telegram_message(
        f"<b>Report generated:</b> {run_time}\n"
        "Note: Scalping signals are intraday prototypes (yfinance latency). Institutional signals are macro-flow indicators (not for scalping)."
    )

    logging.info("All reports processed.")

# ======================
# RUN
# ======================
if __name__ == "__main__":
    run_analysis()
