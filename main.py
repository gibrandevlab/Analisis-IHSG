# ====================== IMPORT ======================
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

# ====================== LOGGING ======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ====================== CONFIG ======================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

VOL_WINDOW = 20
MIN_AVG_VOL = 200_000
DEFAULT_PERIOD_YEARS = "2y"

INSTITUTIONAL = {"CORR_WINDOW":60,"EWMA_LAMBDA":0.94,"TSTAT_THRESHOLD":1.5,"SCORE_TH":0.01}
SCALP = {"CORR_WINDOW":20,"EWMA_LAMBDA":0.85,"TSTAT_THRESHOLD":1.0,"SCORE_TH":0.005}

target_list = ["BBCA.JK","BBRI.JK","BMRI.JK","BBNI.JK","TLKM.JK","ASII.JK",
"MEDC.JK","AKRA.JK","PGAS.JK","ADRO.JK","PTBA.JK","ITMG.JK",
"UNTR.JK","ANTM.JK","MDKA.JK","BRMS.JK","ICBP.JK","UNVR.JK","AMRT.JK"]

macro_tickers = {"^VIX":"VIX","GC=F":"Gold","CL=F":"Oil","YM=F":"DowFut","IDR=X":"USDIDR","JPY=X":"USDJPY"}

# ====================== TELEGRAM ======================
def send_telegram_photo(photo_buffer, caption):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    photo_buffer.seek(0)
    requests.post(url,data={'chat_id': TELEGRAM_CHAT_ID,'caption': caption,'parse_mode':'HTML'},
                  files={'photo':('report.jpg',photo_buffer,'image/jpeg')},timeout=30)

def send_telegram_message(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url,data={'chat_id': TELEGRAM_CHAT_ID,'text': text,'parse_mode':'HTML'},timeout=10)

# ====================== UTIL ======================
def get_market_context():
    tz = pytz.timezone('Asia/Jakarta')
    return datetime.now(tz).strftime('%d %b %Y, %H:%M WIB')

def ewma_weights(n, lam=0.94):
    w = np.array([(1-lam)*(lam**i) for i in range(n)][::-1])
    return w/w.sum()

def rolling_zscore(series, window):
    return (series-series.rolling(window).mean())/series.rolling(window).std()

def get_intraday_macro():
    try:
        macros = yf.download(list(macro_tickers.keys()),interval="5m",period="1d",progress=False)
        if macros is None or 'Close' not in macros: return None
        close = macros['Close'].dropna(how='all')
        return close.pct_change().dropna()
    except Exception as e:
        logging.warning(f"Macro intraday fail: {e}")
        return None

# ====================== FACTOR MODEL (UNCHANGED) ======================
def rolling_factor_model(stock_ret, macro_ret, window_size, lam):
    df = pd.concat([stock_ret, macro_ret], axis=1).dropna()
    if len(df) < window_size: return None
    window = df.iloc[-window_size:]
    y = window.iloc[:,0].values
    X = np.column_stack([np.ones(len(window)), window.iloc[:,1:].values])
    w = ewma_weights(len(y),lam)
    W = np.diag(w)
    XtW = X.T @ W
    beta = np.linalg.pinv(XtW @ X) @ (XtW @ y)
    residuals = y - X@beta
    sigma2 = (w*residuals**2).sum()/max(1,len(y)-X.shape[1])
    var_beta = sigma2*np.linalg.pinv(XtW@X)
    tstats = beta/np.sqrt(np.abs(np.diag(var_beta))+1e-12)
    return beta[1:],tstats[1:],window.columns[1:]

# ====================== SCALPING FIX ======================
def calc_vwap(df):
    pv = (df['Close']*df['Volume']).cumsum()
    return pv/df['Volume'].cumsum().replace(0,np.nan)

def analyze_scalp(stock, macro_intraday=None):
    try:
        intr = yf.download(stock,interval="5m",period="1d",progress=False)
        if intr is None or len(intr)<3: return None

        intr = intr.dropna(subset=['Open','High','Low','Close','Volume'])
        intr['VWAP']=calc_vwap(intr)

        open_price = intr.iloc[0]['Open']
        low_price = intr.iloc[0]['Low']
        first_vol = intr.iloc[0]['Volume']
        avg_vol_20 = intr['Volume'].rolling(20,min_periods=1).mean().iloc[-1]

        hist = yf.download(stock,period="2d",interval="1d",progress=False)
        prev_close = hist['Close'].dropna().iloc[-2] if len(hist['Close'].dropna())>=2 else open_price
        gap_pct = (open_price-prev_close)/prev_close*100

        cond_open_low = abs(open_price-low_price)<=open_price*0.0015
        cond_vol_spike = first_vol>avg_vol_20*1.5
        cond_vwap_reclaim = intr['Close'].iloc[-1]>intr['VWAP'].iloc[-1]
        cond_gap = abs(gap_pct)<3.0
        atr5 = (intr['High']-intr['Low']).rolling(6).mean().iloc[-1]
        range_ok = atr5>0

        macro_bias=0.0
        if macro_intraday is not None and not macro_intraday.empty:
            m = macro_intraday.iloc[-1]
            macro_bias += m.get("YM=F",0)*2.0
            macro_bias += m.get("CL=F",0)*1.2
            macro_bias -= m.get("GC=F",0)*1.0
            macro_bias -= m.get("^VIX",0)*2.5
            macro_bias -= m.get("IDR=X",0)*0.5

        if cond_open_low and cond_vol_spike and cond_vwap_reclaim and cond_gap and range_ok:
            strength = (intr['Close'].iloc[-1]-open_price)/open_price*100
            combined_score = strength + macro_bias*100
            signal="WAIT"
            if combined_score>0.15: signal="BUY"
            elif combined_score<-0.15: signal="SELL"
            confidence=min(95,max(10,(abs(strength)*8+abs(macro_bias)*200)))
            return [stock.replace(".JK",""),signal,f"{intr['Close'].iloc[-1]:,.0f}",f"{confidence:.1f}%","Flow+Macro"]

    except Exception as e:
        logging.debug(f"Scalp err {stock}:{e}")
    return None

# ====================== MAIN ======================
def run_analysis():
    run_time=get_market_context()
    logging.info("Downloading intraday macro flow...")
    macro_intraday=get_intraday_macro()

    scalp_results=[]
    for stock in target_list:
        scalp=analyze_scalp(stock,macro_intraday)
        if scalp: scalp_results.append(scalp)

    if scalp_results:
        df=pd.DataFrame(scalp_results,columns=["TICKER","SIGNAL","PRICE","CONFIDENCE","DRIVER"]).sort_values("CONFIDENCE",ascending=False)
        send_telegram_message(f"⚡ SCALPING SIGNALS {run_time}\n\n{df.to_string(index=False)}")
    else:
        send_telegram_message(f"⚡ SCALPING: No setups {run_time}")

if __name__=="__main__":
    run_analysis()
