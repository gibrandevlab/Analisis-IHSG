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
        print("⚠️ Telegram credentials not found. Skipping message.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    # Telegram has a limit of 4096 chars per message. 
    # If the message is potentially long, we might need to split it, 
    # but for this summary tablest it should be fine.
    # We wrap in <pre> tag for monospaced font preservation.
    # IMPORTANT: Escape HTML characters to avoid 'Bad Request: can't parse entities'
    safe_message = html.escape(message)
    
    payload = {
        'chat_id': chat_id,
        'text': f"<pre>{safe_message}</pre>",
        'parse_mode': 'HTML'
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("✅ Telegram message sent successfully.")
    except Exception as e:
        print(f"❌ Failed to send Telegram message: {e}")
        try:
            print(f"Server response: {response.text}")
        except:
            pass

# Capture stdout to send as message later
class StringCapture:
    def __init__(self):
        self._stdout = sys.stdout
        self._stringio = io.StringIO()

    def __enter__(self):
        sys.stdout = self._stringio
        return self._stringio

    def __exit__(self, *args):
        sys.stdout = self._stdout

# ==========================================
# MAIN LOGIC
# ==========================================
def run_analysis():
    # ==========================================
    # 1. SETUP & CONFIGURATION
    # ==========================================
    target_list = [
        "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", "ASII.JK",
        "MEDC.JK", "AKRA.JK", "PGAS.JK", "ADRO.JK", "PTBA.JK", "ITMG.JK",
        "UNTR.JK", "ANTM.JK", "MDKA.JK", "BRMS.JK", "ICBP.JK", "UNVR.JK", "AMRT.JK"
    ]

    macro_tickers = {
        "^VIX": "VIX (Panic)", 
        "GC=F": "Gold", 
        "CL=F": "Oil", 
        "^DJI": "Dow Jones",
        "IDR=X": "USD/IDR",
        "JPY=X": "USD/JPY"
    }

    # --- PARAMETER UTAMA ---
    CORR_WINDOW = 60      # Jendela Rolling Correlation (60 hari bursa ~ 3 bulan)
    VOL_WINDOW = 20       # Jendela rata-rata Volume
    PENALTY_FACTOR = 0.5  # Diskon skor (50%) jika Volume lemah

    print("⚙️  INITIALIZING SYSTEM V4.1 (Fixed Alignment & Robust Stat)...")

    # ==========================================
    # 2. DATA ACQUISITION
    # ==========================================
    all_tickers = list(macro_tickers.keys()) + target_list
    try:
        # Ambil data 2 tahun untuk buffer perhitungan indikator
        data_full = yf.download(all_tickers, period="2y", progress=False)
        
        # Handle MultiIndex (Format yfinance terbaru sering berubah)
        if isinstance(data_full.columns, pd.MultiIndex):
            try:
                data_close = data_full.xs('Close', level=0, axis=1).ffill()
                data_volume = data_full.xs('Volume', level=0, axis=1).ffill()
            except KeyError:
                 # Fallback for some yfinance versions simply returning 'Adj Close' or similar
                data_close = data_full['Adj Close'].ffill() if 'Adj Close' in data_full else data_full['Close'].ffill()
                data_volume = data_full['Volume'].ffill()
        else:
            data_close = data_full['Close'].ffill()
            data_volume = data_full['Volume'].ffill()
            
    except Exception as e:
        print(f"❌ Error Download Data: {e}")
        return

    # ==========================================
    # 3. HELPER FUNCTIONS
    # ==========================================
    def get_wilders_rsi(series, period=14):
        """Menghitung RSI tanpa look-ahead bias."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_vol_z_score(series, window=20):
        """
        Menghitung Z-Score Volume.
        Positif = Volume di atas rata-rata.
        Negatif = Volume di bawah rata-rata (Sepi).
        """
        roll_mean = series.rolling(window).mean()
        roll_std = series.rolling(window).std()
        # Z = (Current - Mean) / StdDev
        z_score = (series - roll_mean) / roll_std
        return z_score

    # ==========================================
    # 4. MACRO DATA PRE-PROCESSING (ROBUST)
    # ==========================================
    # Filter only available columns
    available_macro = [t for t in macro_tickers.keys() if t in data_close.columns]
    macro_returns = data_close[available_macro].pct_change().dropna()

    if macro_returns.empty:
        print("❌ No macro data available.")
        return

    # Gunakan RobustScaler (Median & IQR) agar outlier ekstrem tidak merusak skala
    scaler = RobustScaler()
    macro_scaled = pd.DataFrame(
        scaler.fit_transform(macro_returns),
        index=macro_returns.index,
        columns=macro_returns.columns
    )

    # ==========================================
    # 5. CORE ENGINE: ANALYSIS LOOP
    # ==========================================
    final_report = []

    for stock in target_list:
        if stock not in data_close.columns:
            continue
            
        try:
            # A. Siapkan Data Saham
            stock_price = data_close[stock].dropna()
            stock_vol = data_volume[stock].dropna()
            stock_ret = stock_price.pct_change()
            
            # B. Sync Timezone (Lead-Lag Analysis)
            # Shift return saham mundur 1 hari (T-1) agar sejajar dengan Macro hari ini (T)
            # Tujuannya: Menggunakan Macro hari ini untuk memprediksi Return Besok.
            # We align using the intersection of indices to avoid mismatch errors
            common_index = macro_scaled.index.intersection(stock_ret.index)
            
            df_individual = pd.concat([macro_scaled.loc[common_index], stock_ret.loc[common_index].shift(-1)], axis=1).dropna()
            
            # --- LOGIC: ROLLING CORRELATION ---
            # Hanya ambil slice data 60 hari terakhir untuk korelasi
            recent_regime_data = df_individual.iloc[-CORR_WINDOW:]
            
            # Skip jika data tidak cukup
            if len(recent_regime_data) < CORR_WINDOW:
                continue
                
            corr_matrix = recent_regime_data.corr()
            
            # The column for stock return in df_individual will be named the same as 'stock' or have a suffix.
            # However, since we concatenated, it might still just be the stock ticker.
            # If duplicates exist (e.g. stock is also in macro), we need to be careful.
            # But here targets are JK stocks, macro are global, so no overlap likely.
            
            # Ambil korelasi saham target terhadap makro
            # corr_matrix columns include macro tickers AND the stock ticker
            if stock not in corr_matrix.columns:
                 # It might be at the end or renamed if collision. 
                 # Given the logic, it's safer to access by index or check logic.
                 # Let's assume standard behavior.
                 continue

            stock_corr = corr_matrix[stock].drop(stock, errors='ignore') 
            
            # C. Hitung Sinyal (Signal Agreement)
            latest_macro_moves = macro_scaled.iloc[-1] # Data Macro Hari Ini
            
            # Ensure indices match for multiplication
            common_factors = latest_macro_moves.index.intersection(stock_corr.index)
            individual_signals = latest_macro_moves[common_factors] * stock_corr[common_factors]
            
            raw_score = individual_signals.sum()
            abs_sum = individual_signals.abs().sum()
            
            # Agreement Ratio: Seberapa kompak indikator makro?
            agreement_ratio = (abs(raw_score) / abs_sum) if abs_sum != 0 else 0
            
            # D. Indikator Teknikal (Snapshot hari ini)
            rsi = get_wilders_rsi(stock_price).iloc[-1]
            price_now = stock_price.iloc[-1]
            
            # --- LOGIC: VOLUME PENALTY ---
            vol_z = calculate_vol_z_score(stock_vol, VOL_WINDOW).iloc[-1]
            
            final_score = raw_score
            penalty_text = ""
            
            # Jika Sinyal BUY (>0.1) tapi Volume Sepi (Z < 0), diskon skornya
            if raw_score > 0.1 and vol_z < 0:
                final_score = raw_score * PENALTY_FACTOR
                penalty_text = " (Vol Penalty)"
            
            # E. Tentukan Output
            if not stock_corr.empty:
                strongest_idx = stock_corr.abs().idxmax()
                confidence_pct = (stock_corr.abs().max() * 100) * (1.0 if penalty_text == "" else PENALTY_FACTOR)
            else:
                strongest_idx = "-"
                confidence_pct = 0

            # Symbols for telegram (colors aren't supported in standard text, using emoji)
            pred = "⚪ Neutral"
            
            # Threshold Penentuan Sinyal
            if final_score > 0.35 and agreement_ratio > 0.6:
                pred = "🚀 BUY SIGNAL" 
            elif final_score > 0.1:
                pred = "📈 Slight Uptick"
            elif final_score < -0.35 and agreement_ratio > 0.6:
                pred = "🔻 SELL SIGNAL" 
            elif final_score < -0.1:
                pred = "📉 Slight Drop"

            final_report.append({
                "Saham": stock, 
                "Pred": pred + penalty_text, 
                "Conf": confidence_pct,
                "Driver": macro_tickers.get(strongest_idx, strongest_idx), 
                "Kompak": agreement_ratio * 100,
                "RSI": rsi, 
                "Vol_Z": vol_z,
                "Harga": price_now, 
                "Score": final_score
            })
        except Exception as e:
            # print(f"Skipping {stock}: {e}")
            continue

    # ==========================================
    # 6. OUTPUT (ALIGNED FORMATTING)
    # ==========================================
    if not final_report:
        print("❌ Tidak ada data yang berhasil diolah.")
        return

    df_final = pd.DataFrame(final_report).sort_values(by="Score", ascending=False)

    # Definisi Lebar Kolom (Fixed Width for text alignment)
    # We reduce widths slightly for mobile screens if needed, but standard fixed width is fine for <pre>
    W_SAHAM = 9
    W_PRED = 20 # Reduced from 28 to fit better
    W_CONF = 6
    W_KOMPAK = 8
    W_DRIVER = 12
    W_RSI = 5
    W_VOL = 6
    W_HARGA = 9

    print(f"\n{'SAHAM':<{W_SAHAM}} | {'PREDIKSI':<{W_PRED}} | {'CONF':<{W_CONF}} | {'KOMPAK':<{W_KOMPAK}} | {'DRIVER':<{W_DRIVER}} | {'RSI':<{W_RSI}} | {'VOL Z':<{W_VOL}} | {'HARGA':<{W_HARGA}}")
    print("-" * 105) # Adjusted length

    for _, row in df_final.iterrows():
        # 1. FORMAT TEXT POLOS
        txt_saham = f"{row['Saham']:<{W_SAHAM}}"
        txt_pred  = f"{row['Pred']:<{W_PRED}}"
        txt_conf  = f"{row['Conf']:>{W_CONF-1}.0f}%" 
        
        # Icon Kompak
        k_icon = "💎" if row['Kompak'] > 80 else "⚠️" if row['Kompak'] < 40 else " "
        txt_kompak_raw = f"{row['Kompak']:>4.0f}%{k_icon}"
        txt_kompak = f"{txt_kompak_raw:<{W_KOMPAK}}" 

        # Truncate driver if too long
        d_val = row['Driver']
        if len(d_val) > W_DRIVER: d_val = d_val[:W_DRIVER-1] + "."
        txt_driver = f"{d_val:<{W_DRIVER}}"
        
        txt_rsi    = f"{row['RSI']:>{W_RSI}.1f}"
        txt_vol    = f"{row['Vol_Z']:>{W_VOL}.1f}" # Reduced decimal
        txt_harga  = f"{row['Harga']:>{W_HARGA},.0f}"

        # 3. PRINT BARIS
        print(f"{txt_saham} | {txt_pred} | {txt_conf} | {txt_kompak} | {txt_driver} | {txt_rsi} | {txt_vol} | {txt_harga}")

    print("-" * 105)
    print("💡 INFO: Rolling Correlation 60 days.")
    print("💡 INFO: Vol Z < 0 means low volume.")

if __name__ == "__main__":
    # Capture output
    output_capture = StringCapture()
    with output_capture as out:
        try:
            run_analysis()
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
    
    output_str = out.getvalue()
    
    # Print to console for Actions logs
    sys.stdout.write(output_str) # Write to actual stdout
    
    # Send to Telegram
    send_telegram_message(output_str)
