import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import io

# ==========================================
# 1. CONFIGURATION
# ==========================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "GANTI_TOKEN_DISINI")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "GANTI_CHAT_ID_DISINI")

# Matplotlib Backend (Agar jalan di server tanpa monitor)
import matplotlib
matplotlib.use('Agg')

target_list = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", "ASII.JK",
    "MEDC.JK", "AKRA.JK", "PGAS.JK", "ADRO.JK", "PTBA.JK", "ITMG.JK",
    "UNTR.JK", "ANTM.JK", "MDKA.JK", "BRMS.JK", "ICBP.JK", "UNVR.JK", "AMRT.JK"
]

macro_tickers = {
    "^VIX": "VIX", "GC=F": "Gold", "CL=F": "Oil",
    "YM=F": "DowFut", "IDR=X": "USDIDR", "JPY=X": "USDJPY"
}

CORR_WINDOW = 60      
VOL_WINDOW = 20       
PENALTY_FACTOR = 0.5  

# ==========================================
# 2. TELEGRAM SENDER (IMAGE SUPPORT)
# ==========================================
def send_telegram_photo(photo_buffer, caption):
    if not TELEGRAM_TOKEN or "GANTI" in TELEGRAM_TOKEN:
        print("⚠️ Token Telegram belum di-setting!")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    
    # Reset pointer buffer ke awal file
    photo_buffer.seek(0)
    
    files = {'photo': ('report.jpg', photo_buffer, 'image/jpeg')}
    data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'HTML'}
    
    try:
        response = requests.post(url, data=data, files=files, timeout=30)
        response.raise_for_status()
        print("✅ Telegram Photo sent successfully.")
    except Exception as e:
        print(f"❌ Failed to send Telegram: {e}")

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def get_wilders_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_vol_z_score(series, window=20):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def generate_hd_table_image(df, title_date):
    """Fungsi untuk menggambar tabel HD menggunakan Matplotlib"""
    
    # 1. Setup Figure & Colors
    # Tinggi gambar dinamis sesuai jumlah baris
    fig_height = 2 + (len(df) * 0.5) 
    fig, ax = plt.subplots(figsize=(10, fig_height)) # Lebar 10 inch
    
    # Hapus Axis (Garis grafik)
    ax.axis('off')
    ax.axis('tight')
    
    # 2. Prepare Data for Table
    # Kita buat kolom baru untuk display agar rapi
    display_data = []
    
    # Header Columns
    col_labels = ["TICKER", "SIGNAL", "PRICE", "CONF", "VOL STAT", "DRIVER"]
    
    for _, row in df.iterrows():
        # Tambahkan tanda seru visual di kolom Vol Stat
        vol_stat = "LOW (!)" if row['vol_low'] else "NORMAL"
        
        display_data.append([
            row['ticker'],
            row['signal_text'], # e.g. BUY 🚀
            f"{row['price']:,.0f}",
            f"{row['conf']:.0f}%",
            vol_stat,
            row['driver']
        ])

    # 3. Draw Table
    # cellLoc='center' agar teks di tengah
    table = ax.table(cellText=display_data, colLabels=col_labels, 
                     loc='center', cellLoc='center', edges='open')
    
    # 4. Styling (The "HD" Part)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.0) # Scaling lebar dan tinggi sel
    
    # Header Style
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('white') # Hapus garis border default hitam
        cell.set_linewidth(0.5)
        
        # Header Row
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50') # Dark Blue Header
            cell.set_height(0.1)
        else:
            # Alternating Row Colors (Zebra Striping)
            if row % 2 == 0:
                cell.set_facecolor('#f2f2f2') # Light Grey
            else:
                cell.set_facecolor('white')
            
            # Text Coloring Logic based on Signal
            signal_val = display_data[row-1][1] # Ambil text signal
            
            # Warnai teks kolom SIGNAL & CONF
            if col in [1, 3]: 
                if "BUY" in signal_val or "UP" in signal_val:
                    cell.set_text_props(color='#27ae60', weight='bold') # Green
                elif "SELL" in signal_val or "DOWN" in signal_val:
                    cell.set_text_props(color='#c0392b', weight='bold') # Red
            
            # Warnai kolom VOL jika Low
            if col == 4 and "LOW" in display_data[row-1][4]:
                cell.set_text_props(color='#e67e22', weight='bold') # Orange

            # Padding kiri untuk Ticker
            if col == 0:
                cell.set_text_props(ha='left')
                
    # 5. Title
    plt.title(f"MARKET QUANT SIGNAL PRO\nDate: {title_date}", 
              fontsize=16, weight='bold', pad=20, color='#34495e')
    
    # 6. Footer/Watermark
    plt.figtext(0.5, 0.02, "Engine: Robust Macro + Volume Penalty Model", 
                ha="center", fontsize=8, color='gray')

    # 7. Save to Buffer
    buf = io.BytesIO()
    # DPI 200 agar HD (High Definition) dan tajam
    plt.savefig(buf, format='jpg', bbox_inches='tight', dpi=200, facecolor='white')
    plt.close(fig)
    return buf

# ==========================================
# 4. MAIN LOGIC
# ==========================================
def run_analysis():
    print("⏳ Downloading data & crunching numbers...")
    
    all_tickers = list(macro_tickers.keys()) + target_list
    try:
        data_full = yf.download(all_tickers, period="2y", progress=False)
        if isinstance(data_full.columns, pd.MultiIndex):
            data_close = data_full.xs('Close', level=0, axis=1).ffill()
            data_volume = data_full.xs('Volume', level=0, axis=1).ffill()
        else:
            data_close = data_full['Close'].ffill()
            data_volume = data_full['Volume'].ffill()
    except Exception as e:
        print(f"❌ Error Download: {e}")
        return

    # --- MACRO PROCESSING ---
    macro_returns = data_close[list(macro_tickers.keys())].pct_change().dropna()
    scaler = RobustScaler()
    macro_scaled = pd.DataFrame(scaler.fit_transform(macro_returns),
                                index=macro_returns.index, columns=macro_returns.columns)

    final_results = []

    for stock in target_list:
        try:
            stock_price = data_close[stock].dropna()
            stock_vol = data_volume[stock].dropna()
            stock_ret = stock_price.pct_change()
            
            df_ind = pd.concat([macro_scaled, stock_ret.shift(-1)], axis=1).dropna()
            if len(df_ind) < CORR_WINDOW: continue
            
            stock_corr = df_ind.iloc[-CORR_WINDOW:].corr()[stock].drop(stock)
            signals = macro_scaled.iloc[-1] * stock_corr
            raw_score = signals.sum()
            
            abs_sum = signals.abs().sum()
            agreement = (abs(raw_score) / abs_sum) if abs_sum != 0 else 0
            
            vol_z = calculate_vol_z_score(stock_vol, VOL_WINDOW).iloc[-1]
            
            final_score = raw_score
            has_penalty = False
            if raw_score > 0.1 and vol_z < 0:
                final_score *= PENALTY_FACTOR
                has_penalty = True

            signal_text = "WAIT"
            if final_score > 0.35 and agreement > 0.6: signal_text = "BUY 🚀"
            elif final_score > 0.1: signal_text = "UP 📈"
            elif final_score < -0.35 and agreement > 0.6: signal_text = "SELL 🔻"
            elif final_score < -0.1: signal_text = "DOWN 📉"

            confidence = (stock_corr.abs().max() * 100) * (PENALTY_FACTOR if has_penalty else 1.0)
            driver = macro_tickers.get(stock_corr.abs().idxmax(), "N/A")

            final_results.append({
                "ticker": stock.replace(".JK", ""),
                "signal_text": signal_text,
                "price": stock_price.iloc[-1],
                "conf": confidence,
                "score": final_score,
                "vol_low": vol_z < 0,
                "driver": driver
            })
        except: continue

    # ==========================================
    # 5. GENERATE IMAGE & SEND
    # ==========================================
    if not final_results:
        print("⚠️ No results generated.")
        return

    # Sort
    sorted_results = sorted(final_results, key=lambda x: x['score'], reverse=True)
    df_results = pd.DataFrame(sorted_results)
    
    # Generate Timestamp
    now_str = pd.Timestamp.now().strftime('%d %b %Y, %H:%M WIB')
    
    print("🎨 Generating HD Image...")
    # Create Image Buffer
    image_buffer = generate_hd_table_image(df_results, now_str)
    
    # Caption Text for Telegram
    caption_text = (
        f"<b>📊 MARKET SIGNAL REPORT</b>\n"
        f"📅 <i>{now_str}</i>\n\n"
        f"💡 <b>Legend:</b>\n"
        f"• <b>CONF:</b> Tingkat keyakinan berdasarkan korelasi makro.\n"
        f"• <b>LOW (!):</b> Volume sedang sepi, sinyal mungkin false alarm (Penalty applied)."
    )
    
    print("🚀 Sending to Telegram...")
    send_telegram_photo(image_buffer, caption_text)

if __name__ == "__main__":
    run_analysis() 