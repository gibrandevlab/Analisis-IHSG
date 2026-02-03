import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import io
import matplotlib

# Matplotlib Backend (Agar jalan di server tanpa monitor/GUI)
matplotlib.use('Agg')

# ==========================================
# 1. CONFIGURATION
# ==========================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

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
# 2. TELEGRAM SENDER
# ==========================================
def send_telegram_photo(photo_buffer, caption):
    if not TELEGRAM_TOKEN or "YOUR_BOT" in TELEGRAM_TOKEN:
        print("⚠️ Token Telegram belum di-setting!")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
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
def calculate_vol_z_score(series, window=20):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def generate_hd_table_image(df, title_date):
    """Fungsi menggambar tabel dengan border, header, dan styling profesional"""
    
    # Setup Figure (Tinggi dinamis berdasarkan jumlah baris)
    fig_height = 2.5 + (len(df) * 0.45) 
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    ax.axis('off')
    ax.axis('tight')
    
    # Kolom header untuk tabel
    col_labels = ["TICKER", "SIGNAL", "LAST PRICE", "CONFIDENCE", "VOL STATUS", "MACRO DRIVER"]
    
    display_data = []
    for _, row in df.iterrows():
        vol_stat = "LOW (!)" if row['vol_low'] else "NORMAL"
        display_data.append([
            row['ticker'],
            row['signal_text'],
            f"{row['price']:,.0f}",
            f"{row['conf']:.1f}%",
            vol_stat,
            row['driver']
        ])

    # Mengaktifkan 'edges=all' untuk menampilkan garis border di semua sel
    table = ax.table(cellText=display_data, 
                    colLabels=col_labels, 
                    loc='center', 
                    cellLoc='center', 
                    edges='all') 
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0) # Mengatur kerenggangan baris (tinggi sel)
    
    # Styling setiap sel (Header & Body)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#bdc3c7') # Warna border abu-abu elegan
        cell.set_linewidth(0.7)
        
        # Style Header
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50') # Warna Biru Gelap
        else:
            # Zebra Striping (Warna selang-seling)
            if row % 2 == 0:
                cell.set_facecolor('#f8f9fa')
            else:
                cell.set_facecolor('white')
            
            # Logika pewarnaan teks berdasarkan isi data
            signal_val = display_data[row-1][1]
            vol_val = display_data[row-1][4]

            # Warna Sinyal (Kolom 1)
            if col == 1:
                if "BUY" in signal_val or "UP" in signal_val:
                    cell.set_text_props(color='#27ae60', weight='bold') # Hijau
                elif "SELL" in signal_val or "DOWN" in signal_val:
                    cell.set_text_props(color='#e74c3c', weight='bold') # Merah
            
            # Highlight Volume Low (Kolom 4)
            if col == 4 and "LOW" in vol_val:
                cell.set_text_props(color='#d35400', weight='bold') # Oranye

    # Title & Metadata
    plt.title(f"MARKET QUANT SIGNAL PRO\nQuantitative Macro & Volume Analysis", 
              fontsize=18, weight='bold', pad=35, color='#2c3e50')
    
    plt.figtext(0.5, 0.90, f"Analysis Date: {title_date}", ha="center", fontsize=11, color='#7f8c8d')
    
    # Footer
    plt.figtext(0.5, 0.04, "Engine: Robust Macro Model + Volume Penalty Factor | Data: Yahoo Finance", 
                ha="center", fontsize=9, color='#95a5a6', style='italic')

    # Save to Buffer dengan DPI tinggi agar tajam
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', bbox_inches='tight', dpi=200, facecolor='white')
    plt.close(fig)
    return buf

# ==========================================
# 4. MAIN LOGIC
# ==========================================
def run_analysis():
    print("⏳ Processing market data...")
    
    all_tickers = list(macro_tickers.keys()) + target_list
    try:
        data_full = yf.download(all_tickers, period="2y", progress=False)
        # Menangani MultiIndex dari yfinance
        if isinstance(data_full.columns, pd.MultiIndex):
            data_close = data_full.xs('Close', level=0, axis=1).ffill()
            data_volume = data_full.xs('Volume', level=0, axis=1).ffill()
        else:
            data_close = data_full['Close'].ffill()
            data_volume = data_full['Volume'].ffill()
    except Exception as e:
        print(f"❌ Error Download: {e}")
        return

    # Macro Processing
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
            
            # Correlation Analysis
            stock_corr = df_ind.iloc[-CORR_WINDOW:].corr()[stock].drop(stock)
            signals = macro_scaled.iloc[-1] * stock_corr
            raw_score = signals.sum()
            
            abs_sum = signals.abs().sum()
            agreement = (abs(raw_score) / abs_sum) if abs_sum != 0 else 0
            
            # Volume Penalty Logic
            vol_z = calculate_vol_z_score(stock_vol, VOL_WINDOW).iloc[-1]
            final_score = raw_score
            has_penalty = False
            
            if raw_score > 0.1 and vol_z < 0:
                final_score *= PENALTY_FACTOR
                has_penalty = True

            # Signal Determination
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

    if not final_results:
        print("⚠️ No results generated.")
        return

    # Sort results by score (Bullish to Bearish)
    sorted_results = sorted(final_results, key=lambda x: x['score'], reverse=True)
    df_results = pd.DataFrame(sorted_results)
    
    # Generate Timestamp
    now_str = pd.Timestamp.now().strftime('%d %b %Y, %H:%M WIB')
    
    print("🎨 Generating HD Table Image...")
    image_buffer = generate_hd_table_image(df_results, now_str)
    
    # Caption Telegram
    caption_text = (
        f"<b>📊 MARKET SIGNAL REPORT</b>\n"
        f"📅 <i>{now_str}</i>\n\n"
        f"💡 <b>Keterangan Kolom:</b>\n"
        f"• <b>CONF:</b> Keyakinan sinyal (makro korelasi).\n"
        f"• <b>VOL STATUS:</b> Jika LOW (!), sinyal lemah karena volume sepi.\n"
        f"• <b>DRIVER:</b> Aset luar yang paling memengaruhi ticker ini."
    )
    
    print("🚀 Sending to Telegram...")
    send_telegram_photo(image_buffer, caption_text)

if __name__ == "__main__":
    run_analysis()