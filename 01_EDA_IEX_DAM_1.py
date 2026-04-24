# ============================================================
#  PROJECT 2 — IEX DAM Price Forecasting Tool
#  Phase 1: Exploratory Data Analysis (EDA)
#  Author : Duvvada Naveen Kumar
#  Org    : MP Power Management Co. Ltd.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Plot style ───────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#c9d1d9',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#c9d1d9',
    'grid.color':       '#21262d',
    'grid.linestyle':   '--',
    'grid.alpha':       0.6,
    'font.family':      'monospace',
})

ACCENT   = '#58a6ff'   # blue
ACCENT2  = '#f78166'   # orange-red  (sell / high)
ACCENT3  = '#3fb950'   # green       (low price)
ACCENT4  = '#d2a8ff'   # purple      (evening peak)

# ============================================================
# STEP 1 — LOAD & MERGE RAW IEX FILES
# ============================================================
# ► Put all downloaded IEX Excel files in:
#     Project2_IEX_Price_Forecasting/data/raw/
#
# ► Each file must have columns (IEX default names):
#     Date | Block No | MCP (Rs/MWh) | MCV (MW) |
#     Purchase Bid (MW) | Sell Bid (MW)
#
# ► Adjust RAW_FOLDER path before running.

import os, glob

RAW_FOLDER = r"data/raw"          # ← change to your actual path

def _detect_header_row(path: str, max_rows: int = 12) -> int:
    """Find the Excel row that contains the real column headers."""
    preview = pd.read_excel(path, engine='openpyxl', header=None, nrows=max_rows)
    for idx, row in preview.iterrows():
        values = [str(v).strip().lower() for v in row.tolist() if pd.notna(v)]
        joined = " | ".join(values)
        if ('date' in joined and 'mcp' in joined) or \
           ('mcp' in joined and 'block' in joined) or \
           ('mcp' in joined and 'month' in joined):
            return idx
    return 0


def load_iex_files(folder: str) -> pd.DataFrame:
    """Load & merge all IEX DAM Excel files from a folder."""
    all_files = glob.glob(os.path.join(folder, "*.xlsx"))
    if not all_files:
        print("⚠️  No Excel files found — generating SYNTHETIC data for demo.")
        return generate_synthetic_data()

    frames = []
    for f in all_files:
        try:
            hdr_row = _detect_header_row(f)
            df = pd.read_excel(f, engine='openpyxl', header=hdr_row)
            df.columns = df.columns.str.strip()
            # Drop fully empty rows and repeated header rows IEX sometimes inserts
            df = df.dropna(how='all')
            first_col = df.columns[0]
            df = df[~df[first_col].astype(str).str.lower().isin(['date', 'nan'])]
            df = df.reset_index(drop=True)
            frames.append(df)
            print(f"  ✅ Loaded: {os.path.basename(f)}  ({len(df)} rows)  [header @ row {hdr_row}]")
        except Exception as e:
            print(f"  ❌ Failed: {os.path.basename(f)} — {e}")

    if not frames:
        print("⚠️  All files failed — generating SYNTHETIC data for demo.")
        return generate_synthetic_data()

    merged = pd.concat(frames, ignore_index=True)
    return merged


def generate_synthetic_data(days: int = 365) -> pd.DataFrame:
    """
    Generates realistic synthetic IEX DAM data.
    Uses sinusoidal patterns for daily + seasonal cycles.
    Replace with real data once downloaded from iexindia.com
    """
    np.random.seed(42)
    dates      = pd.date_range(start='2024-04-01', periods=days, freq='D')
    records    = []

    for date in dates:
        day_of_year = date.day_of_year
        # Seasonal base price — Summer peaks in Apr-Jun
        seasonal   = 500 * np.sin(np.pi * (day_of_year - 60) / 180)
        for block in range(1, 97):
            hour       = (block - 1) * 0.25          # 0 – 23.75
            # Daily pattern — morning & evening peaks
            daily      = (300 * np.sin(np.pi * (hour - 6) / 12) +
                          200 * np.sin(np.pi * (hour - 17) / 6))
            base_mcp   = max(1500, 3500 + seasonal + daily +
                             np.random.normal(0, 150))
            # Weekend dip
            if date.weekday() >= 5:
                base_mcp *= 0.88
            mcv        = max(100, 800 + 0.05 * base_mcp +
                             np.random.normal(0, 80))
            records.append({
                'Date':             date,
                'Block No':         block,
                'MCP (Rs/MWh)':     round(base_mcp, 2),
                'MCV (MW)':         round(mcv, 2),
                'Purchase Bid (MW)':round(mcv * 1.1, 2),
                'Sell Bid (MW)':    round(mcv * 0.95, 2),
            })

    return pd.DataFrame(records)


# ============================================================
# STEP 2 — CLEAN & STANDARDISE
# ============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n📋 Raw shape:", df.shape)

    # Standardise column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if 'date' in cl:                            col_map[c] = 'Date'
        elif 'time block' in cl:                    col_map[c] = 'Time_Block_Label'
        elif 'block' in cl and 'time' not in cl:    col_map[c] = 'Block_No'
        elif cl == 'hour':                          col_map[c] = 'Hour_No'
        elif 'mcp' in cl and 'weighted' not in cl:  col_map[c] = 'MCP'
        elif 'mcv' in cl:                           col_map[c] = 'MCV'
        elif 'final scheduled' in cl:               col_map[c] = 'Final_Scheduled_Vol'
        elif 'purchase' in cl or 'buy' in cl:       col_map[c] = 'Purchase_Bid'
        elif 'sell' in cl:                          col_map[c] = 'Sell_Bid'
    df = df.rename(columns=col_map)

    # IEX 15-Min-Block export uses "Time Block" (e.g. "00:00 - 00:15")
    # instead of a numeric Block_No — derive it automatically.
    if 'Block_No' not in df.columns and 'Time_Block_Label' in df.columns:
        def time_label_to_block(label):
            try:
                start = str(label).split('-')[0].strip()  # "00:00"
                h, m = map(int, start.split(':'))
                return (h * 60 + m) // 15 + 1             # 1-indexed 1..96
            except Exception:
                return np.nan
        df['Block_No'] = df['Time_Block_Label'].apply(time_label_to_block)
        print("  ℹ️  Block_No derived from Time Block column.")

    required = ['Date', 'Block_No', 'MCP']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"\n⚠️  Could not map required columns: {missing}")
        print(f"   Columns seen after mapping:")
        for orig, mapped in col_map.items():
            print(f"     '{orig}'  →  '{mapped}'")
        unmapped = [c for c in df.columns if c not in col_map.values()]
        if unmapped:
            print(f"   Unmapped (unrecognised): {unmapped}")
        raise ValueError(f"Missing columns after mapping: {missing}")

    df['Date']     = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df['Block_No'] = pd.to_numeric(df['Block_No'], errors='coerce')
    df['MCP']      = pd.to_numeric(df['MCP'],      errors='coerce')
    for col in ['MCV', 'Purchase_Bid', 'Sell_Bid', 'Final_Scheduled_Vol']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    before = len(df)
    df = df.dropna(subset=['Date', 'Block_No', 'MCP'])
    df = df[df['Block_No'].between(1, 96)]
    df = df[df['MCP'] > 0]
    df = df.sort_values(['Date', 'Block_No']).reset_index(drop=True)
    print(f"✅ Cleaned shape: {df.shape}  (dropped {before - len(df)} bad rows)")

    # Derived time features
    df['DayOfWeek'] = df['Date'].dt.dayofweek   # 0=Mon … 6=Sun
    df['Month']     = df['Date'].dt.month
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['Hour']      = ((df['Block_No'] - 1) * 15) // 60
    # Peak flags
    df['IsMorningPeak']  = df['Block_No'].between(33, 52).astype(int)  # 08:00-13:00
    df['IsEveningPeak']  = df['Block_No'].between(69, 84).astype(int)  # 17:00-21:00

    return df


# ============================================================
# STEP 3 — EDA ANALYSIS FUNCTIONS
# ============================================================

def print_summary(df: pd.DataFrame):
    print("\n" + "="*55)
    print("  📊 IEX DAM — DATA SUMMARY")
    print("="*55)
    print(f"  Date range  : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"  Total days  : {df['Date'].nunique()}")
    print(f"  Total blocks: {len(df):,}")
    print(f"\n  MCP (₹/MWh)")
    print(f"    Min    : ₹{df['MCP'].min():,.2f}")
    print(f"    Max    : ₹{df['MCP'].max():,.2f}")
    print(f"    Mean   : ₹{df['MCP'].mean():,.2f}")
    print(f"    Median : ₹{df['MCP'].median():,.2f}")
    print(f"    Std Dev: ₹{df['MCP'].std():,.2f}")
    if 'MCV' in df.columns:
        print(f"\n  MCV (MW)")
        print(f"    Mean   : {df['MCV'].mean():,.1f} MW")
        print(f"    Max    : {df['MCV'].max():,.1f} MW")
    print("="*55)


def plot_1_daily_mcp_trend(df: pd.DataFrame):
    """Line chart — average daily MCP over full date range."""
    daily = df.groupby('Date')['MCP'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(daily['Date'], daily['MCP'], color=ACCENT, lw=1.2, alpha=0.9)
    ax.fill_between(daily['Date'], daily['MCP'],
                    daily['MCP'].min(), alpha=0.15, color=ACCENT)
    ax.set_title('📈  Average Daily MCP — Full Period', fontsize=13, pad=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('MCP (₹/MWh)')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('EDA_1_Daily_MCP_Trend.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Saved: EDA_1_Daily_MCP_Trend.png")


def plot_2_blockwise_avg(df: pd.DataFrame):
    """Bar chart — average MCP by block number (1–96)."""
    bw = df.groupby('Block_No')['MCP'].mean()

    colors = []
    for b in bw.index:
        if 33 <= b <= 52:   colors.append(ACCENT2)   # morning peak
        elif 69 <= b <= 84: colors.append(ACCENT4)   # evening peak
        else:               colors.append(ACCENT)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(bw.index, bw.values, color=colors, width=0.8)
    ax.set_title('⏱  Average MCP by Block Number  '
                 '(🔴 Morning Peak | 🟣 Evening Peak)',
                 fontsize=13, pad=12)
    ax.set_xlabel('Block No (1 = 00:00, 96 = 23:45)')
    ax.set_ylabel('Avg MCP (₹/MWh)')
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('EDA_2_Blockwise_Avg_MCP.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Saved: EDA_2_Blockwise_Avg_MCP.png")


def plot_3_monthly_boxplot(df: pd.DataFrame):
    """Box plot — MCP distribution by month."""
    months = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']
    month_data = [df[df['Month'] == m]['MCP'].values
                  for m in range(1, 13)]
    month_data = [x for x in month_data if len(x) > 0]
    labels     = [months[m-1] for m in range(1, 13)
                  if len(df[df['Month'] == m]) > 0]

    fig, ax = plt.subplots(figsize=(14, 4))
    bp = ax.boxplot(month_data, labels=labels, patch_artist=True,
                    medianprops=dict(color='#f0e130', lw=2))
    for patch in bp['boxes']:
        patch.set_facecolor('#1f2937')
        patch.set_edgecolor(ACCENT)
    ax.set_title('📅  MCP Distribution by Month (Seasonal Pattern)',
                 fontsize=13, pad=12)
    ax.set_ylabel('MCP (₹/MWh)')
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('EDA_3_Monthly_Boxplot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Saved: EDA_3_Monthly_Boxplot.png")


def plot_4_weekday_vs_weekend(df: pd.DataFrame):
    """Bar chart — avg MCP weekday vs weekend, by block."""
    wd = df[df['IsWeekend'] == 0].groupby('Block_No')['MCP'].mean()
    we = df[df['IsWeekend'] == 1].groupby('Block_No')['MCP'].mean()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(wd.index, wd.values, color=ACCENT,  lw=1.5, label='Weekday')
    ax.plot(we.index, we.values, color=ACCENT2, lw=1.5,
            label='Weekend', linestyle='--')
    ax.fill_between(wd.index, wd.values, we.values,
                    alpha=0.12, color=ACCENT3)
    ax.set_title('📆  Weekday vs Weekend — Block-wise MCP Pattern',
                 fontsize=13, pad=12)
    ax.set_xlabel('Block No')
    ax.set_ylabel('Avg MCP (₹/MWh)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('EDA_4_Weekday_Weekend.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Saved: EDA_4_Weekday_Weekend.png")


def plot_5_mcp_heatmap(df: pd.DataFrame):
    """Heatmap — avg MCP: Block No (y) × DayOfWeek (x)."""
    pivot = df.pivot_table(
        index='Block_No', columns='DayOfWeek',
        values='MCP', aggfunc='mean'
    )
    pivot.columns = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot.values, aspect='auto', cmap='plasma',
                   origin='lower')
    ax.set_xticks(range(7))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(0, 96, 8))
    ax.set_yticklabels([f'B{b+1}' for b in range(0, 96, 8)])
    ax.set_title('🌡  MCP Heatmap — Block × Day of Week',
                 fontsize=13, pad=12)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Block Number')
    plt.colorbar(im, ax=ax, label='Avg MCP (₹/MWh)')
    plt.tight_layout()
    plt.savefig('EDA_5_Heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Saved: EDA_5_Heatmap.png")


def plot_6_mcp_distribution(df: pd.DataFrame):
    """Histogram + KDE — overall MCP distribution."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df['MCP'], bins=80, color=ACCENT, alpha=0.7,
            edgecolor='none', density=True)
    df['MCP'].plot.kde(ax=ax, color=ACCENT2, lw=2)
    ax.axvline(df['MCP'].mean(),   color='#f0e130', lw=1.5,
               linestyle='--', label=f"Mean ₹{df['MCP'].mean():,.0f}")
    ax.axvline(df['MCP'].median(), color=ACCENT3,  lw=1.5,
               linestyle=':',  label=f"Median ₹{df['MCP'].median():,.0f}")
    ax.set_title('📊  MCP Distribution', fontsize=13, pad=12)
    ax.set_xlabel('MCP (₹/MWh)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('EDA_6_MCP_Distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Saved: EDA_6_MCP_Distribution.png")


def peak_analysis(df: pd.DataFrame):
    """Print peak vs off-peak MCP stats."""
    print("\n" + "="*55)
    print("  ⚡ PEAK HOUR ANALYSIS")
    print("="*55)
    mp  = df[df['IsMorningPeak'] == 1]['MCP']
    ep  = df[df['IsEveningPeak'] == 1]['MCP']
    off = df[(df['IsMorningPeak'] == 0) & (df['IsEveningPeak'] == 0)]['MCP']
    for label, series in [('Morning Peak (B33-52)', mp),
                           ('Evening Peak (B69-84)', ep),
                           ('Off-Peak', off)]:
        print(f"\n  {label}")
        print(f"    Avg MCP : ₹{series.mean():,.2f}/MWh")
        print(f"    Max MCP : ₹{series.max():,.2f}/MWh")
    premium = ((mp.mean() - off.mean()) / off.mean()) * 100
    print(f"\n  Peak Premium over Off-Peak: {premium:.1f}%")
    print("="*55)


def save_processed(df: pd.DataFrame, path: str = 'data/processed/DAM_cleaned.csv'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n💾 Processed data saved → {path}")


# ============================================================
# MAIN — Run all EDA steps
# ============================================================

if __name__ == "__main__":
    print("\n" + "🔵 "*18)
    print("  IEX DAM EDA — Starting Analysis")
    print("🔵 "*18)

    # 1. Load
    raw = load_iex_files(RAW_FOLDER)

    # 2. Clean
    df = clean_data(raw)

    # 3. Summary stats
    print_summary(df)

    # 4. Peak analysis
    peak_analysis(df)

    # 5. Plots
    print("\n📊 Generating charts...\n")
    plot_1_daily_mcp_trend(df)
    plot_2_blockwise_avg(df)
    plot_3_monthly_boxplot(df)
    plot_4_weekday_vs_weekend(df)
    plot_5_mcp_heatmap(df)
    plot_6_mcp_distribution(df)

    # 6. Save cleaned data
    save_processed(df)

    print("\n✅ EDA COMPLETE — All charts saved!")
    print("   Next step → 02_feature_engineering.py\n")
