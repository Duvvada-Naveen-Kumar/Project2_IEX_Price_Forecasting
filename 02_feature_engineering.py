# ============================================================
#  PROJECT 2 — IEX DAM Price Forecasting Tool
#  Phase 2: Feature Engineering
#  Author : Duvvada Naveen Kumar
#  Org    : MP Power Management Co. Ltd.
# ============================================================
#
#  INPUT  → data/processed/DAM_cleaned.csv  (from Phase 1)
#  OUTPUT → data/processed/DAM_features.csv (ready for LSTM)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, warnings
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
    'grid.alpha':       0.5,
    'font.family':      'monospace',
})
ACCENT  = '#58a6ff'
ACCENT2 = '#f78166'
ACCENT3 = '#3fb950'


# ============================================================
# STEP 1 — LOAD CLEANED DATA
# ============================================================

def load_cleaned(path: str = 'data/processed/DAM_cleaned.csv') -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"⚠️  File not found: {path}")
        print("    Generating synthetic data for demo...\n")
        return _synthetic_fallback()

    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.sort_values(['Date', 'Block_No']).reset_index(drop=True)
    print(f"✅ Loaded cleaned data: {df.shape}  "
          f"({df['Date'].min().date()} → {df['Date'].max().date()})")
    return df


def _synthetic_fallback(days: int = 365) -> pd.DataFrame:
    """Same synthetic generator as Phase 1 — for standalone testing."""
    np.random.seed(42)
    dates   = pd.date_range('2024-04-01', periods=days, freq='D')
    records = []
    for date in dates:
        doy = date.day_of_year
        seasonal = 500 * np.sin(np.pi * (doy - 60) / 180)
        for block in range(1, 97):
            hour   = (block - 1) * 0.25
            daily  = (300 * np.sin(np.pi * (hour - 6)  / 12) +
                      200 * np.sin(np.pi * (hour - 17) / 6))
            mcp    = max(1500, 3500 + seasonal + daily + np.random.normal(0, 150))
            if date.weekday() >= 5:
                mcp *= 0.88
            mcv = max(100, 800 + 0.05 * mcp + np.random.normal(0, 80))
            records.append({
                'Date':       date,
                'Block_No':   block,
                'MCP':        round(mcp, 2),
                'MCV':        round(mcv, 2),
                'DayOfWeek':  date.dayofweek,
                'Month':      date.month,
                'IsWeekend':  int(date.dayofweek >= 5),
                'IsMorningPeak': int(33 <= block <= 52),
                'IsEveningPeak': int(69 <= block <= 84),
            })
    df = pd.DataFrame(records)
    print(f"✅ Synthetic data: {df.shape}")
    return df


# ============================================================
# STEP 2 — FEATURE ENGINEERING FUNCTIONS
# ============================================================

# ── 2A: Cyclical Encoding ───────────────────────────────────
def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode periodic variables as (sin, cos) pairs.
    Prevents model treating Block 96 and Block 1 as far apart.

    Block_No  : period = 96
    DayOfWeek : period = 7
    Month     : period = 12
    """
    # Block encoding
    df['block_sin'] = np.sin(2 * np.pi * df['Block_No'] / 96)
    df['block_cos'] = np.cos(2 * np.pi * df['Block_No'] / 96)

    # Day of week encoding
    df['dow_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

    # Month encoding
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    print("  ✅ Cyclical features added: block_sin/cos, dow_sin/cos, month_sin/cos")
    return df


# ── 2B: Lag Features ────────────────────────────────────────
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag MCP by N days for the same block number.
    mcp_lag_1  = same block, 1 day ago
    mcp_lag_7  = same block, 7 days ago  (weekly pattern)
    mcp_lag_14 = same block, 14 days ago (fortnightly pattern)

    Also adds MCV lags if available.
    """
    df = df.sort_values(['Block_No', 'Date']).reset_index(drop=True)

    for lag in [1, 7, 14]:
        col_name      = f'mcp_lag_{lag}'
        df[col_name]  = df.groupby('Block_No')['MCP'].shift(lag)
        print(f"  ✅ Lag feature: {col_name}")

    if 'MCV' in df.columns:
        for lag in [1, 7]:
            col_name     = f'mcv_lag_{lag}'
            df[col_name] = df.groupby('Block_No')['MCV'].shift(lag)
            print(f"  ✅ Lag feature: {col_name}")

    df = df.sort_values(['Date', 'Block_No']).reset_index(drop=True)
    return df


# ── 2C: Rolling Statistics ──────────────────────────────────
def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling mean and std over 7 & 14 days per block.
    Captures local trend and recent price volatility.
    """
    df = df.sort_values(['Block_No', 'Date']).reset_index(drop=True)

    for window in [7, 14]:
        mean_col         = f'mcp_roll_mean_{window}'
        std_col          = f'mcp_roll_std_{window}'
        df[mean_col]     = (df.groupby('Block_No')['MCP']
                              .transform(lambda x: x.shift(1)
                                                    .rolling(window, min_periods=3)
                                                    .mean()))
        df[std_col]      = (df.groupby('Block_No')['MCP']
                              .transform(lambda x: x.shift(1)
                                                    .rolling(window, min_periods=3)
                                                    .std()))
        print(f"  ✅ Rolling features: {mean_col}, {std_col}")

    df = df.sort_values(['Date', 'Block_No']).reset_index(drop=True)
    return df


# ── 2D: Market Pressure Features ────────────────────────────
def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    bid_ratio = Purchase_Bid / Sell_Bid
    High ratio → more buyers than sellers → price pressure up
    Uses lagged values to avoid data leakage.
    """
    if 'Purchase_Bid' in df.columns and 'Sell_Bid' in df.columns:
        df['bid_ratio']      = df['Purchase_Bid'] / df['Sell_Bid'].replace(0, np.nan)
        df['bid_ratio_lag1'] = df.groupby('Block_No')['bid_ratio'].shift(1)
        df.drop(columns=['bid_ratio'], inplace=True)
        print("  ✅ Market feature: bid_ratio_lag1")
    else:
        print("  ℹ️  Skipped bid_ratio (Purchase_Bid / Sell_Bid not in data)")
    return df


# ── 2E: Price Change Features ───────────────────────────────
def add_price_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Day-over-day MCP change for same block.
    Helps LSTM understand price momentum.
    """
    df = df.sort_values(['Block_No', 'Date']).reset_index(drop=True)
    df['mcp_pct_change_1'] = (df.groupby('Block_No')['MCP']
                                .pct_change(1) * 100)   # % change vs yesterday
    df['mcp_pct_change_7'] = (df.groupby('Block_No')['MCP']
                                .pct_change(7) * 100)   # % change vs last week
    df = df.sort_values(['Date', 'Block_No']).reset_index(drop=True)
    print("  ✅ Price change features: mcp_pct_change_1, mcp_pct_change_7")
    return df


# ── 2F: Daily Aggregates ────────────────────────────────────
def add_daily_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily-level stats: avg MCP, peak MCP, off-peak MCP for previous day.
    Gives LSTM a high-level context of yesterday's market.
    """
    daily = (df.groupby('Date')['MCP']
               .agg(daily_avg='mean', daily_max='max', daily_min='min')
               .reset_index())
    daily.columns = ['Date', 'daily_avg_mcp', 'daily_max_mcp', 'daily_min_mcp']

    # Shift by 1 day to avoid leakage
    daily['Date_shifted'] = daily['Date'] + pd.Timedelta(days=1)
    daily = daily.rename(columns={'Date': '_orig_date'})
    daily = daily.rename(columns={'Date_shifted': 'Date'})

    df = df.merge(
        daily[['Date', 'daily_avg_mcp', 'daily_max_mcp', 'daily_min_mcp']],
        on='Date', how='left'
    )
    print("  ✅ Daily aggregate features: daily_avg_mcp, daily_max_mcp, daily_min_mcp")
    return df


# ============================================================
# STEP 3 — FINAL FEATURE LIST & CLEANUP
# ============================================================

FEATURE_COLS = [
    # Target
    'MCP',
    # Cyclical time
    'block_sin', 'block_cos',
    'dow_sin',   'dow_cos',
    'month_sin', 'month_cos',
    # Domain flags
    'IsWeekend', 'IsMorningPeak', 'IsEveningPeak',
    # Lag features
    'mcp_lag_1', 'mcp_lag_7', 'mcp_lag_14',
    'mcv_lag_1', 'mcv_lag_7',
    # Rolling stats
    'mcp_roll_mean_7',  'mcp_roll_std_7',
    'mcp_roll_mean_14', 'mcp_roll_std_14',
    # Price momentum
    'mcp_pct_change_1', 'mcp_pct_change_7',
    # Daily context
    'daily_avg_mcp', 'daily_max_mcp', 'daily_min_mcp',
    # Market pressure (optional)
    'bid_ratio_lag1',
]

def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only available feature columns, drop NaN rows
    (first ~14 days will have NaN from lag/rolling).
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]

    if missing:
        print(f"\n  ℹ️  Optional features not available: {missing}")

    df_feat = df[['Date', 'Block_No'] + available].copy()
    before  = len(df_feat)
    df_feat = df_feat.dropna()
    print(f"\n  ✅ Feature matrix shape : {df_feat.shape}")
    print(f"  ✅ Dropped NaN rows     : {before - len(df_feat)} "
          f"(from lag/rolling warmup)")
    print(f"  ✅ Final feature count  : {len(available)} features + target")
    print(f"\n  Feature list:")
    for i, f in enumerate(available, 1):
        marker = '🎯' if f == 'MCP' else '  '
        print(f"    {marker} {i:02d}. {f}")
    return df_feat


# ============================================================
# STEP 4 — FEATURE CORRELATION PLOT
# ============================================================

def plot_feature_correlation(df: pd.DataFrame):
    """Heatmap of feature correlations with MCP target."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'Block_No']

    corr = df[numeric_cols].corr()[['MCP']].drop('MCP').sort_values('MCP')

    fig, ax = plt.subplots(figsize=(6, 9))
    colors  = [ACCENT2 if v < 0 else ACCENT3 for v in corr['MCP']]
    bars    = ax.barh(corr.index, corr['MCP'], color=colors, edgecolor='none')
    ax.axvline(0, color='#8b949e', lw=1)
    ax.set_title('🔗  Feature Correlation with MCP (Target)',
                 fontsize=12, pad=12)
    ax.set_xlabel('Pearson Correlation')
    ax.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig('FE_Correlation_with_MCP.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Saved: FE_Correlation_with_MCP.png")


def plot_lag_effectiveness(df: pd.DataFrame):
    """Scatter: mcp_lag_1 vs MCP — shows how predictive lag is."""
    if 'mcp_lag_1' not in df.columns:
        return
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, lag in zip(axes, ['mcp_lag_1', 'mcp_lag_7', 'mcp_lag_14']):
        if lag not in df.columns:
            continue
        ax.scatter(sample[lag], sample['MCP'],
                   alpha=0.2, s=5, color=ACCENT)
        corr = sample[[lag, 'MCP']].corr().iloc[0, 1]
        ax.set_title(f'{lag}\nCorr = {corr:.3f}', fontsize=10)
        ax.set_xlabel(f'{lag} (₹/MWh)')
        ax.set_ylabel('MCP (₹/MWh)')
        ax.grid(True)
    fig.suptitle('📈  Lag Feature Effectiveness', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('FE_Lag_Effectiveness.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Saved: FE_Lag_Effectiveness.png")


# ============================================================
# STEP 5 — NORMALISATION (MinMax Scaling)
# ============================================================

def scale_features(df: pd.DataFrame):
    """
    MinMax scale all numeric features to [0, 1].
    IMPORTANT: Fit scaler on TRAIN set only — apply to val/test.
    Returns scaled df, scaler dict (min & max per column).

    We do manual scaling (no sklearn) to keep it transparent
    and easy to inverse-transform MCP predictions later.
    """
    from sklearn.preprocessing import MinMaxScaler

    feature_cols = [c for c in df.columns
                    if c not in ['Date', 'Block_No']]

    scaler       = MinMaxScaler()
    scaled_arr   = scaler.fit_transform(df[feature_cols])
    df_scaled    = pd.DataFrame(scaled_arr,
                                columns=feature_cols,
                                index=df.index)
    df_scaled    = pd.concat([df[['Date', 'Block_No']], df_scaled], axis=1)

    print(f"\n  ✅ MinMax scaling applied to {len(feature_cols)} features")
    print(f"     All values now in range [0, 1]")
    print(f"  ⚠️  Remember: fit scaler on TRAIN data only!\n"
          f"     Apply same scaler to val & test sets.")
    return df_scaled, scaler


# ============================================================
# MAIN — Run full Feature Engineering pipeline
# ============================================================

if __name__ == "__main__":
    print("\n" + "🟠 "*18)
    print("  IEX DAM FEATURE ENGINEERING — Starting")
    print("🟠 "*18 + "\n")

    # 1. Load
    df = load_cleaned('data/processed/DAM_cleaned.csv')

    # 2. Add features (order matters!)
    print("\n🔧 Building features...\n")
    df = add_cyclical_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_market_features(df)
    df = add_price_change_features(df)
    df = add_daily_aggregate_features(df)

    # 3. Finalize
    print("\n📋 Finalizing feature matrix...")
    df_feat = finalize_features(df)

    # 4. Plots
    print("\n📊 Generating feature analysis charts...")
    plot_feature_correlation(df_feat)
    plot_lag_effectiveness(df_feat)

    # 5. Scale
    print("\n⚖️  Scaling features...")
    df_scaled, scaler = scale_features(df_feat)

    # 6. Save
    out_raw    = 'data/processed/DAM_features.csv'
    out_scaled = 'data/processed/DAM_features_scaled.csv'
    os.makedirs('data/processed', exist_ok=True)
    df_feat.to_csv(out_raw,    index=False)
    df_scaled.to_csv(out_scaled, index=False)
    print(f"\n💾 Saved feature matrix   → {out_raw}")
    print(f"💾 Saved scaled features  → {out_scaled}")

    # 7. Summary
    print("\n" + "="*55)
    print("  ✅ PHASE 2 COMPLETE")
    print("="*55)
    print(f"  Rows ready for LSTM : {len(df_feat):,}")
    print(f"  Features built      : {df_feat.shape[1] - 3}")
    print(f"  Date range          : {df_feat['Date'].min().date()}"
          f" → {df_feat['Date'].max().date()}")
    print("\n  ➡️  Next step: 03_lstm_model.py\n")
