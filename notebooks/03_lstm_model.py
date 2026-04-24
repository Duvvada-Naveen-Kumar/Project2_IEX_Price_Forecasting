# ============================================================
#  PROJECT 2 — IEX DAM Price Forecasting Tool
#  Phase 3: LSTM Model — Build, Train, Evaluate, Save
#  Author : Duvvada Naveen Kumar
#  Org    : MP Power Management Co. Ltd.
# ============================================================
#
#  INPUT  → data/processed/DAM_features.csv  (from Phase 2)
#  OUTPUT → models/lstm_mcp_model.keras
#           models/scaler.pkl
#           LSTM_*.png  (evaluation charts)
# ============================================================

import os, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # suppress TF info logs

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                     BatchNormalization, Input)
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Reproducibility ──────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

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
ACCENT4 = '#d2a8ff'

# ── Config ───────────────────────────────────────────────────
SEQ_LEN    = 7          # look back 7 days
BATCH_SIZE = 64
EPOCHS     = 100        # EarlyStopping will stop earlier
LR         = 0.001
MODEL_PATH = 'models/lstm_mcp_model.keras'
SCALER_PATH= 'models/scaler.pkl'
os.makedirs('models', exist_ok=True)

# Feature columns (must match Phase 2 output)
FEATURE_COLS = [
    'block_sin', 'block_cos',
    'dow_sin',   'dow_cos',
    'month_sin', 'month_cos',
    'IsWeekend', 'IsMorningPeak', 'IsEveningPeak',
    'mcp_lag_1', 'mcp_lag_7', 'mcp_lag_14',
    'mcv_lag_1', 'mcv_lag_7',
    'mcp_roll_mean_7',  'mcp_roll_std_7',
    'mcp_roll_mean_14', 'mcp_roll_std_14',
    'mcp_pct_change_1', 'mcp_pct_change_7',
    'daily_avg_mcp', 'daily_max_mcp', 'daily_min_mcp',
]
TARGET_COL = 'MCP'


# ============================================================
# STEP 1 — LOAD & PREPARE DATA
# ============================================================

def load_features(path: str = 'data/processed/DAM_features.csv') -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"⚠️  {path} not found — generating synthetic data...")
        return _synthetic_features()

    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.sort_values(['Date', 'Block_No']).reset_index(drop=True)
    print(f"✅ Loaded features: {df.shape}  "
          f"({df['Date'].min().date()} → {df['Date'].max().date()})")
    return df


def _synthetic_features(days: int = 365) -> pd.DataFrame:
    """Generate synthetic feature data for standalone testing."""
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
            records.append({
                'Date': date, 'Block_No': block, 'MCP': round(mcp, 2),
                'block_sin': np.sin(2*np.pi*block/96),
                'block_cos': np.cos(2*np.pi*block/96),
                'dow_sin':   np.sin(2*np.pi*date.dayofweek/7),
                'dow_cos':   np.cos(2*np.pi*date.dayofweek/7),
                'month_sin': np.sin(2*np.pi*date.month/12),
                'month_cos': np.cos(2*np.pi*date.month/12),
                'IsWeekend':      int(date.dayofweek >= 5),
                'IsMorningPeak':  int(33 <= block <= 52),
                'IsEveningPeak':  int(69 <= block <= 84),
                'mcp_lag_1':      mcp + np.random.normal(0, 100),
                'mcp_lag_7':      mcp + np.random.normal(0, 150),
                'mcp_lag_14':     mcp + np.random.normal(0, 180),
                'mcv_lag_1':      800 + np.random.normal(0, 80),
                'mcv_lag_7':      800 + np.random.normal(0, 80),
                'mcp_roll_mean_7':  mcp + np.random.normal(0, 80),
                'mcp_roll_std_7':   abs(np.random.normal(150, 40)),
                'mcp_roll_mean_14': mcp + np.random.normal(0, 100),
                'mcp_roll_std_14':  abs(np.random.normal(170, 40)),
                'mcp_pct_change_1': np.random.normal(0, 3),
                'mcp_pct_change_7': np.random.normal(0, 5),
                'daily_avg_mcp':    mcp + np.random.normal(0, 80),
                'daily_max_mcp':    mcp + 300,
                'daily_min_mcp':    mcp - 300,
            })
    return pd.DataFrame(records).dropna()


# ============================================================
# STEP 2 — BUILD SEQUENCES FOR LSTM
# ============================================================

def build_sequences(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    """
    For each block, create sequences of seq_len consecutive days.

    Example (seq_len=7, predicting Block 33):
      X[i] = features of Block 33 from day i to day i+6  → shape (7, n_features)
      y[i] = MCP of Block 33 on day i+7                  → shape (1,)

    This is done PER BLOCK to keep temporal order correct.
    """
    available_feats = [c for c in FEATURE_COLS if c in df.columns]
    n_features      = len(available_feats)

    print(f"\n  Building sequences: seq_len={seq_len}, features={n_features}")

    X_list, y_list, meta_list = [], [], []

    for block_no, grp in df.groupby('Block_No'):
        grp = grp.sort_values('Date').reset_index(drop=True)
        feat_arr   = grp[available_feats].values   # shape (n_days, n_features)
        target_arr = grp[TARGET_COL].values        # shape (n_days,)

        for i in range(seq_len, len(grp)):
            X_list.append(feat_arr[i - seq_len : i])   # past seq_len days
            y_list.append(target_arr[i])                # next day target
            meta_list.append({
                'Date':     grp['Date'].iloc[i],
                'Block_No': block_no,
            })

    X = np.array(X_list, dtype=np.float32)   # (samples, seq_len, n_features)
    y = np.array(y_list,  dtype=np.float32)  # (samples,)
    meta = pd.DataFrame(meta_list)

    print(f"  X shape: {X.shape}  |  y shape: {y.shape}")
    return X, y, meta, available_feats


def train_val_test_split(X, y, meta,
                         train_ratio=0.80,
                         val_ratio=0.10):
    """
    CHRONOLOGICAL split — no random shuffling!
    Random split leaks future info into training → inflated metrics.
    """
    n        = len(y)
    n_train  = int(n * train_ratio)
    n_val    = int(n * val_ratio)

    X_train, y_train = X[:n_train],          y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:],    y[n_train+n_val:]
    meta_test        = meta.iloc[n_train+n_val:].reset_index(drop=True)

    print(f"\n  ✅ Chronological split:")
    print(f"     Train : {X_train.shape[0]:>6,} samples  ({train_ratio*100:.0f}%)")
    print(f"     Val   : {X_val.shape[0]:>6,} samples  ({val_ratio*100:.0f}%)")
    print(f"     Test  : {X_test.shape[0]:>6,} samples  "
          f"({(1-train_ratio-val_ratio)*100:.0f}%)")
    return X_train, y_train, X_val, y_val, X_test, y_test, meta_test


def scale_data(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Scale features and target separately.
    Fit ONLY on train — transform val & test with same scaler.
    """
    n_train, sl, nf = X_train.shape

    # Flatten → scale → reshape
    feat_scaler = MinMaxScaler()
    X_train_2d  = X_train.reshape(-1, nf)
    X_train_sc  = feat_scaler.fit_transform(X_train_2d).reshape(n_train, sl, nf)
    X_val_sc    = feat_scaler.transform(X_val.reshape(-1, nf)).reshape(X_val.shape)
    X_test_sc   = feat_scaler.transform(X_test.reshape(-1, nf)).reshape(X_test.shape)

    # Scale target (MCP) separately for easy inverse transform
    target_scaler = MinMaxScaler()
    y_train_sc    = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_sc      = target_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_sc     = target_scaler.transform(y_test.reshape(-1, 1)).ravel()

    # Save scalers
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump({'feature_scaler': feat_scaler,
                     'target_scaler':  target_scaler}, f)
    print(f"\n  ✅ Scalers saved → {SCALER_PATH}")

    return (X_train_sc, y_train_sc,
            X_val_sc,   y_val_sc,
            X_test_sc,  y_test_sc,
            feat_scaler, target_scaler)


# ============================================================
# STEP 3 — BUILD LSTM MODEL
# ============================================================

def build_lstm_model(seq_len: int, n_features: int) -> tf.keras.Model:
    """
    Architecture:
      Input → LSTM(128) → Dropout → BatchNorm
            → LSTM(64)  → Dropout → BatchNorm
            → Dense(32, relu)
            → Dense(16, relu)
            → Dense(1)  [MCP output]
    """
    model = Sequential([
        Input(shape=(seq_len, n_features)),

        # Layer 1 — LSTM 128 units
        LSTM(128, return_sequences=True,
             kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        Dropout(0.2),
        BatchNormalization(),

        # Layer 2 — LSTM 64 units
        LSTM(64, return_sequences=False,
             kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        Dropout(0.2),
        BatchNormalization(),

        # Dense head
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1),                        # linear output → MCP
    ], name='IEX_DAM_LSTM')

    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss='mae',                      # MAE — robust to outliers
        metrics=['mse']
    )
    return model


# ============================================================
# STEP 4 — TRAIN
# ============================================================

def train_model(model, X_train, y_train, X_val, y_val):
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        ),
    ]

    print(f"\n🚀 Training started — max {EPOCHS} epochs "
          f"(EarlyStopping patience=12)...\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    print(f"\n✅ Training complete! Best model saved → {MODEL_PATH}")
    return history


# ============================================================
# STEP 5 — EVALUATE
# ============================================================

def evaluate_model(model, X_test, y_test, target_scaler, meta_test):
    """
    Predict on test set → inverse transform → compute metrics.
    Returns actual & predicted MCP in original ₹/MWh scale.
    """
    y_pred_sc = model.predict(X_test, verbose=0).ravel()

    # Inverse transform to ₹/MWh
    y_actual  = target_scaler.inverse_transform(
                    y_test.reshape(-1, 1)).ravel()
    y_pred    = target_scaler.inverse_transform(
                    y_pred_sc.reshape(-1, 1)).ravel()

    # Metrics
    mae  = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
    r2   = 1 - (np.sum((y_actual - y_pred)**2) /
                np.sum((y_actual - y_actual.mean())**2))

    print("\n" + "="*55)
    print("  📊 MODEL EVALUATION — TEST SET")
    print("="*55)
    print(f"  MAE  : ₹{mae:>10.2f} /MWh")
    print(f"  RMSE : ₹{rmse:>10.2f} /MWh")
    print(f"  MAPE :  {mape:>9.2f} %")
    print(f"  R²   :  {r2:>9.4f}")
    print("="*55)

    if mape < 5:
        print("  🏆 Excellent! MAPE < 5%")
    elif mape < 10:
        print("  ✅ Good! MAPE < 10% — meets project target")
    else:
        print("  ⚠️  MAPE > 10% — consider more data or tuning")

    results = meta_test.copy()
    results['Actual_MCP']    = y_actual
    results['Predicted_MCP'] = y_pred
    results['Error']         = y_pred - y_actual
    results['AbsPctError']   = np.abs(results['Error'] / y_actual) * 100
    return results, {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}


# ============================================================
# STEP 6 — EVALUATION CHARTS
# ============================================================

def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Loss
    axes[0].plot(history.history['loss'],     color=ACCENT,  label='Train Loss')
    axes[0].plot(history.history['val_loss'], color=ACCENT2, label='Val Loss')
    axes[0].set_title('📉  Training & Validation Loss (MAE)', fontsize=12)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MAE (scaled)')
    axes[0].legend()
    axes[0].grid(True)

    # MSE
    axes[1].plot(history.history['mse'],     color=ACCENT,  label='Train MSE')
    axes[1].plot(history.history['val_mse'], color=ACCENT2, label='Val MSE')
    axes[1].set_title('📉  Training & Validation MSE', fontsize=12)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE (scaled)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('LSTM_1_Training_History.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Saved: LSTM_1_Training_History.png")


def plot_actual_vs_predicted(results: pd.DataFrame, n_days: int = 7):
    """
    Show actual vs predicted MCP for last n_days of test set,
    for a sample block (Block 48 = midday).
    """
    sample_block = 48
    blk = results[results['Block_No'] == sample_block].tail(n_days * 1)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(range(len(blk)), blk['Actual_MCP'],
            color=ACCENT,  lw=2,   label='Actual MCP',    marker='o', ms=4)
    ax.plot(range(len(blk)), blk['Predicted_MCP'],
            color=ACCENT2, lw=1.5, label='Predicted MCP', marker='s', ms=4,
            linestyle='--')
    ax.fill_between(range(len(blk)),
                    blk['Actual_MCP'], blk['Predicted_MCP'],
                    alpha=0.12, color=ACCENT3)
    ax.set_title(f'🎯  Actual vs Predicted MCP — Block {sample_block} '
                 f'(~12:00 noon)', fontsize=12)
    ax.set_xlabel('Test Sample Index')
    ax.set_ylabel('MCP (₹/MWh)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('LSTM_2_Actual_vs_Predicted.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Saved: LSTM_2_Actual_vs_Predicted.png")


def plot_blockwise_mape(results: pd.DataFrame):
    """Bar chart — MAPE by block number. Reveals which blocks are hardest."""
    bw_mape = (results.groupby('Block_No')['AbsPctError']
                      .mean()
                      .reset_index())
    bw_mape.columns = ['Block_No', 'MAPE']

    colors = [ACCENT2 if v > 10 else ACCENT3 if v < 5 else ACCENT
              for v in bw_mape['MAPE']]

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(bw_mape['Block_No'], bw_mape['MAPE'], color=colors, width=0.8)
    ax.axhline(10, color=ACCENT2, lw=1.2, linestyle='--', label='10% target')
    ax.axhline(5,  color=ACCENT3, lw=1.2, linestyle=':', label='5% excellent')
    ax.set_title('⏱  Block-wise MAPE  '
                 '(🔴 >10% | 🔵 5-10% | 🟢 <5%)', fontsize=12)
    ax.set_xlabel('Block Number')
    ax.set_ylabel('MAPE (%)')
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('LSTM_3_Blockwise_MAPE.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Saved: LSTM_3_Blockwise_MAPE.png")


def plot_error_distribution(results: pd.DataFrame):
    """Histogram of prediction errors."""
    errors = results['Error']
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(errors, bins=60, color=ACCENT, alpha=0.75, edgecolor='none')
    ax.axvline(0,            color='#f0e130', lw=2,   label='Zero error')
    ax.axvline(errors.mean(),color=ACCENT2,  lw=1.5,
               linestyle='--', label=f'Mean error ₹{errors.mean():.1f}')
    ax.set_title('📊  Prediction Error Distribution', fontsize=12)
    ax.set_xlabel('Error (Predicted − Actual)  ₹/MWh')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('LSTM_4_Error_Distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Saved: LSTM_4_Error_Distribution.png")


def plot_scatter_actual_vs_pred(results: pd.DataFrame):
    """Scatter plot — perfect prediction = diagonal line."""
    sample = results.sample(min(3000, len(results)), random_state=42)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sample['Actual_MCP'], sample['Predicted_MCP'],
               alpha=0.2, s=6, color=ACCENT)
    lims = [results['Actual_MCP'].min(), results['Actual_MCP'].max()]
    ax.plot(lims, lims, color=ACCENT2, lw=1.5, linestyle='--',
            label='Perfect prediction')
    ax.set_title('🎯  Actual vs Predicted Scatter', fontsize=12)
    ax.set_xlabel('Actual MCP (₹/MWh)')
    ax.set_ylabel('Predicted MCP (₹/MWh)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('LSTM_5_Scatter.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Saved: LSTM_5_Scatter.png")


# ============================================================
# STEP 7 — PREDICT NEXT DAY (Inference function for GUI)
# ============================================================

def predict_next_day(input_date: str,
                     df_full: pd.DataFrame,
                     model_path: str = MODEL_PATH,
                     scaler_path: str = SCALER_PATH) -> pd.DataFrame:
    """
    Given a target date, predict MCP for all 96 blocks.
    Uses last SEQ_LEN days of data as input sequence.

    Returns DataFrame: Block_No | Predicted_MCP | Peak_Flag
    This function is called by both Tkinter & Streamlit GUIs.
    """
    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    feat_scaler   = scalers['feature_scaler']
    target_scaler = scalers['target_scaler']

    target_date   = pd.Timestamp(input_date)
    available_feats = [c for c in FEATURE_COLS if c in df_full.columns]
    n_features      = len(available_feats)

    predictions = []
    for block_no in range(1, 97):
        blk = (df_full[df_full['Block_No'] == block_no]
               .sort_values('Date'))
        blk = blk[blk['Date'] < target_date].tail(SEQ_LEN)

        if len(blk) < SEQ_LEN:
            predictions.append({
                'Block_No': block_no,
                'Predicted_MCP': np.nan,
                'Peak_Flag': ''
            })
            continue

        seq = blk[available_feats].values.reshape(1, SEQ_LEN, n_features)
        seq_sc  = feat_scaler.transform(seq.reshape(-1, n_features)).reshape(
                      1, SEQ_LEN, n_features)
        pred_sc = model.predict(seq_sc, verbose=0).ravel()[0]
        pred    = target_scaler.inverse_transform([[pred_sc]])[0][0]

        # Peak flag
        if 33 <= block_no <= 52:   flag = '🌅 Morning Peak'
        elif 69 <= block_no <= 84: flag = '🌆 Evening Peak'
        else:                      flag = 'Off-Peak'

        predictions.append({
            'Block_No':      block_no,
            'Time':          f"{((block_no-1)*15)//60:02d}:{((block_no-1)*15)%60:02d}",
            'Predicted_MCP': round(pred, 2),
            'Peak_Flag':     flag,
        })

    return pd.DataFrame(predictions)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "🧠 "*18)
    print("  IEX DAM LSTM MODEL — Starting")
    print("🧠 "*18 + "\n")

    # 1. Load features
    df = load_features('data/processed/DAM_features.csv')

    # 2. Build sequences
    X, y, meta, feat_cols = build_sequences(df, seq_len=SEQ_LEN)
    n_features = X.shape[2]

    # 3. Split
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     meta_test) = train_val_test_split(X, y, meta)

    # 4. Scale
    (X_train_sc, y_train_sc,
     X_val_sc,   y_val_sc,
     X_test_sc,  y_test_sc,
     feat_scaler, target_scaler) = scale_data(
         X_train, y_train, X_val, y_val, X_test, y_test)

    # 5. Build model
    print("\n🏗️  Building LSTM architecture...")
    model = build_lstm_model(SEQ_LEN, n_features)
    model.summary()

    # 6. Train
    history = train_model(model, X_train_sc, y_train_sc,
                                 X_val_sc,   y_val_sc)

    # 7. Evaluate
    results, metrics = evaluate_model(
        model, X_test_sc, y_test_sc, target_scaler, meta_test)

    # 8. Charts
    print("\n📊 Generating evaluation charts...")
    plot_training_history(history)
    plot_actual_vs_predicted(results)
    plot_blockwise_mape(results)
    plot_error_distribution(results)
    plot_scatter_actual_vs_pred(results)

    # 9. Save results CSV
    results.to_csv('models/test_predictions.csv', index=False)
    print("\n💾 Test predictions saved → models/test_predictions.csv")

    # 10. Final summary
    print("\n" + "="*55)
    print("  ✅ PHASE 3 COMPLETE — LSTM MODEL READY")
    print("="*55)
    print(f"  Model saved  → {MODEL_PATH}")
    print(f"  Scaler saved → {SCALER_PATH}")
    print(f"  MAE  : ₹{metrics['MAE']:,.2f}/MWh")
    print(f"  RMSE : ₹{metrics['RMSE']:,.2f}/MWh")
    print(f"  MAPE :  {metrics['MAPE']:.2f}%")
    print(f"  R²   :  {metrics['R2']:.4f}")
    print("\n  ➡️  Next: 04_gui_tkinter.py  &  05_streamlit_app.py\n")
