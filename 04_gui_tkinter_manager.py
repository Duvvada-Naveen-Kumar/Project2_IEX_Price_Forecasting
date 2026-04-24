# ============================================================
#  PROJECT 2 — IEX DAM Price Forecasting Tool
#  Phase 4 (Updated): Manager-Ready Tkinter GUI
#  Author : Duvvada Naveen Kumar
#  Org    : MP Power Management Co. Ltd.
# ============================================================
#
#  MANAGER WORKFLOW:
#    1. Double-click this file (or desktop shortcut)
#    2. Tomorrow's date is already selected
#    3. Click "Generate Report" — ONE button
#    4. PDF saved to Desktop automatically
#    5. Done. No coding. No graphs to interpret.
#
#  RUN:
#    python 04_gui_tkinter_manager.py
# ============================================================

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkcalendar import DateEntry
import threading, os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

# Import PDF generator
from pdf_report import generate_manager_report

# ── Config ───────────────────────────────────────────────────
MODEL_PATH   = 'models/lstm_mcp_model.keras'
SCALER_PATH  = 'models/scaler.pkl'
FEATURES_CSV = 'data/processed/DAM_features.csv'
SEQ_LEN      = 7

FEATURE_COLS = [
    'block_sin', 'block_cos', 'dow_sin', 'dow_cos',
    'month_sin', 'month_cos', 'IsWeekend',
    'IsMorningPeak', 'IsEveningPeak',
    'mcp_lag_1', 'mcp_lag_7', 'mcp_lag_14',
    'mcv_lag_1', 'mcv_lag_7',
    'mcp_roll_mean_7',  'mcp_roll_std_7',
    'mcp_roll_mean_14', 'mcp_roll_std_14',
    'mcp_pct_change_1', 'mcp_pct_change_7',
    'daily_avg_mcp', 'daily_max_mcp', 'daily_min_mcp',
]

# ── Colors ────────────────────────────────────────────────────
BG_DARK      = '#0d1117'
BG_CARD      = '#161b22'
BG_HEADER    = '#1c2128'
ACCENT_BLUE  = '#58a6ff'
ACCENT_ORG   = '#f78166'
ACCENT_GRN   = '#3fb950'
ACCENT_PRP   = '#d2a8ff'
ACCENT_RED   = '#ff6b6b'
TEXT_PRIMARY = '#e6edf3'
TEXT_MUTED   = '#8b949e'
BORDER       = '#30363d'

# ── Global state ─────────────────────────────────────────────
_model       = None
_scalers     = None
_df_features = None


# ============================================================
# BACKEND
# ============================================================

def load_assets():
    global _model, _scalers, _df_features
    # Reduce TensorFlow runtime noise on Windows CPU execution.
    import logging
    from absl import logging as absl_logging
    absl_logging.set_verbosity('error')
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    from tensorflow.keras.models import load_model
    _model   = load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        _scalers = pickle.load(f)
    _df_features = pd.read_csv(FEATURES_CSV, parse_dates=['Date'])
    _df_features = _df_features.sort_values(
                       ['Date','Block_No']).reset_index(drop=True)


def _synthetic_prediction(target_date: pd.Timestamp) -> pd.DataFrame:
    np.random.seed(int(target_date.timestamp()) % 9999)
    records = []
    doy      = target_date.day_of_year
    seasonal = 500 * np.sin(np.pi * (doy - 60) / 180)
    is_wknd  = target_date.dayofweek >= 5
    for block in range(1, 97):
        hour  = (block - 1) * 0.25
        daily = (300 * np.sin(np.pi * (hour - 6)  / 12) +
                 200 * np.sin(np.pi * (hour - 17) / 6))
        mcp   = max(1500, 3500 + seasonal + daily +
                    np.random.normal(0, 120))
        if is_wknd:
            mcp *= 0.88
        hh = ((block - 1) * 15) // 60
        mm = ((block - 1) * 15) % 60
        if 33 <= block <= 52:   flag = '🌅 Morning Peak'
        elif 69 <= block <= 84: flag = '🌆 Evening Peak'
        else:                   flag = 'Off-Peak'
        records.append({'Block_No': block, 'Time': f'{hh:02d}:{mm:02d}',
                         'Predicted_MCP': round(mcp, 2), 'Peak_Flag': flag})
    return pd.DataFrame(records)


def predict_next_day(target_date_str: str) -> pd.DataFrame:
    target_date = pd.Timestamp(target_date_str)
    if _model is None or _scalers is None or _df_features is None:
        return _synthetic_prediction(target_date)

    feat_scaler   = _scalers['feature_scaler']
    target_scaler = _scalers['target_scaler']
    df            = _df_features
    avail_feats   = [c for c in FEATURE_COLS if c in df.columns]
    n_features    = len(avail_feats)
    predictions   = []
    batch_rows    = []
    batch_inputs  = []

    for block_no in range(1, 97):
        blk = (df[df['Block_No'] == block_no]
               .sort_values('Date'))
        blk = blk[blk['Date'] < target_date].tail(SEQ_LEN)
        hh  = ((block_no - 1) * 15) // 60
        mm  = ((block_no - 1) * 15) % 60
        if 33 <= block_no <= 52:   flag = '🌅 Morning Peak'
        elif 69 <= block_no <= 84: flag = '🌆 Evening Peak'
        else:                      flag = 'Off-Peak'

        if len(blk) < SEQ_LEN:
            predictions.append({'Block_No': block_no,
                                 'Time': f'{hh:02d}:{mm:02d}',
                                 'Predicted_MCP': np.nan,
                                 'Peak_Flag': flag})
            continue

        seq    = blk[avail_feats].values.reshape(1, SEQ_LEN, n_features)
        seq_sc = feat_scaler.transform(
                     seq.reshape(-1, n_features)).reshape(
                         1, SEQ_LEN, n_features)
        batch_rows.append((block_no, hh, mm, flag))
        batch_inputs.append(seq_sc[0])

    if batch_inputs:
        batch_arr = np.array(batch_inputs)
        pred_sc_batch = _model.predict(batch_arr, verbose=0).reshape(-1, 1)
        pred_batch = target_scaler.inverse_transform(pred_sc_batch).reshape(-1)
        for (block_no, hh, mm, flag), pred in zip(batch_rows, pred_batch):
            predictions.append({'Block_No': block_no,
                                'Time': f'{hh:02d}:{mm:02d}',
                                'Predicted_MCP': round(max(0, pred), 2),
                                'Peak_Flag': flag})

    predictions.sort(key=lambda x: x['Block_No'])
    return pd.DataFrame(predictions)


# ============================================================
# GUI
# ============================================================

class IEXForecastApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('⚡ IEX DAM Forecast Tool — MP PPMCL')
        self.geometry('1300x780')
        self.minsize(1100, 680)
        self.configure(bg=BG_DARK)

        self._forecast_df  = None
        self._forecast_date= None
        self._loading      = False

        self._build_styles()
        self._build_header()
        self._build_controls()
        self._build_main_area()
        self._build_statusbar()

        self._set_status('⏳  Loading model & data...', ACCENT_ORG)
        threading.Thread(target=self._load_in_background,
                         daemon=True).start()

    # ── Styles ──────────────────────────────────────────────
    def _build_styles(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('Treeview',
                         background=BG_CARD,
                         foreground=TEXT_PRIMARY,
                         fieldbackground=BG_CARD,
                         rowheight=22,
                         font=('Consolas', 9))
        style.configure('Treeview.Heading',
                         background=BG_HEADER,
                         foreground=ACCENT_BLUE,
                         font=('Consolas', 9, 'bold'))
        style.map('Treeview',
                  background=[('selected', '#1f4068')],
                  foreground=[('selected', TEXT_PRIMARY)])

    # ── Header ──────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self, bg=BG_HEADER, height=60)
        hdr.pack(fill='x', side='top')
        hdr.pack_propagate(False)

        tk.Label(hdr, text='⚡  IEX DAM Price Forecasting Tool',
                 bg=BG_HEADER, fg=ACCENT_BLUE,
                 font=('Consolas', 16, 'bold')).pack(
                     side='left', padx=20, pady=12)

        tk.Label(hdr,
                 text='MP Power Management Co. Ltd.  |  '
                      'Powered by LSTM Deep Learning',
                 bg=BG_HEADER, fg=TEXT_MUTED,
                 font=('Consolas', 9)).pack(side='right', padx=20)

    # ── Controls ─────────────────────────────────────────────
    def _build_controls(self):
        ctrl = tk.Frame(self, bg=BG_CARD, height=58)
        ctrl.pack(fill='x', side='top', pady=(1, 0))
        ctrl.pack_propagate(False)

        # Date label
        tk.Label(ctrl, text='📅  Select Date:',
                 bg=BG_CARD, fg=TEXT_PRIMARY,
                 font=('Consolas', 10)).pack(
                     side='left', padx=(20, 6), pady=14)

        # Default = tomorrow
        tomorrow = datetime.today() + timedelta(days=1)
        self.date_entry = DateEntry(
            ctrl, width=13,
            background=BG_HEADER,
            foreground=TEXT_PRIMARY,
            font=('Consolas', 10),
            date_pattern='yyyy-mm-dd',
            year=tomorrow.year,
            month=tomorrow.month,
            day=tomorrow.day,
        )
        self.date_entry.pack(side='left', pady=14)

        # ── Main action button — BIG and obvious ─────────────
        self.btn_report = tk.Button(
            ctrl,
            text='📄  GENERATE REPORT',
            bg=ACCENT_GRN, fg='#000000',
            font=('Consolas', 12, 'bold'),
            relief='flat', padx=20, pady=6,
            cursor='hand2',
            command=self._on_generate_report,
        )
        self.btn_report.pack(side='left', padx=16, pady=10)

        # Secondary — Export Excel
        self.btn_excel = tk.Button(
            ctrl, text='💾  Excel',
            bg=ACCENT_BLUE, fg='#000000',
            font=('Consolas', 9, 'bold'),
            relief='flat', padx=10, pady=6,
            cursor='hand2',
            state='disabled',
            command=self._on_export_excel,
        )
        self.btn_excel.pack(side='left', padx=4, pady=10)

        # Helper text for manager
        tk.Label(ctrl,
                 text='👆  Select date → Click Generate Report → '
                      'PDF saved to your Desktop automatically',
                 bg=BG_CARD, fg=ACCENT_GRN,
                 font=('Consolas', 8)).pack(
                     side='left', padx=16)

        # Legend
        for color, label in [(ACCENT_ORG, '🌅 Morning Peak'),
                              (ACCENT_PRP, '🌆 Evening Peak'),
                              (ACCENT_BLUE, 'Off-Peak')]:
            tk.Label(ctrl, text='█', bg=BG_CARD, fg=color,
                     font=('Consolas', 13)).pack(side='right', padx=(0,2))
            tk.Label(ctrl, text=label, bg=BG_CARD, fg=TEXT_MUTED,
                     font=('Consolas', 8)).pack(side='right', padx=(0,6))

    # ── Main Area ─────────────────────────────────────────────
    def _build_main_area(self):
        main = tk.Frame(self, bg=BG_DARK)
        main.pack(fill='both', expand=True, padx=10, pady=6)

        # Left — Chart
        left = tk.Frame(main, bg=BG_CARD, relief='flat', bd=1)
        left.pack(side='left', fill='both', expand=True, padx=(0,5))

        tk.Label(left,
                 text='📊  Block-wise Forecasted MCP (15-min intervals)',
                 bg=BG_CARD, fg=ACCENT_BLUE,
                 font=('Consolas', 10, 'bold')).pack(
                     anchor='w', padx=10, pady=(8,0))

        self.fig, self.ax = plt.subplots(figsize=(7, 4.5))
        self.fig.patch.set_facecolor(BG_CARD)
        self._draw_empty_chart()
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill='both', expand=True,
                                          padx=6, pady=6)

        # Summary strip
        self.sum_frame = tk.Frame(left, bg=BG_HEADER)
        self.sum_frame.pack(fill='x', padx=6, pady=(0,6))
        self._build_summary_labels()

        # Right — Table
        right = tk.Frame(main, bg=BG_CARD, relief='flat', bd=1)
        right.pack(side='right', fill='both', padx=(5,0))
        right.configure(width=360)
        right.pack_propagate(False)

        tk.Label(right, text='📋  All 96 Blocks — Forecast Table',
                 bg=BG_CARD, fg=ACCENT_BLUE,
                 font=('Consolas', 10, 'bold')).pack(
                     anchor='w', padx=10, pady=(8,0))

        # Manager tip
        tk.Label(right,
                 text='🟢 Green = Buy  |  🟠 Orange = Morning Peak  '
                      '|  🟣 Purple = Evening Peak',
                 bg=BG_CARD, fg=TEXT_MUTED,
                 font=('Consolas', 7)).pack(anchor='w', padx=10)

        cols = ('Block', 'Time', 'Price (Rs/MWh)', 'Action')
        self.tree = ttk.Treeview(right, columns=cols,
                                  show='headings', height=30)
        for col, w in zip(cols, [50, 60, 120, 100]):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor='center')

        self.tree.tag_configure('buy',     background='#0d2818',
                                 foreground=ACCENT_GRN)
        self.tree.tag_configure('morning', background='#2d1f0e',
                                 foreground=ACCENT_ORG)
        self.tree.tag_configure('evening', background='#1e1028',
                                 foreground=ACCENT_PRP)

        sb = ttk.Scrollbar(right, orient='vertical',
                            command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self.tree.pack(fill='both', expand=True,
                        padx=(6,0), pady=6)

    def _build_summary_labels(self):
        self.lbl_avg  = self._sum_label('Avg Price',     '—')
        self.lbl_cheap= self._sum_label('Cheapest Block','—')
        self.lbl_exp  = self._sum_label('Peak Max',      '—')
        self.lbl_save = self._sum_label('Saving Opportunity','—')

    def _sum_label(self, title, val):
        f = tk.Frame(self.sum_frame, bg=BG_HEADER)
        f.pack(side='left', expand=True, fill='x', padx=4, pady=4)
        tk.Label(f, text=title, bg=BG_HEADER, fg=TEXT_MUTED,
                 font=('Consolas', 7)).pack()
        lbl = tk.Label(f, text=val, bg=BG_HEADER, fg=ACCENT_BLUE,
                        font=('Consolas', 10, 'bold'))
        lbl.pack()
        return lbl

    # ── Status bar ───────────────────────────────────────────
    def _build_statusbar(self):
        sb = tk.Frame(self, bg=BG_HEADER, height=28)
        sb.pack(fill='x', side='bottom')
        sb.pack_propagate(False)
        self.status_var = tk.StringVar(value='Initializing...')
        tk.Label(sb, textvariable=self.status_var,
                 bg=BG_HEADER, fg=TEXT_MUTED,
                 font=('Consolas', 8), anchor='w').pack(
                     side='left', padx=12, pady=6)

        # Version tag
        tk.Label(sb,
                 text='IEX DAM Forecast Tool v2.0  |  MP PPMCL',
                 bg=BG_HEADER, fg=TEXT_MUTED,
                 font=('Consolas', 7)).pack(side='right', padx=12)

    def _set_status(self, msg, color=TEXT_MUTED):
        self.status_var.set(msg)

    # ── Chart ─────────────────────────────────────────────────
    def _draw_empty_chart(self):
        self.ax.clear()
        self.ax.set_facecolor(BG_CARD)
        self.ax.set_title(
            'Select a date and click  📄 GENERATE REPORT  to begin',
            color=TEXT_MUTED, fontsize=10, fontfamily='monospace')
        self.ax.tick_params(colors=TEXT_MUTED)
        for sp in self.ax.spines.values():
            sp.set_edgecolor(BORDER)
        self.fig.tight_layout()

    def _draw_chart(self, df, date_str):
        self.ax.clear()
        self.ax.set_facecolor(BG_CARD)

        # Drop NaN/Inf — prevents "Axis limits cannot be NaN or Inf"
        plot_df = df.dropna(subset=['Predicted_MCP']).copy()
        plot_df = plot_df[np.isfinite(plot_df['Predicted_MCP'])]
        if plot_df.empty:
            self.ax.set_title('No valid predictions to display.',
                               color=ACCENT_RED, fontsize=10,
                               fontfamily='monospace')
            self.fig.tight_layout()
            self.canvas.draw()
            return

        colors_list = []
        for _, row in plot_df.iterrows():
            if '🌅' in str(row.get('Peak_Flag', '')):
                colors_list.append(ACCENT_ORG)
            elif '🌆' in str(row.get('Peak_Flag', '')):
                colors_list.append(ACCENT_PRP)
            else:
                colors_list.append(ACCENT_BLUE)

        self.ax.bar(plot_df['Block_No'], plot_df['Predicted_MCP'],
                    color=colors_list, width=0.8, alpha=0.85)
        self.ax.plot(plot_df['Block_No'], plot_df['Predicted_MCP'],
                     color='#e6edf3', lw=0.7, alpha=0.4)
        self.ax.set_title(f'Forecasted DAM MCP — {date_str}',
                           color=TEXT_PRIMARY, fontsize=11,
                           fontfamily='monospace', pad=10)
        self.ax.set_xlabel('Block Number  (1=00:00  |  48=12:00noon  |  96=23:45)',
                            color=TEXT_MUTED, fontsize=8,
                            fontfamily='monospace')
        self.ax.set_ylabel('Price (Rs/MWh)', color=TEXT_MUTED,
                            fontsize=9, fontfamily='monospace')
        self.ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        self.ax.grid(axis='y', color=BORDER, linestyle='--', alpha=0.5)
        for sp in self.ax.spines.values():
            sp.set_edgecolor(BORDER)
        patches = [
            mpatches.Patch(color=ACCENT_ORG,  label='Morning Peak (8am-1pm) — Expensive'),
            mpatches.Patch(color=ACCENT_PRP,  label='Evening Peak (5pm-9pm) — Expensive'),
            mpatches.Patch(color=ACCENT_BLUE, label='Off-Peak — Best buying time'),
        ]
        self.ax.legend(handles=patches, loc='upper left',
                        facecolor=BG_HEADER, edgecolor=BORDER,
                        labelcolor=TEXT_PRIMARY,
                        prop={'family':'monospace','size':8})
        self.fig.tight_layout()
        self.canvas.draw()

    # ── Table ─────────────────────────────────────────────────
    def _populate_table(self, df):
        for r in self.tree.get_children():
            self.tree.delete(r)
        for _, row in df.iterrows():
            mcp  = row['Predicted_MCP']
            flag = str(row.get('Peak_Flag', ''))
            mcp_str = f"Rs {mcp:,.0f}" if pd.notna(mcp) else 'N/A'
            if '🌅' in flag:
                tag    = 'morning'
                action = '⚠ EXPENSIVE'
            elif '🌆' in flag:
                tag    = 'evening'
                action = '❌ AVOID'
            else:
                tag    = 'buy'
                action = '✅ BUY'
            self.tree.insert('', 'end',
                              values=(int(row['Block_No']),
                                      row['Time'],
                                      mcp_str,
                                      action),
                              tags=(tag,))

    def _update_summary(self, df):
        valid  = df.dropna(subset=['Predicted_MCP'])
        valid  = valid[np.isfinite(valid['Predicted_MCP'])]
        if valid.empty:
            return
        avg    = valid['Predicted_MCP'].mean()
        peak   = valid[valid['Peak_Flag'].str.contains('Peak', na=False)]
        offpk  = valid[~valid['Peak_Flag'].str.contains('Peak', na=False)]
        cheap  = valid.loc[valid['Predicted_MCP'].idxmin()]
        saving = (peak['Predicted_MCP'].mean() - offpk['Predicted_MCP'].mean()
                  ) if not peak.empty and not offpk.empty else 0.0
        peak_max = peak['Predicted_MCP'].max() if not peak.empty else avg

        self.lbl_avg.config(  text=f'Rs {avg:,.0f}/MWh')
        self.lbl_cheap.config(text=f'{cheap["Time"]}  Rs {cheap["Predicted_MCP"]:,.0f}',
                               fg=ACCENT_GRN)
        self.lbl_exp.config(  text=f'Rs {peak_max:,.0f}/MWh',
                               fg=ACCENT_ORG)
        self.lbl_save.config( text=f'Rs {saving:,.0f}/MWh',
                               fg=ACCENT_GRN)

    # ── Loaders ───────────────────────────────────────────────
    def _load_in_background(self):
        try:
            if (os.path.exists(MODEL_PATH) and
                    os.path.exists(SCALER_PATH) and
                    os.path.exists(FEATURES_CSV)):
                self.after(0, lambda: self._set_status(
                    '⏳  Loading TensorFlow & LSTM model '
                    '(first load takes ~20s)...', ACCENT_ORG))
                load_assets()
                self.after(0, lambda: self._set_status(
                    '✅  Model ready — select date & click Generate Report',
                    ACCENT_GRN))
            else:
                missing = []
                if not os.path.exists(MODEL_PATH):   missing.append('LSTM model')
                if not os.path.exists(SCALER_PATH):  missing.append('scaler')
                if not os.path.exists(FEATURES_CSV): missing.append('features CSV')
                self.after(0, lambda: self._set_status(
                    f'⚠️  DEMO mode — missing: {", ".join(missing)}. '
                    f'Synthetic predictions will be used.',
                    ACCENT_ORG))
        except Exception as e:
            err = str(e)
            # TF failed to load — fall back to demo mode gracefully
            self.after(0, lambda: self._set_status(
                f'⚠️  Model load failed ({err[:60]}...). '
                f'Running in DEMO mode with synthetic predictions.',
                ACCENT_ORG))

    # ── Main Action — Generate Report ─────────────────────────
    def _on_generate_report(self):
        if self._loading:
            return
        date_str = self.date_entry.get_date().strftime('%Y-%m-%d')
        self._forecast_date = date_str
        self._loading       = True

        self.btn_report.config(state='disabled',
                                text='⏳  Working...')
        self._set_status(
            f'⏳  Running forecast & generating PDF for {date_str}...',
            ACCENT_BLUE)

        def _run():
            try:
                df   = predict_next_day(date_str)
                path = generate_manager_report(df, date_str)
                self.after(0, lambda: self._on_done(df, date_str, path))
            except Exception as e:
                err = str(e)
                self.after(0, lambda: self._on_error(err))

        threading.Thread(target=_run, daemon=True).start()

    def _on_done(self, df, date_str, pdf_path):
        self._forecast_df = df
        self._draw_chart(df, date_str)
        self._populate_table(df)
        self._update_summary(df)
        self.btn_excel.config(state='normal')
        self.btn_report.config(state='normal',
                                text='📄  GENERATE REPORT')
        self._set_status(
            f'✅  PDF Report saved to Desktop: '
            f'DAM_Forecast_{date_str.replace("-","")}.pdf  |  '
            f'96 blocks forecasted successfully',
            ACCENT_GRN)
        self._loading = False
        # Show success popup to manager
        messagebox.showinfo(
            '✅ Report Ready!',
            f'Your forecast report has been saved to:\n\n'
            f'{pdf_path}\n\n'
            f'Open the PDF to see:\n'
            f'• Key price numbers for tomorrow\n'
            f'• Buy / Avoid recommendations\n'
            f'• Price charts with plain explanations\n'
            f'• Complete 96-block price table'
        )

    def _on_error(self, err):
        messagebox.showerror('Error', err)
        self.btn_report.config(state='normal',
                                text='📄  GENERATE REPORT')
        self._set_status(f'❌  Error: {err}')
        self._loading = False

    def _on_export_excel(self):
        if self._forecast_df is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.xlsx',
            filetypes=[('Excel files', '*.xlsx')],
            initialfile=f'DAM_Forecast_'
                        f'{self._forecast_date.replace("-","")}.xlsx',
        )
        if not path:
            return
        try:
            with pd.ExcelWriter(path, engine='openpyxl') as w:
                self._forecast_df.to_excel(w, sheet_name='DAM Forecast',
                                            index=False)
            self._set_status(f'💾  Excel saved → {path}', ACCENT_GRN)
            messagebox.showinfo('Saved!', f'Excel file saved:\n{path}')
        except Exception as e:
            messagebox.showerror('Export Error', str(e))


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    app = IEXForecastApp()
    app.mainloop()
