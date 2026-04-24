# ============================================================
#  PROJECT 2 — IEX DAM Price Forecasting Tool
#  PDF Report Generator — Manager Ready
#  Author : Duvvada Naveen Kumar
#  Org    : MP Power Management Co. Ltd.
# ============================================================
#
#  This script generates a plain-English PDF report
#  from forecast results. No technical knowledge needed
#  to READ the report — designed for managers.
#
#  INPUT  : forecast DataFrame (from predict_next_day())
#  OUTPUT : Desktop/DAM_Forecast_YYYYMMDD.pdf
# ============================================================

import os
import tempfile                  # ✅ Windows-safe temp folder
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image, PageBreak,
                                 HRFlowable, KeepTogether)

# ── Color palette ─────────────────────────────────────────────
RL_DARK      = colors.HexColor('#0d1117')
RL_CARD      = colors.HexColor('#161b22')
RL_HEADER    = colors.HexColor('#1c2128')
RL_BLUE      = colors.HexColor('#58a6ff')
RL_ORANGE    = colors.HexColor('#f78166')
RL_GREEN     = colors.HexColor('#3fb950')
RL_PURPLE    = colors.HexColor('#d2a8ff')
RL_TEXT      = colors.HexColor('#e6edf3')
RL_MUTED     = colors.HexColor('#8b949e')
RL_WHITE     = colors.white

PAGE_W, PAGE_H = A4

# ✅ Windows-safe temp folder (works on Windows, Linux, Mac)
TEMP_DIR = tempfile.gettempdir()


# ============================================================
# CHART GENERATOR — saves temp PNG for embedding in PDF
# ============================================================

def _generate_forecast_chart(df: pd.DataFrame,
                              date_str: str) -> str:
    """Generate block-wise MCP bar chart. Returns saved file path."""
    path = os.path.join(TEMP_DIR, 'forecast_chart.png')   # ✅ fixed

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')

    colors_list = []
    for _, row in df.iterrows():
        if '🌅' in str(row.get('Peak_Flag', '')):
            colors_list.append('#f78166')
        elif '🌆' in str(row.get('Peak_Flag', '')):
            colors_list.append('#d2a8ff')
        else:
            colors_list.append('#58a6ff')

    ax.bar(df['Block_No'], df['Predicted_MCP'],
           color=colors_list, width=0.8, alpha=0.9)
    ax.plot(df['Block_No'], df['Predicted_MCP'],
            color='#e6edf3', lw=0.6, alpha=0.4)

    ax.set_title(f'Block-wise Forecasted MCP — {date_str}',
                 color='#e6edf3', fontsize=12,
                 fontfamily='monospace', pad=10)
    ax.set_xlabel(
        'Block Number  '
        '(Block 1 = 00:00 midnight  |  Block 96 = 23:45 night)',
        color='#8b949e', fontsize=8, fontfamily='monospace')
    ax.set_ylabel('Price (Rs/MWh)', color='#8b949e',
                  fontsize=9, fontfamily='monospace')
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.grid(axis='y', color='#30363d', linestyle='--', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

    patches = [
        mpatches.Patch(color='#f78166',
                       label='Morning Peak (8am-1pm) — Expensive'),
        mpatches.Patch(color='#d2a8ff',
                       label='Evening Peak (5pm-9pm) — Expensive'),
        mpatches.Patch(color='#58a6ff',
                       label='Off-Peak — Cheap buying opportunity'),
    ]
    ax.legend(handles=patches, loc='upper left',
              facecolor='#1c2128', edgecolor='#30363d',
              labelcolor='#e6edf3',
              prop={'family': 'monospace', 'size': 8})

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor='#161b22')
    plt.close()
    return path


def _generate_peak_comparison_chart(df: pd.DataFrame) -> str:
    """Session comparison bar chart. Returns saved file path."""
    path = os.path.join(TEMP_DIR, 'peak_comparison.png')  # ✅ fixed

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')

    morning = df[df['Peak_Flag'].str.contains(
        'Morning', na=False)]['Predicted_MCP'].mean()
    evening = df[df['Peak_Flag'].str.contains(
        'Evening', na=False)]['Predicted_MCP'].mean()
    # ✅ Fixed: 'Off-Peak' also contains 'Peak' — use Morning|Evening instead
    offpeak = df[~df['Peak_Flag'].str.contains(
        'Morning|Evening', na=False)]['Predicted_MCP'].mean()

    categories = [
        'Off-Peak\n(Midnight-8am\n& 9pm-midnight)',
        'Morning Peak\n(8am - 1pm)',
        'Evening Peak\n(5pm - 9pm)',
    ]
    values     = [offpeak, morning, evening]
    bar_colors = ['#58a6ff', '#f78166', '#d2a8ff']

    bars = ax.bar(categories, values, color=bar_colors,
                  width=0.5, alpha=0.9)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 50,
                f'Rs {val:,.0f}',
                ha='center', va='bottom',
                color='#e6edf3', fontsize=10,
                fontfamily='monospace', fontweight='bold')

    ax.set_title('Average Price by Session',
                 color='#e6edf3', fontsize=11,
                 fontfamily='monospace', pad=10)
    ax.set_ylabel('Avg Price (Rs/MWh)', color='#8b949e',
                  fontsize=9, fontfamily='monospace')
    ax.tick_params(colors='#8b949e', labelsize=9)
    ax.grid(axis='y', color='#30363d', linestyle='--', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.set_ylim(0, max(values) * 1.2)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor='#161b22')
    plt.close()
    return path


# ============================================================
# PDF BUILDER
# ============================================================

def generate_manager_report(df: pd.DataFrame,
                             forecast_date: str,
                             output_path: str = None) -> str:
    """
    Generate a complete manager-ready PDF report.

    Parameters:
        df            : forecast DataFrame from predict_next_day()
        forecast_date : 'YYYY-MM-DD' string
        output_path   : where to save PDF (default = Desktop)

    Returns:
        path to saved PDF
    """
    # ── Output path ─────────────────────────────────────────
    if output_path is None:
        desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
        os.makedirs(desktop, exist_ok=True)
        fname       = f"DAM_Forecast_{forecast_date.replace('-', '')}.pdf"
        output_path = os.path.join(desktop, fname)

    # ── Summary stats ────────────────────────────────────────
    valid       = df.dropna(subset=['Predicted_MCP'])
    avg_mcp     = valid['Predicted_MCP'].mean()
    max_mcp     = valid['Predicted_MCP'].max()
    min_mcp     = valid['Predicted_MCP'].min()

    morning     = valid[valid['Peak_Flag'].str.contains('Morning', na=False)]
    evening     = valid[valid['Peak_Flag'].str.contains('Evening', na=False)]
    # ✅ Fixed: filter by Morning|Evening, not 'Peak'
    # ('Off-Peak' also contains 'Peak' — old filter gave empty set → Rs 0)
    offpeak     = valid[~valid['Peak_Flag'].str.contains('Morning|Evening', na=False)]

    morning_avg = morning['Predicted_MCP'].mean() if len(morning) else 0
    evening_avg = evening['Predicted_MCP'].mean() if len(evening) else 0
    offpeak_avg = offpeak['Predicted_MCP'].mean() if len(offpeak) else 0

    cheapest_row  = valid.loc[valid['Predicted_MCP'].idxmin()]
    expensive_row = valid.loc[valid['Predicted_MCP'].idxmax()]
    saving_opp    = evening_avg - offpeak_avg

    # ── Generate charts ──────────────────────────────────────
    chart1_path = _generate_forecast_chart(df, forecast_date)
    chart2_path = _generate_peak_comparison_chart(df)

    # ── Document setup ───────────────────────────────────────
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    # ── Styles ───────────────────────────────────────────────
    def style(name, **kwargs):
        return ParagraphStyle(name, **kwargs)

    S_TITLE   = style('Title2', fontSize=20, textColor=RL_BLUE,
                       fontName='Helvetica-Bold',
                       alignment=TA_CENTER, spaceAfter=4)
    S_SUBTITLE= style('Sub', fontSize=10, textColor=RL_MUTED,
                       fontName='Helvetica',
                       alignment=TA_CENTER, spaceAfter=2)
    S_H1      = style('H1', fontSize=13, textColor=RL_BLUE,
                       fontName='Helvetica-Bold',
                       spaceBefore=14, spaceAfter=6)
    S_H2      = style('H2', fontSize=10, textColor=RL_ORANGE,
                       fontName='Helvetica-Bold',
                       spaceBefore=8, spaceAfter=4)
    S_BODY    = style('Body2', fontSize=9,
                       textColor=colors.HexColor('#333333'),
                       fontName='Helvetica',
                       leading=14, spaceAfter=4)
    S_CAPTION = style('Caption', fontSize=8,
                       textColor=colors.HexColor('#555555'),
                       fontName='Helvetica-Oblique',
                       alignment=TA_CENTER, spaceAfter=8)
    S_REC_GRN = style('RecGrn', fontSize=9,
                       textColor=colors.HexColor('#1a5c2a'),
                       fontName='Helvetica-Bold',
                       leading=16, spaceAfter=3)
    S_REC_RED = style('RecRed', fontSize=9,
                       textColor=colors.HexColor('#5c1a1a'),
                       fontName='Helvetica-Bold',
                       leading=16, spaceAfter=3)

    story = []

    # ════════════════════════════════════════════════════════
    # PAGE 1 — EXECUTIVE SUMMARY
    # ════════════════════════════════════════════════════════

    # Header bar
    header_data = [[
        Paragraph('MP Power Management Co. Ltd.',
                  style('hdr', fontSize=11, textColor=RL_WHITE,
                         fontName='Helvetica-Bold', alignment=TA_LEFT)),
        Paragraph(
            f'Generated: {datetime.now().strftime("%d %b %Y, %I:%M %p")}',
            style('hdr2', fontSize=9, textColor=RL_MUTED,
                   fontName='Helvetica', alignment=TA_RIGHT)),
    ]]
    header_tbl = Table(header_data, colWidths=[10 * cm, 7.5 * cm])
    header_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), RL_DARK),
        ('TOPPADDING',    (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING',   (0, 0), (-1, -1), 14),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 14),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 14))

    # Title
    story.append(Paragraph('IEX DAM Price Forecast Report', S_TITLE))
    story.append(Paragraph(
        f'Forecast Date: <b>'
        f'{pd.Timestamp(forecast_date).strftime("%A, %d %B %Y")}'
        f'</b>  |  Day-Ahead Market (DAM)  |  All 96 Time Blocks',
        S_SUBTITLE))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width='100%', thickness=1,
                             color=RL_BLUE, spaceAfter=10))

    # ── KPI Cards ────────────────────────────────────────────
    story.append(Paragraph('KEY NUMBERS FOR TOMORROW', S_H1))

    def kpi_cell(label, value, unit, bg, border):
        inner = Table([
            [Paragraph(label, style('kl', fontSize=8,
                                     textColor=RL_MUTED,
                                     fontName='Helvetica',
                                     alignment=TA_CENTER))],
            [Paragraph(f'<b>{value}</b>',
                        style('kv', fontSize=16,
                               textColor=border,
                               fontName='Helvetica-Bold',
                               alignment=TA_CENTER))],
            [Paragraph(unit, style('ku', fontSize=7,
                                    textColor=RL_MUTED,
                                    fontName='Helvetica',
                                    alignment=TA_CENTER))],
        ], colWidths=[5.2 * cm])
        inner.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), bg),
            ('BOX',           (0, 0), (-1, -1), 0.8, border),
            ('TOPPADDING',    (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING',   (0, 0), (-1, -1), 6),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 6),
        ]))
        return inner

    kpi_row = [[
        kpi_cell('AVERAGE PRICE TOMORROW',
                 f'Rs {avg_mcp:,.0f}', 'Rs/MWh',
                 colors.HexColor('#0d1f35'), RL_BLUE),
        kpi_cell('CHEAPEST TIME BLOCK',
                 f'Rs {min_mcp:,.0f}',
                 cheapest_row['Time'],
                 colors.HexColor('#0d2818'), RL_GREEN),
        kpi_cell('MOST EXPENSIVE BLOCK',
                 f'Rs {max_mcp:,.0f}',
                 expensive_row['Time'],
                 colors.HexColor('#2d1f0e'), RL_ORANGE),
    ]]
    kpi_tbl = Table(kpi_row,
                    colWidths=[5.4 * cm, 5.4 * cm, 5.4 * cm],
                    rowHeights=[3.2 * cm])
    kpi_tbl.setStyle(TableStyle([
        ('ALIGN',  (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING',  (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 10))

    # Session breakdown
    session_data = [
        ['SESSION', 'TIME', 'AVG PRICE', 'WHAT THIS MEANS'],
        ['Off-Peak',
         'Midnight-8am\n& 9pm-midnight',
         f'Rs {offpeak_avg:,.0f}/MWh',
         'CHEAPEST — Best time to buy from IEX'],
        ['Morning Peak',
         '8:00 AM - 1:00 PM',
         f'Rs {morning_avg:,.0f}/MWh',
         'EXPENSIVE — Reduce exchange purchases'],
        ['Evening Peak',
         '5:00 PM - 9:00 PM',
         f'Rs {evening_avg:,.0f}/MWh',
         'MOST EXPENSIVE — Use own generation'],
    ]
    sess_tbl = Table(session_data,
                     colWidths=[3 * cm, 4.2 * cm, 3.5 * cm, 6.8 * cm])
    sess_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0), RL_DARK),
        ('TEXTCOLOR',     (0, 0), (-1, 0), RL_BLUE),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0), 8),
        ('BACKGROUND',    (0, 1), (-1, 1), colors.HexColor('#e8f5e9')),
        ('BACKGROUND',    (0, 2), (-1, 2), colors.HexColor('#fff3e0')),
        ('BACKGROUND',    (0, 3), (-1, 3), colors.HexColor('#f3e5f5')),
        ('FONTNAME',      (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',      (0, 1), (-1, -1), 8),
        ('FONTNAME',      (2, 1), (2, -1), 'Helvetica-Bold'),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID',          (0, 0), (-1, -1), 0.3,
         colors.HexColor('#cccccc')),
        ('TOPPADDING',    (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(sess_tbl)
    story.append(Spacer(1, 14))

    # ── Recommendations ──────────────────────────────────────
    story.append(Paragraph('PROCUREMENT RECOMMENDATIONS', S_H1))
    story.append(HRFlowable(width='100%', thickness=0.5,
                             color=RL_GREEN, spaceAfter=8))

    rec_buy = [
        f'BUY MAXIMUM during Off-Peak hours (midnight to 8am). '
        f'Price is around Rs {offpeak_avg:,.0f}/MWh — lowest of the day.',
        f'If DAM purchase is needed in morning, bid before Block 33 '
        f'(before 8:00 AM) when prices are still low.',
        f'Cost saving opportunity: Buying in Off-Peak instead of '
        f'Evening Peak saves approximately Rs {saving_opp:,.0f}/MWh.',
    ]
    rec_avoid = [
        f'MINIMIZE exchange purchase during Evening Peak '
        f'(5pm-9pm, Blocks 69-84). '
        f'Average price is Rs {evening_avg:,.0f}/MWh.',
        f'Use own generation (thermal/hydro) to cover '
        f'evening peak demand instead of buying from IEX.',
        f'If RTM purchase becomes unavoidable in peak hours, '
        f'keep it minimal — RTM price is typically even higher.',
    ]

    buy_rows  = [[Paragraph(f'  {r}', S_REC_GRN)] for r in rec_buy]
    buy_table = Table(buy_rows, colWidths=[15.7 * cm])
    buy_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), colors.HexColor('#e8f5e9')),
        ('BOX',           (0, 0), (-1, -1), 0.8, RL_GREEN),
        ('LEFTPADDING',   (0, 0), (-1, -1), 10),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(buy_table)
    story.append(Spacer(1, 8))

    avoid_rows  = [[Paragraph(f'  {r}', S_REC_RED)] for r in rec_avoid]
    avoid_table = Table(avoid_rows, colWidths=[15.7 * cm])
    avoid_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), colors.HexColor('#ffebee')),
        ('BOX',           (0, 0), (-1, -1), 0.8, RL_ORANGE),
        ('LEFTPADDING',   (0, 0), (-1, -1), 10),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(avoid_table)

    # ════════════════════════════════════════════════════════
    # PAGE 2 — CHARTS
    # ════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph('PRICE FORECAST CHARTS', S_H1))
    story.append(HRFlowable(width='100%', thickness=1,
                             color=RL_BLUE, spaceAfter=10))

    story.append(Paragraph(
        'Chart 1: Price for Each 15-Minute Block of the Day', S_H2))
    story.append(Image(chart1_path, width=17 * cm, height=6 * cm))
    story.append(Paragraph(
        'How to read this chart: Each bar is one 15-minute time block '
        '(Block 1 = midnight, Block 96 = 11:45pm). '
        'ORANGE bars = Morning Peak (8am-1pm, expensive). '
        'PURPLE bars = Evening Peak (5pm-9pm, most expensive). '
        'BLUE bars = Off-Peak (cheap — best buying opportunity).',
        S_CAPTION))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        'Chart 2: Average Price by Session (Simple Comparison)', S_H2))
    story.append(Image(chart2_path, width=10 * cm, height=5 * cm))
    story.append(Paragraph(
        'How to read this chart: Three bars showing average price per session. '
        'BLUE (Off-Peak) is cheapest — best time to buy from IEX. '
        'PURPLE (Evening Peak) is most expensive — avoid buying here.',
        S_CAPTION))

    story.append(Spacer(1, 10))
    explain_data = [[Paragraph(
        '<b>In Simple Terms:</b> Think of IEX electricity price like '
        'auto-rickshaw rates. Late night and early morning = meter rate '
        '(cheap). Office rush hours morning and evening = surge pricing '
        '(expensive). Our job is to buy maximum electricity during cheap '
        'hours and use our own generation during expensive hours — this '
        'directly reduces MP PPMCL\'s power purchase cost.',
        style('exp', fontSize=9,
               textColor=colors.HexColor('#1a1a2e'),
               fontName='Helvetica', leading=15))]]
    exp_tbl = Table(explain_data, colWidths=[15.7 * cm])
    exp_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), colors.HexColor('#e3f2fd')),
        ('BOX',           (0, 0), (-1, -1), 0.8, RL_BLUE),
        ('LEFTPADDING',   (0, 0), (-1, -1), 12),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 12),
        ('TOPPADDING',    (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(exp_tbl)

    # ════════════════════════════════════════════════════════
    # PAGE 3 — FULL 96-BLOCK TABLE
    # ════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph('COMPLETE 96-BLOCK PRICE TABLE', S_H1))
    story.append(Paragraph(
        'Full block-wise forecast for all 96 time slots (15 minutes each). '
        'Green rows = cheap buying opportunity. '
        'Orange rows = morning peak, expensive. '
        'Purple rows = evening peak, most expensive.',
        S_BODY))
    story.append(HRFlowable(width='100%', thickness=0.5,
                             color=RL_BLUE, spaceAfter=8))

    tbl_data = [['Block', 'Time', 'Forecasted Price\n(Rs/MWh)',
                  'Session', 'Action']]

    for _, row in valid.iterrows():
        block = int(row['Block_No'])
        time  = row['Time']
        mcp   = row['Predicted_MCP']
        flag  = str(row.get('Peak_Flag', 'Off-Peak'))

        if '🌅' in flag:
            session = 'Morning Peak'
            action  = 'AVOID'
        elif '🌆' in flag:
            session = 'Evening Peak'
            action  = 'AVOID'
        else:
            session = 'Off-Peak'
            action  = 'BUY'

        tbl_data.append([str(block), time,
                          f'Rs {mcp:,.0f}', session, action])

    block_tbl = Table(tbl_data,
                       colWidths=[1.6*cm, 2*cm, 4*cm, 3.8*cm, 2.5*cm],
                       repeatRows=1)

    tbl_style = [
        ('BACKGROUND',    (0, 0), (-1, 0), RL_DARK),
        ('TEXTCOLOR',     (0, 0), (-1, 0), RL_BLUE),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0), 8),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME',      (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',      (0, 1), (-1, -1), 7.5),
        ('GRID',          (0, 0), (-1, -1), 0.2,
         colors.HexColor('#dddddd')),
        ('TOPPADDING',    (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]

    for i, row_data in enumerate(tbl_data[1:], start=1):
        session_val = row_data[3]
        if session_val == 'Off-Peak':
            tbl_style.append(('BACKGROUND', (0, i), (-1, i),
                               colors.HexColor('#e8f5e9')))
            tbl_style.append(('TEXTCOLOR', (4, i), (4, i),
                               colors.HexColor('#1a5c2a')))
        elif session_val == 'Morning Peak':
            tbl_style.append(('BACKGROUND', (0, i), (-1, i),
                               colors.HexColor('#fff3e0')))
            tbl_style.append(('TEXTCOLOR', (4, i), (4, i),
                               colors.HexColor('#5c2a1a')))
        elif session_val == 'Evening Peak':
            tbl_style.append(('BACKGROUND', (0, i), (-1, i),
                               colors.HexColor('#f3e5f5')))
            tbl_style.append(('TEXTCOLOR', (4, i), (4, i),
                               colors.HexColor('#4a1a5c')))

    block_tbl.setStyle(TableStyle(tbl_style))
    story.append(block_tbl)

    # Footer
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width='100%', thickness=0.5,
                             color=RL_MUTED, spaceAfter=6))
    story.append(Paragraph(
        'Report generated by IEX DAM Price Forecasting Tool  |  '
        'MP Power Management Co. Ltd.  |  '
        'Model: LSTM Deep Learning  |  '
        'Target Accuracy: less than 10% error  |  '
        'For internal use only.',
        style('footer', fontSize=7, textColor=RL_MUTED,
               fontName='Helvetica', alignment=TA_CENTER)))

    # ── Build PDF ────────────────────────────────────────────
    doc.build(story)

    # Cleanup temp chart images
    for p in [chart1_path, chart2_path]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    print(f'\n✅ PDF Report saved → {output_path}')
    return output_path


# ============================================================
# STANDALONE TEST — run with synthetic data
# ============================================================

if __name__ == '__main__':
    print('Generating sample PDF report with synthetic data...\n')

    np.random.seed(42)
    records = []
    for block in range(1, 97):
        hour  = (block - 1) * 0.25
        daily = (300 * np.sin(np.pi * (hour - 6)  / 12) +
                 200 * np.sin(np.pi * (hour - 17) / 6))
        mcp   = max(1500, 3800 + daily + np.random.normal(0, 120))
        hh    = ((block - 1) * 15) // 60
        mm    = ((block - 1) * 15) % 60
        if 33 <= block <= 52:   flag = '🌅 Morning Peak'
        elif 69 <= block <= 84: flag = '🌆 Evening Peak'
        else:                   flag = 'Off-Peak'
        records.append({
            'Block_No':      block,
            'Time':          f'{hh:02d}:{mm:02d}',
            'Predicted_MCP': round(mcp, 2),
            'Peak_Flag':     flag,
        })

    df          = pd.DataFrame(records)
    report_path = generate_manager_report(df, '2025-04-24')
    print(f'Done! Open this file: {report_path}')
