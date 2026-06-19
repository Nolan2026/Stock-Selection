
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import os, tempfile
from datetime import datetime
from pathlib import Path

def create_portfolio_report(data: dict, format: str = "pdf") -> str:
    """
    Generates a premium dark-mode portfolio report that matches the UI dashboard.
    Returns the absolute path to the generated file.
    """
    import matplotlib
    matplotlib.use("Agg")
    
    # --- UI Design System ---
    BG="#0a0f1a"; PAN="#0d1525"; GRD="#1a2535"
    TEXT="#c8d6e8"; TEXT_DIM="#6b7280"
    TEAL="#26c6da"; GOLD="#ffd740"; RED="#ef9a9a"; GREEN="#34d399"
    PURPLE="#a78bfa"
    
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": PAN,
        "axes.edgecolor": GRD,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "text.color": TEXT,
        "grid.color": GRD,
        "font.family": "sans-serif", # Use sans-serif for a cleaner look like the UI
        "figure.dpi": 150,
    })

    pm = data.get("portfolio_metrics", {})
    results = data.get("results", [])
    rebalance = data.get("rebalance_flags", [])
    
    # Calculate dynamic height
    num_holdings = len(results)
    fig_height = max(14, 10 + num_holdings * 0.5)
    fig = plt.figure(figsize=(16, fig_height), facecolor=BG)
    
    # Layout Grid: Header (0), Metrics Row 1 (1), Metrics Row 2 (2), Charts (3), Table (4)
    # 5 rows and 12 columns for dynamic scaling without overlap
    gs = gridspec.GridSpec(5, 12, fig, hspace=0.45, wspace=0.25,
                             top=0.95, bottom=0.02, left=0.05, right=0.95,
                             height_ratios=[0.6, 0.7, 0.7, 2.5, max(4, num_holdings * 0.4)])

    # --- 1. HEADER ---
    ax_head = fig.add_subplot(gs[0, :])
    ax_head.axis("off")
    portfolio_name = data.get("name")
    heading_text = f"NSE SIGNAL — {portfolio_name.upper()}" if portfolio_name else "NSE SIGNAL — PORTFOLIO DASHBOARD"
    ax_head.text(0.0, 0.6, heading_text, 
                 ha="left", va="center", fontsize=22, fontweight="bold", color=TEAL)
    ax_head.text(0.0, 0.2, f"LIVE ANALYTICS | {datetime.now().strftime('%d %b %Y, %H:%M')}", 
                 ha="left", va="center", fontsize=10, color=TEXT_DIM)

    # --- 2. METRIC CARDS (Matching UI layout, dynamic sizing) ---
    card_data = [
        ("TOTAL VALUE", f"₹{pm.get('total_value', 0):,.0f}", TEAL),
        ("TOTAL P&L", f"₹{pm.get('total_pnl_abs', 0):,.0f}", GREEN if pm.get('overall_pnl_pct',0)>=0 else RED),
        ("RETURN %", f"{pm.get('overall_pnl_pct', 0):+.2f}%", GREEN if pm.get('overall_pnl_pct',0)>=0 else RED),
        ("WEIGHTED BETA", f"{pm.get('weighted_beta', 0):.2f}", RED if pm.get('weighted_beta',0)>1.2 else (GOLD if pm.get('weighted_beta',0)>0.8 else GREEN)),
        ("BETA REGIME", pm.get("beta_regime", "NORMAL"), RED if pm.get('weighted_beta',0)>1.2 else (GOLD if pm.get('weighted_beta',0)>0.8 else GREEN)),
        ("AGG MARGIN OF SAFETY", f"{pm.get('agg_margin_of_safety', 0):.1f}%", GREEN if pm.get('agg_margin_of_safety',0)>=10 else GOLD),
        ("HOLDINGS", f"{len(results)}", TEXT)
    ]
    
    # Render 7 cards in a flexible grid inside the gridspec
    for i, (label, val, clr) in enumerate(card_data):
        row = i // 4
        col = i % 4
        
        # Each card spans 3 columns out of 12
        ax = fig.add_subplot(gs[1 + row, col*3 : (col+1)*3])
        ax.set_facecolor(PAN)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.08, 0.7, label, ha="left", va="center", fontsize=8, color=TEXT_DIM, fontweight="bold", transform=ax.transAxes)
        ax.text(0.08, 0.3, val, ha="left", va="center", fontsize=16, color=clr, fontweight="bold", transform=ax.transAxes)
        for spine in ax.spines.values(): spine.set_edgecolor(GRD); spine.set_linewidth(1)

    # --- 3. CHARTS ---
    # A. Sector Exposure (Matching UI)
    ax_sec = fig.add_subplot(gs[3, 0:4]) # Left column (columns 0-4)
    sectors = sorted(pm.get("sector_exposure", {}).items(), key=lambda x: x[1])
    if sectors:
        names = [x[0] for x in sectors]
        vals = [x[1] for x in sectors]
        bars = ax_sec.barh(names, vals, color=TEAL, alpha=0.8, height=0.4)
        ax_sec.set_title("  SECTOR EXPOSURE", fontsize=11, fontweight="bold", pad=20, loc="left", color=PURPLE)
        ax_sec.set_xlim(0, 100)
        ax_sec.grid(False)
        ax_sec.tick_params(axis='y', length=0, pad=10)
        for i, bar in enumerate(bars):
            ax_sec.text(98, bar.get_y() + bar.get_height()/2, 
                        f"{vals[i]:.1f}%", va="center", ha="right", color=TEXT, fontsize=9, fontweight="bold")
        for spine in ax_sec.spines.values(): spine.set_visible(False)
    else:
        ax_sec.axis("off")

    # B. Rebalancing Alerts (Matching UI Red Box)
    ax_reb = fig.add_subplot(gs[3, 4:12]) # Right columns (columns 4-12)
    ax_reb.set_facecolor(PAN)
    ax_reb.set_xticks([]); ax_reb.set_yticks([])
    ax_reb.set_title("  REBALANCING ALERTS", fontsize=11, fontweight="bold", pad=20, loc="left", color=RED)
    
    if rebalance:
        # Draw the red-tinted background box
        rect = plt.Rectangle((0.05, 0.1), 0.9, 0.75, transform=ax_reb.transAxes, 
                             color=RED, alpha=0.05, zorder=1)
        ax_reb.add_patch(rect)
        # Draw red border
        border = plt.Rectangle((0.05, 0.1), 0.9, 0.75, transform=ax_reb.transAxes, 
                               fill=False, edgecolor=RED, alpha=0.3, lw=1, zorder=2)
        ax_reb.add_patch(border)
        
        y_pos = 0.7
        for r in rebalance[:3]:
            # Action Tag
            ax_reb.text(0.08, y_pos, f" {r['action']} ", ha="left", va="center", transform=ax_reb.transAxes,
                        color="white", fontsize=8, fontweight="bold", 
                        bbox=dict(facecolor=RED, edgecolor="none", boxstyle="round,pad=0.3"))
            # Symbol
            ax_reb.text(0.22, y_pos, r['symbol'], ha="left", va="center", transform=ax_reb.transAxes,
                        color=TEXT, fontsize=12, fontweight="bold")
            # Reason
            ax_reb.text(0.08, y_pos - 0.12, f"Signal dropped to {r['action'].replace('ROTATE OUT','AVOID')} | P&L: {r['pnl_pct']:+.2f}%", 
                        ha="left", va="center", transform=ax_reb.transAxes, color=TEXT_DIM, fontsize=9)
            # Suggestion
            if r.get('rotate_into'):
                ax_reb.text(0.08, y_pos - 0.22, f"Suggested rotation into: {', '.join(r['rotate_into'])}", 
                            ha="left", va="center", transform=ax_reb.transAxes, color=GREEN, fontsize=8, fontweight="bold")
            y_pos -= 0.4
    else:
        ax_reb.text(0.5, 0.5, "✓ Portfolio is balanced. All signals are stable.", 
                    ha="center", va="center", transform=ax_reb.transAxes, color=GREEN, fontsize=10)
    for spine in ax_reb.spines.values(): spine.set_edgecolor(GRD)

    # --- 4. HOLDINGS TABLE ---
    ax_tab = fig.add_subplot(gs[4, :])
    ax_tab.axis("off")
    ax_tab.set_title("  LIVE HOLDINGS ANALYSIS", fontsize=11, fontweight="bold", pad=20, loc="left", color=GOLD)
    
    headers = ["SYMBOL", "ENTRY PRICE", "PRICE", "P&L %", "P&L ₹", "TODAY P&L", "GAIN/SHARE", "VALUE", "SIGNAL", "P(TARGET)", "MOS %", "ACTION", "BETA"]
    table_data = []
    cell_colors = []
    
    for r in results:
        pnl_clr = GREEN if r["pnl_pct"] >= 0 else RED
        sig_clr = GREEN if "BUY" in r["signal"] else (GOLD if r["signal"] == "WATCH" else RED)
        
        # Calculate Gain/Share
        diff_val = r["current_price"] - r["avg_cost"]
        diff_clr = GREEN if diff_val >= 0 else RED
        diff_sign = "+" if diff_val >= 0 else ""
        
        # Today's P&L
        today_pnl_val = r.get("today_pnl_abs", 0.0)
        today_pnl_pct = r.get("today_pnl_pct", 0.0)
        today_pnl_clr = GREEN if today_pnl_val >= 0 else RED
        today_pnl_sign = "+" if today_pnl_val >= 0 else ""
        today_pnl_str = f"{today_pnl_sign}₹{today_pnl_val:,.0f}\n({today_pnl_pct:+.2f}%)"
        
        # Star marker for favorites in PDF/Image
        sym_label = f"* {r['symbol']}" if r.get("favorite") else r["symbol"]
        
        row = [
            sym_label,
            f"₹{r['avg_cost']:,.2f}",
            f"₹{r['current_price']:,.2f}",
            f"{r['pnl_pct']:+.2f}%",
            f"₹{r['pnl_abs']:,.0f}",
            today_pnl_str,
            f"{diff_sign}₹{diff_val:,.2f}",
            f"₹{r['current_value']:,.0f}",
            r["signal"],
            f"{r.get('prob_hit_target','—')}%" if r.get("prob_hit_target") else "—",
            f"{r.get('margin_of_safety', 0):.2f}%",
            r.get("rebalance_flag", "HOLD").replace("_", " "),
            f"{r.get('beta', 1.0):.2f}",
        ]
        table_data.append(row)
        
        # UI-specific cell coloring (matching headers)
        mos_clr = GREEN if r.get('margin_of_safety', 0) > 10 else (GOLD if r.get('margin_of_safety', 0) > 0 else RED)
        
        colors = [
            TEAL,        # Symbol
            TEXT_DIM,    # Entry Price (Avg Cost)
            TEXT,        # Price
            pnl_clr,     # P&L %
            pnl_clr,     # P&L ₹
            today_pnl_clr, # TODAY P&L
            diff_clr,    # Gain/Share
            TEXT,        # Value
            sig_clr,     # Signal
            TEXT,        # P(Target)
            mos_clr,     # MOS %
            GOLD,        # Action
            TEXT,        # Beta
        ]
        cell_colors.append(colors)

    if table_data:
        table = ax_tab.table(cellText=table_data, colLabels=headers, 
                             loc="upper center", cellLoc="center",
                             colColours=[GRD]*len(headers))
        
        table.auto_set_font_size(False)
        table.set_fontsize(7.0) # Slightly smaller font since we have more columns
        table.scale(1, 2.2)
        
        # Style the table to match UI exactly
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(GRD)
            cell.set_linewidth(0.5)
            if row == 0:
                cell.set_text_props(weight="bold", color=TEXT_DIM, fontsize=7)
                cell.set_facecolor(BG) # Header background matches main BG
            else:
                cell.set_facecolor(PAN)
                cell.set_text_props(color=cell_colors[row-1][col], fontweight="bold" if col == 0 else "normal")

    # --- SAVE ---
    tmp_dir = Path(tempfile.gettempdir())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == "pdf":
        out_path = tmp_dir / f"NSE_Portfolio_Report_{timestamp}.pdf"
        fig.savefig(out_path, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        return str(out_path)
    else:
        out_path = tmp_dir / f"NSE_Portfolio_Report_{timestamp}.jpg"
        # High DPI for JPG to make it look "premium"
        fig.savefig(out_path, bbox_inches="tight", facecolor=BG, dpi=200)
        plt.close(fig)
        return str(out_path)

