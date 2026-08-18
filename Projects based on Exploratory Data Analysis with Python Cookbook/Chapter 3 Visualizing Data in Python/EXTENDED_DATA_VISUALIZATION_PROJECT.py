#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
EXTENDED DATA VISUALIZATION PROJECT — AMSTERDAM HOUSE PRICES
================================================================================
Project:       Exploratory Data Analysis with Python — Extended Edition
Source:        Based on Oluleye A., "Exploratory Data Analysis with Python
               Cookbook" (2023), Chapter 3 — "Visualizing Data in Python"
Dataset:       Amsterdam House Prices Data (Kaggle)
Author:        Extended Project Script
Date:          31 July 2026
Description:   A formal, report-style Python script covering data loading,
               inspection, feature engineering, and comparative visualisation
               across four major Python plotting libraries:
                   1. Matplotlib
                   2. Seaborn
                   3. GGPLOT (plotnine)
                   4. Bokeh
               Each section mirrors the cookbook recipes but is extended with
               additional charts, annotations, and a unified reporting layout.

Dependencies:
    pip install pandas matplotlib seaborn plotnine bokeh numpy

Usage:
    python eda_visualization_project.py

Output:
    All figures are saved to ./report_figures/ and displayed interactively.
================================================================================
"""

# ---------------------------------------------------------------------------
# SECTION 0 — IMPORTS & ENVIRONMENT SETUP
# ---------------------------------------------------------------------------

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# Suppress plotnine and bokeh warnings for clean report output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Create output directory for saved figures
OUTPUT_DIR = "report_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(filename, fig=None):
    """Save a matplotlib figure to the report directory."""
    path = os.path.join(OUTPUT_DIR, filename)
    if fig is None:
        fig = plt.gcf()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  ✓ Saved: {path}")


# ---------------------------------------------------------------------------
# SECTION 1 — DATA LOADING & INSPECTION
# ---------------------------------------------------------------------------

print("=" * 72)
print("SECTION 1 — DATA LOADING & INSPECTION")
print("=" * 72)

# Load the Amsterdam House Prices dataset
# Expected file: data/HousingPricesData.csv
# If unavailable, fall back to the GitHub repository copy.
DATA_PATH = "data/HousingPricesData.csv"
GITHUB_URL = (
    "https://raw.githubusercontent.com/PacktPublishing/"
    "Exploratory-Data-Analysis-with-Python-Cookbook/main/"
    "Chapter03/data/HousingPricesData.csv"
)

try:
    houseprices_data = pd.read_csv(DATA_PATH)
    print(f"  Loaded from local file: {DATA_PATH}")
except FileNotFoundError:
    print(f"  Local file not found. Fetching from GitHub repository…")
    houseprices_data = pd.read_csv(GITHUB_URL)
    print(f"  Loaded from GitHub: {GITHUB_URL}")

# Subset to relevant columns
houseprices_data = houseprices_data[["Zip", "Price", "Area", "Room"]]

# --- 1.1 Head ---
print("\n  First 5 rows:")
print(houseprices_data.head().to_string(index=False))

# --- 1.2 Shape ---
print(f"\n  Dataset shape: {houseprices_data.shape}")
print(f"  Rows: {houseprices_data.shape[0]}  |  Columns: {houseprices_data.shape[1]}")

# --- 1.3 Data types ---
print("\n  Data types:")
print(houseprices_data.dtypes.to_string())

# --- 1.4 Descriptive statistics ---
print("\n  Descriptive statistics:")
print(houseprices_data.describe().round(2).to_string())

# --- 1.5 Missing values check ---
missing = houseprices_data.isnull().sum()
print("\n  Missing values per column:")
print(missing.to_string())


# ---------------------------------------------------------------------------
# SECTION 2 — FEATURE ENGINEERING
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("SECTION 2 — FEATURE ENGINEERING")
print("=" * 72)

# Create PricePerSqm: a standard real-estate metric
houseprices_data["PriceperSqm"] = houseprices_data["Price"] / houseprices_data["Area"]

# Create PriceCategory for categorical analysis
houseprices_data["PriceCategory"] = pd.cut(
    houseprices_data["Price"],
    bins=[0, 500_000, 1_000_000, 2_000_000, 6_000_000],
    labels=["Budget", "Mid-range", "Premium", "Luxury"],
)

print("  Created features:")
print("    • PriceperSqm  — Price divided by Area (€/sqm)")
print("    • PriceCategory — Budget / Mid-range / Premium / Luxury")
print("\n  First 5 rows after engineering:")
print(houseprices_data.head().to_string(index=False))

# Sort for top-N visualisations
houseprices_sorted = houseprices_data.sort_values("Price", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# SECTION 3 — MATPLOTLIB VISUALISATIONS
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("SECTION 3 — MATPLOTLIB VISUALISATIONS")
print("=" * 72)

# ---- 3.1 Basic Bar Chart: Top 10 by Price ----
plt.figure(figsize=(12, 6))
x_mp = houseprices_sorted["Zip"][0:10]
y_mp = houseprices_sorted["Price"][0:10]
plt.bar(x_mp, y_mp, color="#6d4aff")
plt.title("Top 10 Areas with the Highest House Prices", fontsize=15, fontweight="bold")
plt.xlabel("Zip Code", fontsize=12)
plt.ylabel("House Price (€ millions)", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"€{v/1e6:.1f}M"))
plt.tight_layout()
save_fig("01_matplotlib_bar_top10_price.png")
plt.show()

# ---- 3.2 Subplots: Price vs PricePerSqm ----
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Left subplot — Price
axes[0].bar(x_mp, y_mp, color="#6d4aff")
axes[0].set_title("Top 10 Areas by House Price", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Zip Code", fontsize=12)
axes[0].set_ylabel("Price (€ millions)", fontsize=12)
axes[0].tick_params(axis="x", rotation=45)

# Right subplot — Price per Sqm
y1_mp = houseprices_sorted["PriceperSqm"][0:10]
axes[1].bar(x_mp, y1_mp, color="#ff6d4a")
axes[1].set_title("Top 10 Areas by Price per Sqm", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Zip Code", fontsize=12)
axes[1].set_ylabel("Price per Sqm (€)", fontsize=12)
axes[1].tick_params(axis="x", rotation=45)

fig.suptitle("Matplotlib Subplots — Price Comparison (Extended)", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig("02_matplotlib_subplots_price_vs_persqm.png")
plt.show()

# ---- 3.3 Histogram: Distribution of House Prices ----
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(houseprices_data["Price"] / 1e6, bins=40, color="#6d4aff", edgecolor="white", alpha=0.8)
ax.set_title("Distribution of House Prices in Amsterdam", fontsize=15, fontweight="bold")
ax.set_xlabel("Price (€ millions)", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.axvline(houseprices_data["Price"].mean() / 1e6, color="red", linestyle="--", linewidth=2, label=f"Mean: €{houseprices_data['Price'].mean()/1e6:.2f}M")
ax.axvline(houseprices_data["Price"].median() / 1e6, color="orange", linestyle="-.", linewidth=2, label=f"Median: €{houseprices_data['Price'].median()/1e6:.2f}M")
ax.legend(fontsize=11)
plt.tight_layout()
save_fig("03_matplotlib_histogram_prices.png")
plt.show()

# ---- 3.4 Scatter Plot: Price vs Area ----
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(
    houseprices_data["Area"],
    houseprices_data["Price"] / 1e6,
    c=houseprices_data["Room"],
    cmap="Purples",
    s=60,
    alpha=0.7,
    edgecolors="grey",
    linewidths=0.5,
)
ax.set_title("House Price vs Area (coloured by Room Count)", fontsize=15, fontweight="bold")
ax.set_xlabel("Area (sqm)", fontsize=12)
ax.set_ylabel("Price (€ millions)", fontsize=12)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Number of Rooms", fontsize=11)
plt.tight_layout()
save_fig("04_matplotlib_scatter_price_area.png")
plt.show()

# ---- 3.5 Box Plot: Price by Room Count ----
fig, ax = plt.subplots(figsize=(14, 7))
room_groups = houseprices_data.groupby("Room")["Price"].apply(list).sort_index()
ax.boxplot(room_groups, labels=room_groups.index, patch_artist=True,
           boxprops=dict(facecolor="#6d4aff", alpha=0.6),
           medianprops=dict(color="red", linewidth=2))
ax.set_title("House Price Distribution by Room Count", fontsize=15, fontweight="bold")
ax.set_xlabel("Number of Rooms", fontsize=12)
ax.set_ylabel("Price (€)", fontsize=12)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"€{v/1e6:.1f}M"))
plt.tight_layout()
save_fig("05_matplotlib_boxplot_rooms.png")
plt.show()


# ---------------------------------------------------------------------------
# SECTION 4 — SEABORN VISUALISATIONS
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("SECTION 4 — SEABORN VISUALISATIONS")
print("=" * 72)

# Apply a clean seaborn theme
sns.set_theme(style="whitegrid", font_scale=1.1)

# ---- 4.1 Bar Chart with Error Bars ----
plt.figure(figsize=(12, 6))
data_sb = houseprices_sorted[0:10]
ax = sns.barplot(data=data_sb, x="Zip", y="Price", color="#6d4aff")
ax.set_title("Top 10 Areas — House Prices (Seaborn)", fontsize=15, fontweight="bold")
ax.set_xlabel("Zip Code", fontsize=12)
ax.set_ylabel("House Price (€ millions)", fontsize=12)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"€{v/1e6:.1f}M"))
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
save_fig("06_seaborn_bar_top10.png")
plt.show()

# ---- 4.2 Subplots: Price + PricePerSqm ----
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
sns.barplot(data=data_sb, x="Zip", y="Price", ax=axes[0], color="#6d4aff")
axes[0].set_title("Top 10 by Price", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Zip Code")
axes[0].set_ylabel("Price (€)")
axes[0].tick_params(axis="x", rotation=45)

sns.barplot(data=data_sb, x="Zip", y="PriceperSqm", ax=axes[1], color="#ff6d4a")
axes[1].set_title("Top 10 by Price per Sqm", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Zip Code")
axes[1].set_ylabel("Price per Sqm (€)")
axes[1].tick_params(axis="x", rotation=45)

fig.suptitle("Seaborn Subplots — Dual Perspective (Extended)", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig("07_seaborn_subplots.png")
plt.show()

# ---- 4.3 Regression Plot: Price vs Area ----
fig, ax = plt.subplots(figsize=(12, 8))
sns.regplot(
    data=houseprices_data,
    x="Area",
    y="Price",
    scatter_kws={"s": 50, "alpha": 0.5, "color": "#6d4aff"},
    line_kws={"color": "red", "linewidth": 2},
    ax=ax,
)
ax.set_title("Regression: Price vs Area (Seaborn)", fontsize=15, fontweight="bold")
ax.set_xlabel("Area (sqm)", fontsize=12)
ax.set_ylabel("Price (€)", fontsize=12)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"€{v/1e6:.1f}M"))
plt.tight_layout()
save_fig("08_seaborn_regplot.png")
plt.show()

# ---- 4.4 Box Plot: Price by Price Category ----
fig, ax = plt.subplots(figsize=(12, 7))
order = ["Budget", "Mid-range", "Premium", "Luxury"]
sns.boxplot(
    data=houseprices_data,
    x="PriceCategory",
    y="PriceperSqm",
    order=order,
    palette="Purples",
    ax=ax,
)
ax.set_title("Price per Sqm by Category (Seaborn)", fontsize=15, fontweight="bold")
ax.set_xlabel("Price Category", fontsize=12)
ax.set_ylabel("Price per Sqm (€)", fontsize=12)
plt.tight_layout()
save_fig("09_seaborn_box_category.png")
plt.show()

# ---- 4.5 Pairplot: Multivariate EDA ----
pairplot = sns.pairplot(
    houseprices_data[["Price", "Area", "Room", "PriceperSqm"]],
    diag_kind="kde",
    plot_kws={"alpha": 0.5, "s": 30, "color": "#6d4aff"},
)
pairplot.fig.suptitle("Pairplot — Multivariate Overview (Extended)", fontsize=16, fontweight="bold", y=1.02)
save_fig("10_seaborn_pairplot.png", pairplot.fig)
plt.show()

# ---- 4.6 Correlation Heatmap ----
fig, ax = plt.subplots(figsize=(10, 8))
corr = houseprices_data[["Price", "Area", "Room", "PriceperSqm"]].corr()
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="Purples",
    square=True,
    linewidths=0.5,
    ax=ax,
    vmin=-1,
    vmax=1,
)
ax.set_title("Correlation Heatmap (Seaborn)", fontsize=15, fontweight="bold")
plt.tight_layout()
save_fig("11_seaborn_heatmap.png")
plt.show()


# ---------------------------------------------------------------------------
# SECTION 5 — GGPLOT (PLOTNINE) VISUALISATIONS
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("SECTION 5 — GGPLOT (PLOTNINE) VISUALISATIONS")
print("=" * 72)

from plotnine import (
    ggplot, aes, geom_bar, geom_point, geom_histogram,
    geom_boxplot, labs, theme, element_text, scale_x_discrete,
    scale_fill_manual, facet_wrap, ggtitle,
)

chart_data = houseprices_sorted[0:10].copy()

# ---- 5.1 Basic GGPLOT Bar Chart ----
p_gg1 = (
    ggplot(chart_data, aes(x="Zip", y="Price"))
    + geom_bar(stat="identity", fill="#6d4aff")
    + scale_x_discrete(limits=chart_data["Zip"].tolist())
    + labs(
        title="Top 10 Areas by House Price (GGPLOT)",
        x="Zip Code",
        y="House Price (€)",
    )
    + theme(
        figure_size=(14, 7),
        axis_title=element_text(face="bold", size=12),
        axis_text=element_text(size=10),
        plot_title=element_text(face="bold", size=14),
    )
)
p_gg1.save(os.path.join(OUTPUT_DIR, "12_ggplot_bar_top10_price.png"), dpi=200)
print("  ✓ Saved: 12_ggplot_bar_top10_price.png")
print(p_gg1)

# ---- 5.2 GGPLOT Scatter — Price vs Area ----
p_gg2 = (
    ggplot(houseprices_data, aes(x="Area", y="Price", color="factor(Room)"))
    + geom_point(alpha=0.6, size=3)
    + labs(
        title="Price vs Area by Room Count (GGPLOT)",
        x="Area (sqm)",
        y="Price (€)",
        color="Rooms",
    )
    + theme(
        figure_size=(14, 8),
        axis_title=element_text(face="bold", size=12),
        axis_text=element_text(size=10),
        plot_title=element_text(face="bold", size=14),
        legend_position="right",
    )
)
p_gg2.save(os.path.join(OUTPUT_DIR, "13_ggplot_scatter_price_area.png"), dpi=200)
print("  ✓ Saved: 13_ggplot_scatter_price_area.png")
print(p_gg2)

# ---- 5.3 GGPLOT Histogram — Price Distribution ----
p_gg3 = (
    ggplot(houseprices_data, aes(x="Price"))
    + geom_histogram(bins=40, fill="#6d4aff", color="white", alpha=0.8)
    + labs(
        title="Distribution of House Prices (GGPLOT)",
        x="House Price (€)",
        y="Frequency",
    )
    + theme(
        figure_size=(14, 7),
        axis_title=element_text(face="bold", size=12),
        axis_text=element_text(size=10),
        plot_title=element_text(face="bold", size=14),
    )
)
p_gg3.save(os.path.join(OUTPUT_DIR, "14_ggplot_histogram_price.png"), dpi=200)
print("  ✓ Saved: 14_ggplot_histogram_price.png")
print(p_gg3)

# ---- 5.4 GGPLOT Boxplot — PricePerSqm by Room ----
p_gg4 = (
    ggplot(houseprices_data, aes(x="factor(Room)", y="PriceperSqm"))
    + geom_boxplot(fill="#6d4aff", alpha=0.6)
    + labs(
        title="Price per Sqm by Room Count (GGPLOT)",
        x="Number of Rooms",
        y="Price per Sqm (€)",
    )
    + theme(
        figure_size=(14, 7),
        axis_title=element_text(face="bold", size=12),
        axis_text=element_text(size=10),
        plot_title=element_text(face="bold", size=14),
    )
)
p_gg4.save(os.path.join(OUTPUT_DIR, "15_ggplot_boxplot_persqm.png"), dpi=200)
print("  ✓ Saved: 15_ggplot_boxplot_persqm.png")
print(p_gg4)


# ---------------------------------------------------------------------------
# SECTION 6 — BOKEH VISUALISATIONS
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("SECTION 6 — BOKEH VISUALISATIONS")
print("=" * 72)

from bokeh.plotting import figure, show, output_file, save as bk_save
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter
from bokeh.layouts import gridplot, row
from bokeh.io import output_notebook

output_notebook()

data_bk = houseprices_sorted[0:10].copy()
source = ColumnDataSource(data_bk)

# ---- 6.1 Basic Bokeh Bar Chart ----
output_file(os.path.join(OUTPUT_DIR, "16_bokeh_bar_top10.html"))

p1 = figure(
    x_range=list(data_bk["Zip"]),
    width=700,
    height=500,
    title="Top 10 Areas with the Highest House Prices",
    x_axis_label="Zip Code",
    y_axis_label="House Price (€)",
    tools="pan,box_zoom,wheel_zoom,reset,save",
)
p1.vbar(x="Zip", top="Price", width=0.9, source=source, fill_color="#6d4aff")
p1.yaxis.formatter = NumeralTickFormatter(format="€0,0")
p1.title.text_font_size = "15pt"
p1.xaxis.axis_label_text_font_size = "12pt"
p1.yaxis.axis_label_text_font_size = "12pt"
bk_save(p1)
print("  ✓ Saved: 16_bokeh_bar_top10.html")
show(p1)

# ---- 6.2 Bokeh Interactive Scatter — Price vs Area ----
output_file(os.path.join(OUTPUT_DIR, "17_bokeh_scatter.html"))

scatter_source = ColumnDataSource(houseprices_data)

hover = HoverTool(
    tooltips=[
        ("Zip", "@Zip"),
        ("Price", "€@{Price}{0,0}"),
        ("Area", "@Area{0} sqm"),
        ("Rooms", "@Room"),
        ("Price/Sqm", "€@{PriceperSqm}{0,0}"),
    ]
)

p_scatter = figure(
    width=750,
    height=550,
    title="House Price vs Area (Interactive)",
    x_axis_label="Area (sqm)",
    y_axis_label="Price (€)",
    tools=[hover, "pan", "box_zoom", "wheel_zoom", "reset", "save"],
)
p_scatter.circle(
    x="Area",
    y="Price",
    source=scatter_source,
    size=8,
    fill_color="#6d4aff",
    fill_alpha=0.6,
    line_color="grey",
    line_width=0.5,
)
p_scatter.yaxis.formatter = NumeralTickFormatter(format="€0,0")
p_scatter.title.text_font_size = "15pt"
bk_save(p_scatter)
print("  ✓ Saved: 17_bokeh_scatter.html")
show(p_scatter)

# ---- 6.3 Bokeh Subplots — Price + PricePerSqm ----
output_file(os.path.join(OUTPUT_DIR, "18_bokeh_subplots.html"))

# Left: Price
pl = figure(
    x_range=list(data_bk["Zip"]),
    width=480,
    height=400,
    title="Top 10 by Price",
    x_axis_label="Zip Code",
    y_axis_label="Price (€)",
    tools="pan,box_zoom,wheel_zoom,reset",
)
pl.vbar(x="Zip", top="Price", width=0.9, source=source, fill_color="#6d4aff")
pl.yaxis.formatter = NumeralTickFormatter(format="€0,0")
pl.title.text_font_size = "13pt"

# Right: PricePerSqm
pr = figure(
    x_range=list(data_bk["Zip"]),
    width=480,
    height=400,
    title="Top 10 by Price per Sqm",
    x_axis_label="Zip Code",
    y_axis_label="Price/Sqm (€)",
    tools="pan,box_zoom,wheel_zoom,reset",
)
pr.vbar(x="Zip", top="PriceperSqm", width=0.9, source=source, fill_color="#ff6d4a")
pr.yaxis.formatter = NumeralTickFormatter(format="€0,0")
pr.title.text_font_size = "13pt"

gp = gridplot(children=[[pl, pr]], sizing_mode="fixed")
bk_save(gp)
print("  ✓ Saved: 18_bokeh_subplots.html")
show(gp)

# ---- 6.4 Bokeh Histogram — Price Distribution ----
output_file(os.path.join(OUTPUT_DIR, "19_bokeh_histogram.html"))

hist, edges = np.histogram(houseprices_data["Price"], bins=40)
p_hist = figure(
    width=750,
    height=450,
    title="Distribution of House Prices (Bokeh)",
    x_axis_label="Price (€)",
    y_axis_label="Frequency",
    tools="pan,box_zoom,wheel_zoom,reset,save",
)
p_hist.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="#6d4aff", line_color="white", alpha=0.8)
p_hist.xaxis.formatter = NumeralTickFormatter(format="€0,0")
p_hist.title.text_font_size = "15pt"
bk_save(p_hist)
print("  ✓ Saved: 19_bokeh_histogram.html")
show(p_hist)


# ---------------------------------------------------------------------------
# SECTION 7 — SUMMARY REPORT
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("SECTION 7 — SUMMARY REPORT")
print("=" * 72)

summary_table = pd.DataFrame({
    "Library":     ["Matplotlib", "Seaborn", "GGPLOT (plotnine)", "Bokeh"],
    "Charts Produced": [5, 6, 4, 4],
    "Key Strength": [
        "Maximum flexibility, low-level control",
        "Statistical graphics, tight pandas integration",
        "Grammar of Graphics, elegant aesthetics",
        "Interactive plots, web embedding",
    ],
    "Use Case": [
        "Publication-quality static figures",
        "Fast statistical EDA",
        "Declarative, layered plotting",
        "Dashboards & exploratory interaction",
    ],
})

print(summary_table.to_string(index=False))

print(f"\n  Total figures saved to: ./{OUTPUT_DIR}/")
print(f"  Total charts generated: {summary_table['Charts Produced'].sum()}")

print("\n" + "=" * 72)
print("  PROJECT COMPLETE — ALL VISUALISATIONS RENDERED SUCCESSFULLY")
print("=" * 72)