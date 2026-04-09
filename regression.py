# =============================================================================
#  NUMERICAL METHODS — LINEAR REGRESSION FROM SCRATCH
#  Dataset  : Housing.csv
#  Features : X = Avg. Area Income   |   Y = Price
#  Subject  : Numerical Methods
#  Note     : NO sklearn / NO ML libraries used.
#             All regression performed using pure statistical formulas.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from collections import Counter

# -----------------------------------------------------------------------------
# MATPLOTLIB STYLE SETUP
# Dark academic theme — clear for academic submission
# -----------------------------------------------------------------------------
plt.rcParams.update({
    'figure.facecolor'  : '#0F1117',
    'axes.facecolor'    : '#1A1D27',
    'axes.edgecolor'    : '#3A3F52',
    'axes.labelcolor'   : '#C8D0E7',
    'xtick.color'       : '#8A91A8',
    'ytick.color'       : '#8A91A8',
    'text.color'        : '#C8D0E7',
    'grid.color'        : '#2A2D3E',
    'grid.linestyle'    : '--',
    'grid.alpha'        : 0.6,
    'font.family'       : 'DejaVu Sans',
    'axes.titlesize'    : 12,
    'axes.labelsize'    : 10,
    'legend.fontsize'   : 9,
    'legend.framealpha' : 0.4,
})

# Colour palette
C_BLUE   = '#7C83FD'   # scatter / main data
C_RED    = '#FF6B6B'   # regression line / error
C_GREEN  = '#4ECDC4'   # saturation curve
C_GOLD   = '#FFD700'   # titles / highlights
C_ORANGE = '#FFA552'   # histogram bars
C_PURPLE = '#B48EF5'   # residual line


# =============================================================================
# STEP 1 — LOAD & CLEAN DATA
# =============================================================================
print("=" * 65)
print("  NUMERICAL METHODS — LINEAR REGRESSION")
print("=" * 65)

# --- 1.1  Load raw CSV ---------------------------------------------------------
raw_df = pd.read_csv('Housing.csv')
print(f"\n[1.1] Raw dataset loaded : {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
print(f"      Columns : {list(raw_df.columns)}")

# --- 1.2  Drop non-numeric 'Address' column -----------------------------------
#  'Address' is a string column — it has no mathematical meaning for regression.
df = raw_df.drop(columns=['Address'])
print(f"\n[1.2] Dropped 'Address' column. Remaining columns: {list(df.columns)}")

# --- 1.3  Select only X and Y ------------------------------------------------
#  X = Avg. Area Income  (independent / predictor variable)
#  Y = Price             (dependent / response variable)
X_raw = df['Avg. Area Income'].astype(float).values
Y_raw = df['Price'].astype(float).values
print(f"\n[1.3] Selected X = 'Avg. Area Income'  |  Y = 'Price'")
print(f"      Samples before cleaning : {len(X_raw)}")

# --- 1.4  Z-score Outlier Removal (Numerical Method) -------------------------
#
#  Z-score formula:
#       Z = (x - μ) / σ
#
#  Any sample where |Z_x| > 3 OR |Z_y| > 3 is considered an outlier
#  (lies beyond 3 standard deviations from the mean — ~0.3% of normal data).
#  This is a purely statistical / numerical cleaning method.

def z_scores(arr):
    """Compute Z-score for every element in arr using the formula Z=(x-μ)/σ."""
    mu    = sum(arr) / len(arr)                          # population mean
    sigma = math.sqrt(sum((xi - mu)**2 for xi in arr) / len(arr))  # population σ
    return np.array([(xi - mu) / sigma for xi in arr])

Z_x = z_scores(X_raw)
Z_y = z_scores(Y_raw)

# Keep only rows where both |Z_x| ≤ 3 and |Z_y| ≤ 3
mask = (np.abs(Z_x) <= 3) & (np.abs(Z_y) <= 3)
X    = X_raw[mask]
Y    = Y_raw[mask]

removed = len(X_raw) - len(X)
print(f"\n[1.4] Z-score Outlier Removal (|Z| > 3 threshold)")
print(f"      Outliers removed : {removed}")
print(f"      Samples after cleaning : {len(X)}")


# =============================================================================
# STEP 2 — STATISTICAL CALCULATIONS  (all from scratch — no numpy stats used)
# =============================================================================
print("\n" + "=" * 65)
print("  STEP 2 — STATISTICAL CALCULATIONS")
print("=" * 65)

n = len(X)   # number of clean samples

# ── 2.1  Mean -----------------------------------------------------------------
#  Formula: μ = (1/n) Σ xᵢ
def calc_mean(arr):
    return sum(arr) / len(arr)

mean_x = calc_mean(X)
mean_y = calc_mean(Y)
print(f"\n[2.1] Mean")
print(f"      μ_X = {mean_x:,.4f}")
print(f"      μ_Y = {mean_y:,.4f}")

# ── 2.2  Median ---------------------------------------------------------------
#  Formula: middle value of sorted array (average of two middle values if even)
def calc_median(arr):
    s = sorted(arr)
    m = len(s)
    if m % 2 == 1:
        return s[m // 2]
    else:
        return (s[m // 2 - 1] + s[m // 2]) / 2.0

median_x = calc_median(X)
median_y = calc_median(Y)
print(f"\n[2.2] Median")
print(f"      Median_X = {median_x:,.4f}")
print(f"      Median_Y = {median_y:,.4f}")

# ── 2.3  Mode -----------------------------------------------------------------
#  Formula: most frequently occurring value.
#  For continuous data we round to nearest 1000 (Y) or 100 (X) before counting.
def calc_mode(arr, round_to=1):
    rounded   = [round(v / round_to) * round_to for v in arr]
    frequency = Counter(rounded)
    return frequency.most_common(1)[0][0]

mode_x = calc_mode(X, round_to=100)
mode_y = calc_mode(Y, round_to=1000)
print(f"\n[2.3] Mode  (rounded to nearest bin for continuous data)")
print(f"      Mode_X ≈ {mode_x:,.2f}")
print(f"      Mode_Y ≈ {mode_y:,.2f}")

# ── 2.4  Variance & Standard Deviation ----------------------------------------
#  Population Variance : σ² = (1/n) Σ (xᵢ - μ)²
#  Population Std Dev  : σ  = √σ²
def calc_variance(arr, mu):
    return sum((xi - mu)**2 for xi in arr) / len(arr)

def calc_std(arr, mu):
    return math.sqrt(calc_variance(arr, mu))

var_x  = calc_variance(X, mean_x)
var_y  = calc_variance(Y, mean_y)
std_x  = calc_std(X, mean_x)
std_y  = calc_std(Y, mean_y)

print(f"\n[2.4] Variance & Standard Deviation")
print(f"      Var(X)  = {var_x:,.4f}    Std(X)  = {std_x:,.4f}")
print(f"      Var(Y)  = {var_y:,.4f}    Std(Y)  = {std_y:,.4f}")

# ── 2.5  Deviation from Mean --------------------------------------------------
#  Deviation of each point from its mean: dᵢ = xᵢ - μ
deviation_x = X - mean_x     # numpy vectorised subtraction
deviation_y = Y - mean_y
print(f"\n[2.5] Deviation from Mean (first 5 samples shown)")
print(f"      X deviations : {deviation_x[:5].round(2)}")
print(f"      Y deviations : {deviation_y[:5].round(2)}")


# =============================================================================
# STEP 3 — CORRELATION & REGRESSION COEFFICIENTS
# =============================================================================
print("\n" + "=" * 65)
print("  STEP 3 — CORRELATION AND REGRESSION")
print("=" * 65)

# ── 3.1  Pearson Correlation Coefficient --------------------------------------
#
#  Formula:
#       r = Σ (xᵢ - x̄)(yᵢ - ȳ)
#           ──────────────────────────────────────────
#           √[ Σ(xᵢ - x̄)² ] · √[ Σ(yᵢ - ȳ)² ]
#
#  r close to +1 → strong positive linear relationship
#  r close to  0 → weak / no linear relationship

numerator_r   = sum((X[i] - mean_x) * (Y[i] - mean_y) for i in range(n))
denominator_r = math.sqrt(
    sum((X[i] - mean_x)**2 for i in range(n)) *
    sum((Y[i] - mean_y)**2 for i in range(n))
)
r = numerator_r / denominator_r

print(f"\n[3.1] Pearson Correlation Coefficient")
print(f"      r = {r:.6f}  →  {'Strong' if abs(r) > 0.6 else 'Moderate'} positive correlation")

# ── 3.2  Slope (m) and Intercept (c) ------------------------------------------
#
#  Least-squares regression line:   Y = mX + c
#
#  Slope formula:
#       m = Σ (xᵢ - x̄)(yᵢ - ȳ)
#           ─────────────────────
#              Σ (xᵢ - x̄)²
#
#  Intercept formula:
#       c = ȳ - m · x̄

sum_xy = sum((X[i] - mean_x) * (Y[i] - mean_y) for i in range(n))
sum_x2 = sum((X[i] - mean_x)**2                 for i in range(n))

slope     = sum_xy / sum_x2          # m
intercept = mean_y - slope * mean_x  # c

print(f"\n[3.2] Regression Coefficients  (Y = mX + c)")
print(f"      Slope (m)     = {slope:.6f}")
print(f"      Intercept (c) = {intercept:,.4f}")
print(f"\n      Equation: Price = {slope:.4f} × Income + ({intercept:,.2f})")

# ── 3.3  Predicted Values -------------------------------------------------------
#  Apply: Ŷ = mX + c  for every sample
Y_pred = np.array([slope * xi + intercept for xi in X])

print(f"\n[3.3] Predicted Values (first 5)")
for i in range(5):
    print(f"      X={X[i]:,.0f}  →  Ŷ={Y_pred[i]:,.0f}  (actual Y={Y[i]:,.0f})")


# =============================================================================
# STEP 4 — ERROR AND ACCURACY METRICS
# =============================================================================
print("\n" + "=" * 65)
print("  STEP 4 — ERROR AND ACCURACY METRICS")
print("=" * 65)

# ── 4.1  Residuals (errors) ---------------------------------------------------
#  eᵢ = Yᵢ - Ŷᵢ
residuals = Y - Y_pred

# ── 4.2  Absolute Error -------------------------------------------------------
#  |eᵢ| = |Yᵢ - Ŷᵢ|
abs_errors = np.abs(residuals)

# ── 4.3  Squared Error --------------------------------------------------------
#  eᵢ² = (Yᵢ - Ŷᵢ)²
sq_errors = residuals ** 2

# ── 4.4  MAE — Mean Absolute Error -------------------------------------------
#  MAE = (1/n) Σ |eᵢ|
MAE = sum(abs_errors) / n

# ── 4.5  MSE — Mean Squared Error --------------------------------------------
#  MSE = (1/n) Σ eᵢ²
MSE = sum(sq_errors) / n

# ── 4.6  RMSE — Root Mean Squared Error --------------------------------------
#  RMSE = √MSE
RMSE = math.sqrt(MSE)

# ── 4.7  R² — Coefficient of Determination -----------------------------------
#
#  SS_res = Σ (Yᵢ - Ŷᵢ)²   ← sum of squared residuals
#  SS_tot = Σ (Yᵢ - Ȳ)²    ← total sum of squares
#
#  R² = 1 - SS_res / SS_tot
#
#  R² = 1  → perfect fit
#  R² = 0  → model predicts no better than using the mean

SS_res = sum((Y[i] - Y_pred[i])**2 for i in range(n))
SS_tot = sum((Y[i] - mean_y)   **2 for i in range(n))
R2     = 1 - (SS_res / SS_tot)

print(f"\n[4.1] Residuals     — first 5 samples:")
print(f"      {residuals[:5].round(2)}")
print(f"\n[4.2] Absolute Errors (first 5): {abs_errors[:5].round(2)}")
print(f"\n[4.3] Squared Errors  (first 5): {sq_errors[:5].round(2)}")
print(f"\n[4.4] MAE  (Mean Absolute Error)          = ${MAE:>15,.2f}")
print(f"[4.5] MSE  (Mean Squared Error)           = {MSE:>20,.2f}")
print(f"[4.6] RMSE (Root Mean Squared Error)      = ${RMSE:>15,.2f}")
print(f"[4.7] R²   (Coefficient of Determination) = {R2:.6f}  ({R2*100:.2f}% variance explained)")


# =============================================================================
# STEP 5 — PLOTS
# All 4 required plots arranged in a single figure for submission
# =============================================================================
print("\n" + "=" * 65)
print("  STEP 5 — GENERATING PLOTS")
print("=" * 65)

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0F1117')

# Master title
fig.suptitle(
    "Numerical Methods — Linear Regression  |  Housing Price Dataset",
    fontsize=17, color=C_GOLD, fontweight='bold', y=0.98
)

# 2×2 grid
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

ax1 = fig.add_subplot(gs[0, 0])   # Scatter + regression line
ax2 = fig.add_subplot(gs[0, 1])   # Saturation curve
ax3 = fig.add_subplot(gs[1, 0])   # Residual plot
ax4 = fig.add_subplot(gs[1, 1])   # Histogram of errors

# ── Helper: axis formatting ────────────────────────────────────────────────
def fmt_millions(ax, axis='y'):
    fmt = plt.FuncFormatter(lambda v, _: f'${v/1e6:.1f}M')
    if axis == 'y': ax.yaxis.set_major_formatter(fmt)
    else:           ax.xaxis.set_major_formatter(fmt)

def fmt_thousands(ax, axis='x'):
    fmt = plt.FuncFormatter(lambda v, _: f'${v/1e3:.0f}K')
    if axis == 'x': ax.xaxis.set_major_formatter(fmt)
    else:           ax.yaxis.set_major_formatter(fmt)

def annotate_box(ax, text):
    """Small stats box in top-left corner of axes."""
    ax.text(0.03, 0.97, text, transform=ax.transAxes,
            va='top', ha='left', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#0F1117',
                      edgecolor=C_GOLD, alpha=0.85),
            color='#C8D0E7', fontfamily='monospace')


# ── PLOT 1 — Scatter Plot + Regression Line ────────────────────────────────
#
#  Shows: raw (X, Y) data points and the fitted line  Ŷ = mX + c
#  The regression line passes through (x̄, ȳ) by construction.
# ──────────────────────────────────────────────────────────────────────────
ax1.scatter(X, Y, alpha=0.18, s=9, color=C_BLUE, label='Observed data')

x_line = np.linspace(X.min(), X.max(), 300)
y_line = slope * x_line + intercept
ax1.plot(x_line, y_line, color=C_RED, lw=2.5,
         label=f'Ŷ = {slope:.2f}·X + ({intercept:,.0f})')

# Mark mean point (x̄, ȳ) — the regression line always passes through it
ax1.scatter([mean_x], [mean_y], color=C_GOLD, s=80, zorder=5,
            marker='D', label=f'(x̄, ȳ) = ({mean_x:,.0f}, {mean_y:,.0f})')

ax1.set_title("① Scatter Plot + Regression Line", color=C_GOLD, pad=10)
ax1.set_xlabel("Avg. Area Income ($)")
ax1.set_ylabel("House Price ($)")
fmt_thousands(ax1, axis='x')
fmt_millions(ax1,  axis='y')
ax1.legend()
ax1.grid(True)

annotate_box(ax1,
    f"m (slope)     = {slope:.4f}\n"
    f"c (intercept) = {intercept:,.0f}\n"
    f"r             = {r:.4f}\n"
    f"R²            = {R2:.4f}"
)


# ── PLOT 2 — Saturation Curve ─────────────────────────────────────────────
#
#  Definition: X is sorted ascending; predicted Ŷ is plotted against sorted X.
#  Shows how the model's output grows (saturates) across the income range.
#  A linear model gives a straight saturation curve; non-linear models curve.
# ──────────────────────────────────────────────────────────────────────────
sort_idx    = np.argsort(X)
X_sorted    = X[sort_idx]
Y_sorted    = Y[sort_idx]
Y_pred_sort = Y_pred[sort_idx]

ax2.plot(X_sorted, Y_sorted,    color=C_BLUE,  lw=0.8, alpha=0.5, label='Actual (sorted)')
ax2.plot(X_sorted, Y_pred_sort, color=C_GREEN, lw=2.5,            label='Predicted (saturation)')

ax2.set_title("② Saturation Curve  (sorted X vs Predicted Y)", color=C_GOLD, pad=10)
ax2.set_xlabel("Avg. Area Income — sorted ($)")
ax2.set_ylabel("House Price ($)")
fmt_thousands(ax2, axis='x')
fmt_millions(ax2,  axis='y')
ax2.legend()
ax2.grid(True)

annotate_box(ax2,
    f"Min  Ŷ = ${Y_pred_sort.min():,.0f}\n"
    f"Max  Ŷ = ${Y_pred_sort.max():,.0f}\n"
    f"Range  = ${(Y_pred_sort.max()-Y_pred_sort.min()):,.0f}"
)


# ── PLOT 3 — Residual Plot ─────────────────────────────────────────────────
#
#  Residual eᵢ = Yᵢ − Ŷᵢ plotted against fitted values Ŷᵢ.
#  A good model has residuals randomly scattered around the zero line.
#  Patterns (funnel, curve) indicate model mis-specification.
# ──────────────────────────────────────────────────────────────────────────
ax3.scatter(Y_pred, residuals, alpha=0.20, s=8, color=C_PURPLE)
ax3.axhline(0, color=C_RED, lw=2, linestyle='--', label='Zero error line')

# Mark ±1 RMSE bands
ax3.axhline( RMSE, color=C_ORANGE, lw=1.2, linestyle=':', alpha=0.8, label=f'+RMSE = ${RMSE/1e3:.1f}K')
ax3.axhline(-RMSE, color=C_ORANGE, lw=1.2, linestyle=':', alpha=0.8, label=f'−RMSE = ${RMSE/1e3:.1f}K')

ax3.set_title("③ Residual Plot  (Errors vs Fitted Values)", color=C_GOLD, pad=10)
ax3.set_xlabel("Fitted (Predicted) Values Ŷ ($)")
ax3.set_ylabel("Residual  eᵢ = Y − Ŷ  ($)")
fmt_millions(ax3, axis='x')
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v/1e3:.0f}K'))
ax3.legend()
ax3.grid(True)

annotate_box(ax3,
    f"MAE  = ${MAE/1e3:,.1f}K\n"
    f"MSE  = {MSE:.2e}\n"
    f"RMSE = ${RMSE/1e3:,.1f}K"
)


# ── PLOT 4 — Histogram of Residuals / Errors ──────────────────────────────
#
#  Shows the distribution of eᵢ = Yᵢ − Ŷᵢ.
#  A well-behaved model has residuals normally distributed around 0 (zero mean).
#  Skew or heavy tails indicate systematic bias or non-linearity.
# ──────────────────────────────────────────────────────────────────────────
counts, bin_edges = np.histogram(residuals, bins=50)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

ax4.bar(bin_centers, counts, width=np.diff(bin_edges),
        color=C_ORANGE, alpha=0.80, edgecolor='none', label='Residual frequency')

ax4.axvline(0,              color=C_RED,    lw=2,   linestyle='--', label='Zero (perfect fit)')
ax4.axvline( RMSE,          color=C_GREEN,  lw=1.5, linestyle=':',  label=f'+RMSE=${RMSE/1e3:.1f}K')
ax4.axvline(-RMSE,          color=C_GREEN,  lw=1.5, linestyle=':',  label=f'−RMSE=${RMSE/1e3:.1f}K')
ax4.axvline(residuals.mean(), color=C_GOLD, lw=1.5, linestyle='-.',  label=f'Mean err≈{residuals.mean():.0f}')

ax4.set_title("④ Histogram of Residuals  (Error Distribution)", color=C_GOLD, pad=10)
ax4.set_xlabel("Residual  eᵢ  ($)")
ax4.set_ylabel("Frequency")
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v/1e3:.0f}K'))
ax4.legend()
ax4.grid(True, axis='y')

residual_std = math.sqrt(sum(e**2 for e in residuals) / n)
annotate_box(ax4,
    f"Mean error ≈ {residuals.mean():,.0f}\n"
    f"Std(errors) = ${residual_std/1e3:,.1f}K\n"
    f"R²          = {R2:.4f}"
)

# ── Save figure ───────────────────────────────────────────────────────────
plt.savefig('regression_output.png', dpi=150, bbox_inches='tight',
            facecolor='#0F1117')
plt.close()

print("\n  → All 4 plots saved to: regression_output.png")


# =============================================================================
# FINAL SUMMARY PRINTOUT
# =============================================================================
print("\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)
print(f"\n  Dataset    : Housing.csv")
print(f"  X (input)  : Avg. Area Income")
print(f"  Y (output) : Price")
print(f"  Samples    : {n}  (after Z-score outlier removal)")
print(f"\n  ── Statistical Measures ──────────────────────────────")
print(f"  Mean X          : {mean_x:>15,.4f}")
print(f"  Median X        : {median_x:>15,.4f}")
print(f"  Mode X          : {mode_x:>15,.2f}")
print(f"  Std Dev X (σ)   : {std_x:>15,.4f}")
print(f"  Variance X (σ²) : {var_x:>15,.4f}")
print(f"")
print(f"  Mean Y          : {mean_y:>15,.4f}")
print(f"  Median Y        : {median_y:>15,.4f}")
print(f"  Mode Y          : {mode_y:>15,.2f}")
print(f"  Std Dev Y (σ)   : {std_y:>15,.4f}")
print(f"  Variance Y (σ²) : {var_y:>15,.4f}")
print(f"\n  ── Regression ────────────────────────────────────────")
print(f"  Correlation r   : {r:>15.6f}")
print(f"  Slope  (m)      : {slope:>15.6f}")
print(f"  Intercept (c)   : {intercept:>15.4f}")
print(f"  Equation        : Ŷ = {slope:.4f} × X + ({intercept:,.2f})")
print(f"\n  ── Error Metrics ─────────────────────────────────────")
print(f"  MAE             : ${MAE:>15,.2f}")
print(f"  MSE             : {MSE:>20,.2f}")
print(f"  RMSE            : ${RMSE:>15,.2f}")
print(f"  R²              : {R2:>15.6f}  ({R2*100:.2f}% variance explained)")
print(f"\n  Output file     : regression_output.png")
print("=" * 65)