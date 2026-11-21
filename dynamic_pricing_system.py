"""
Dynamic Pricing & Demand Intelligence System for SMEs
=====================================================
Author: Siratal Mustakim Ador | BYSDO Internship Project
Purpose: Translate pricing data into profit-driven decisions for SMEs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────
# STYLE CONFIG
# ─────────────────────────────────────────
COLORS = {
    "primary":   "#0A2342",
    "accent":    "#E63946",
    "gold":      "#F4A261",
    "teal":      "#2A9D8F",
    "light":     "#F1FAEE",
    "mid":       "#457B9D",
    "bg":        "#0D1B2A",
}

plt.rcParams.update({
    "figure.facecolor":  COLORS["bg"],
    "axes.facecolor":    "#0F2035",
    "axes.edgecolor":    "#1E3A5F",
    "axes.labelcolor":   COLORS["light"],
    "xtick.color":       COLORS["light"],
    "ytick.color":       COLORS["light"],
    "text.color":        COLORS["light"],
    "grid.color":        "#1E3A5F",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})

OUTPUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# 1. DATASET GENERATION
# ─────────────────────────────────────────
print("\n" + "="*60)
print("  DYNAMIC PRICING & DEMAND INTELLIGENCE SYSTEM")
print("="*60)
print("\n[1/5] Generating realistic dataset...")

products = {
    "P001": {"name": "Wireless Earbuds",  "base_price": 45.0, "cost": 18.0, "base_demand": 80},
    "P002": {"name": "Leather Wallet",    "base_price": 30.0, "cost": 10.0, "base_demand": 110},
    "P003": {"name": "Desk Organizer",    "base_price": 22.0, "cost":  7.5, "base_demand": 140},
}

date_range = pd.date_range(start="2024-01-01", periods=85, freq="D")
records = []

for date in date_range:
    dow   = date.dayofweek         # 0=Mon…6=Sun
    month = date.month

    # Season mapping
    if month in [12, 1, 2]:   season = "Winter"
    elif month in [3, 4, 5]:  season = "Spring"
    elif month in [6, 7, 8]:  season = "Summer"
    else:                      season = "Autumn"

    for pid, meta in products.items():
        # Price variation: ±20% around base
        price_variation = np.random.uniform(-0.20, 0.20)
        price = round(meta["base_price"] * (1 + price_variation), 2)

        # Promotion: 25% chance; more likely on weekends
        promo_prob = 0.35 if dow >= 5 else 0.20
        promotion  = int(np.random.random() < promo_prob)

        # Demand model: price elasticity + promo boost + weekday pattern + noise
        elasticity     = -1.8   # 1% price rise → 1.8% demand drop
        price_effect   = meta["base_demand"] * (1 + elasticity * price_variation)
        promo_boost    = price_effect * 0.25 * promotion
        weekend_boost  = price_effect * 0.10 if dow >= 5 else 0
        season_factor  = {"Winter": 1.12, "Spring": 1.00, "Summer": 0.90, "Autumn": 1.05}[season]
        noise          = np.random.normal(0, meta["base_demand"] * 0.08)

        units_sold = max(1, int((price_effect + promo_boost + weekend_boost) * season_factor + noise))

        revenue = round(price * units_sold, 2)
        profit  = round((price - meta["cost"]) * units_sold, 2)

        records.append({
            "date":        date.strftime("%Y-%m-%d"),
            "product_id":  pid,
            "product_name":meta["name"],
            "price":       price,
            "cost":        meta["cost"],
            "units_sold":  units_sold,
            "revenue":     revenue,
            "profit":      profit,
            "day_of_week": dow,
            "day_name":    date.strftime("%A"),
            "promotion":   promotion,
            "season":      season,
            "month":       month,
        })

df = pd.DataFrame(records)
df.to_csv(f"{OUTPUT_DIR}/pricing_dataset.csv", index=False)
print(f"  ✓ Dataset created: {len(df)} rows × {len(df.columns)} columns")
print(f"  ✓ Products: {', '.join([m['name'] for m in products.values()])}")
print(f"  ✓ Date range: {df['date'].min()} → {df['date'].max()}")


# ─────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────
print("\n[2/5] Running Exploratory Data Analysis...")

# Summary stats
print("\n  ── Summary Statistics ──")
summary = df.groupby("product_name").agg(
    Avg_Price=("price","mean"),
    Avg_Units=("units_sold","mean"),
    Total_Revenue=("revenue","sum"),
    Total_Profit=("profit","sum"),
    Promo_Rate=("promotion","mean"),
).round(2)
print(summary.to_string())

# Correlation
corr_cols = ["price","units_sold","revenue","profit","promotion","day_of_week"]
corr_matrix = df[corr_cols].corr()

# Price elasticity per product
print("\n  ── Price Elasticity ──")
for pid, meta in products.items():
    sub = df[df["product_id"]==pid]
    X = sub[["price"]]; y = sub["units_sold"]
    lr = LinearRegression().fit(X, y)
    pct_demand_change = (lr.coef_[0] * 1) / sub["units_sold"].mean() * 100
    print(f"  {meta['name']}: +$1 price → {pct_demand_change:.1f}% demand change")


# ── EDA PLOTS ──
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Dynamic Pricing — Exploratory Data Analysis", fontsize=16,
             fontweight="bold", color=COLORS["gold"], y=0.98)

palette = [COLORS["accent"], COLORS["teal"], COLORS["gold"]]
prod_names = [m["name"] for m in products.values()]

# Plot 1: Price vs Units Sold
ax = axes[0, 0]
for i, (pid, meta) in enumerate(products.items()):
    sub = df[df["product_id"]==pid]
    ax.scatter(sub["price"], sub["units_sold"], alpha=0.55, s=40,
               color=palette[i], label=meta["name"])
    # Trend line
    m, b = np.polyfit(sub["price"], sub["units_sold"], 1)
    xline = np.linspace(sub["price"].min(), sub["price"].max(), 100)
    ax.plot(xline, m*xline+b, color=palette[i], lw=1.5, linestyle="--", alpha=0.8)
ax.set_title("Price vs Units Sold"); ax.set_xlabel("Price ($)"); ax.set_ylabel("Units Sold")
ax.legend(fontsize=8); ax.grid(True)

# Plot 2: Price vs Revenue
ax = axes[0, 1]
for i, (pid, meta) in enumerate(products.items()):
    sub = df[df["product_id"]==pid]
    ax.scatter(sub["price"], sub["revenue"], alpha=0.55, s=40,
               color=palette[i], label=meta["name"])
ax.set_title("Price vs Revenue"); ax.set_xlabel("Price ($)"); ax.set_ylabel("Revenue ($)")
ax.legend(fontsize=8); ax.grid(True)

# Plot 3: Price vs Profit
ax = axes[0, 2]
for i, (pid, meta) in enumerate(products.items()):
    sub = df[df["product_id"]==pid]
    ax.scatter(sub["price"], sub["profit"], alpha=0.55, s=40,
               color=palette[i], label=meta["name"])
ax.set_title("Price vs Profit"); ax.set_xlabel("Price ($)"); ax.set_ylabel("Profit ($)")
ax.legend(fontsize=8); ax.grid(True)

# Plot 4: Correlation Heatmap
ax = axes[1, 0]
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_matrix, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.5, mask=mask,
            annot_kws={"size":9}, cbar_kws={"shrink":0.7})
ax.set_title("Correlation Matrix"); ax.tick_params(labelsize=8)

# Plot 5: Promo vs Non-Promo units
ax = axes[1, 1]
promo_data = df.groupby(["product_name","promotion"])["units_sold"].mean().unstack()
promo_data.plot(kind="bar", ax=ax, color=[COLORS["mid"], COLORS["accent"]],
                edgecolor="none", width=0.6)
ax.set_title("Avg Units: Promo vs No Promo"); ax.set_xlabel("")
ax.set_ylabel("Avg Units Sold"); ax.legend(["No Promo","Promo"], fontsize=9)
ax.tick_params(axis="x", rotation=20, labelsize=8); ax.grid(axis="y")

# Plot 6: Day-of-week demand pattern
ax = axes[1, 2]
dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
dow_avg = df.groupby("day_of_week")["units_sold"].mean()
ax.bar(range(7), dow_avg.values, color=COLORS["teal"], edgecolor="none", alpha=0.85)
ax.set_xticks(range(7)); ax.set_xticklabels(dow_labels)
ax.set_title("Avg Units Sold by Day of Week")
ax.set_ylabel("Avg Units Sold"); ax.grid(axis="y")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_eda_analysis.png", dpi=150, bbox_inches="tight",
            facecolor=COLORS["bg"])
plt.close()
print("  ✓ EDA chart saved")


# ─────────────────────────────────────────
# 3. DEMAND PREDICTION MODEL
# ─────────────────────────────────────────
print("\n[3/5] Building Demand Prediction Models...")

model_results = {}
product_models = {}

for pid, meta in products.items():
    sub = df[df["product_id"]==pid].copy()
    features = ["price","promotion","day_of_week","month"]
    X = sub[features]
    y = sub["units_sold"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Gradient Boosting (better for nonlinear demand)
    model = GradientBoostingRegressor(n_estimators=120, learning_rate=0.08,
                                       max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    model_results[pid] = {"name": meta["name"], "R2": r2, "MAE": mae}
    product_models[pid] = {"model": model, "features": features, "cost": meta["cost"]}

    print(f"  {meta['name']}: R² = {r2:.3f}  |  MAE = {mae:.1f} units")

print("  ✓ Models trained (Gradient Boosting Regressor)")


# ─────────────────────────────────────────
# 4. PROFIT OPTIMIZATION ENGINE
# ─────────────────────────────────────────
print("\n[4/5] Running Profit Optimization Engine...")

optimization_results = {}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Profit Optimization Engine — Price Simulation", fontsize=15,
             fontweight="bold", color=COLORS["gold"], y=1.02)

for idx, (pid, meta) in enumerate(products.items()):
    cost = meta["cost"]
    base_price = meta["base_price"]

    # Simulate price range: cost+1 to base_price×2
    price_range = np.linspace(cost + 1, base_price * 2, 300)
    m = product_models[pid]

    revenues = []
    profits  = []

    for p in price_range:
        # Simulate both promo and no-promo, weighted average
        sim_df = pd.DataFrame({
            "price":       [p, p],
            "promotion":   [0, 1],
            "day_of_week": [2, 2],  # mid-week baseline
            "month":       [4, 4],
        })
        preds = m["model"].predict(sim_df[m["features"]])
        # 75% non-promo / 25% promo (realistic mix)
        units = 0.75 * preds[0] + 0.25 * preds[1]
        units = max(0, units)
        revenues.append(p * units)
        profits.append((p - cost) * units)

    revenues = np.array(revenues)
    profits  = np.array(profits)

    opt_profit_idx  = np.argmax(profits)
    opt_revenue_idx = np.argmax(revenues)

    opt_price_profit  = price_range[opt_profit_idx]
    opt_price_revenue = price_range[opt_revenue_idx]
    max_profit        = profits[opt_profit_idx]
    max_revenue       = revenues[opt_revenue_idx]

    optimization_results[pid] = {
        "name":               meta["name"],
        "cost":               cost,
        "base_price":         base_price,
        "opt_price_profit":   round(opt_price_profit, 2),
        "max_profit":         round(max_profit, 2),
        "opt_price_revenue":  round(opt_price_revenue, 2),
        "max_revenue":        round(max_revenue, 2),
        "profit_margin_pct":  round((opt_price_profit - cost) / opt_price_profit * 100, 1),
    }

    ax = axes[idx]
    ax.plot(price_range, profits,  color=COLORS["teal"],  lw=2,   label="Profit/Day")
    ax.plot(price_range, revenues, color=COLORS["gold"],  lw=1.5, label="Revenue/Day", alpha=0.7)
    ax.axvline(opt_price_profit,  color=COLORS["accent"], lw=2, linestyle="--",
               label=f"Profit Max ${opt_price_profit:.1f}")
    ax.axvline(opt_price_revenue, color=COLORS["mid"],    lw=1.5, linestyle=":",
               label=f"Revenue Max ${opt_price_revenue:.1f}")

    ax.scatter([opt_price_profit],  [max_profit],  color=COLORS["accent"], s=80, zorder=5)
    ax.set_title(meta["name"], fontweight="bold")
    ax.set_xlabel("Price ($)"); ax.set_ylabel("$"); ax.legend(fontsize=8); ax.grid(True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_profit_optimization.png", dpi=150, bbox_inches="tight",
            facecolor=COLORS["bg"])
plt.close()

print("  ✓ Optimization complete")
for pid, res in optimization_results.items():
    print(f"\n  ── {res['name']} ──")
    print(f"     Current base price   : ${res['base_price']:.2f}")
    print(f"     Profit-optimal price : ${res['opt_price_profit']:.2f}  → Max profit/day: ${res['max_profit']:.0f}")
    print(f"     Revenue-optimal price: ${res['opt_price_revenue']:.2f}  → Max revenue/day: ${res['max_revenue']:.0f}")
    print(f"     Optimal profit margin: {res['profit_margin_pct']}%")


# ─────────────────────────────────────────
# 5. BUSINESS INTELLIGENCE INSIGHTS
# ─────────────────────────────────────────
print("\n[5/5] Generating Business Intelligence Insights...")

insights = []
for pid, res in optimization_results.items():
    base   = res["base_price"]
    opt    = res["opt_price_profit"]
    change = (opt - base) / base * 100
    direction = "Increasing" if change > 0 else "Reducing"
    abs_change = abs(change)

    # Estimate profit lift vs current base
    sub = df[df["product_id"]==pid]
    current_daily_profit = sub["profit"].sum() / sub["date"].nunique()
    lift_pct = (res["max_profit"] - current_daily_profit) / current_daily_profit * 100

    insight = {
        "product":        res["name"],
        "base_price":     base,
        "opt_price":      opt,
        "price_change":   round(change, 1),
        "profit_lift_pct":round(lift_pct, 1),
        "max_profit":     res["max_profit"],
        "recommendation": (
            f"{direction} price by {abs_change:.1f}% (from ${base} to ${opt:.2f}) "
            f"is projected to improve daily profit by {lift_pct:.1f}%."
        ),
    }
    insights.append(insight)
    print(f"\n  [{res['name']}]")
    print(f"  → {insight['recommendation']}")

# Promotion timing analysis
promo_lift = df.groupby(["product_name","promotion"])["profit"].mean().unstack()
print("\n  ── Promotion Impact on Avg Daily Profit ──")
for pname in promo_lift.index:
    if 0 in promo_lift.columns and 1 in promo_lift.columns:
        lift = (promo_lift.loc[pname, 1] - promo_lift.loc[pname, 0]) / promo_lift.loc[pname, 0] * 100
        print(f"  {pname}: Promotions improve daily profit by {lift:.1f}%")

# Weekend vs weekday
weekend_avg = df[df["day_of_week"]>=5]["units_sold"].mean()
weekday_avg = df[df["day_of_week"]<5]["units_sold"].mean()
weekend_lift = (weekend_avg - weekday_avg) / weekday_avg * 100
print(f"\n  ── Weekend Effect ──")
print(f"  Weekend demand is {weekend_lift:.1f}% higher than weekday baseline")
print(f"  → Best promo timing: Friday–Sunday price promotions")


# ─────────────────────────────────────────
# BI SUMMARY DASHBOARD (Static PNG)
# ─────────────────────────────────────────
fig = plt.figure(figsize=(20, 13), facecolor=COLORS["bg"])
fig.suptitle("DYNAMIC PRICING INTELLIGENCE — EXECUTIVE DASHBOARD",
             fontsize=18, fontweight="bold", color=COLORS["gold"], y=0.98)

gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.4)

# KPI Cards
kpis = [
    ("Total Revenue",  f"${df['revenue'].sum():,.0f}",  COLORS["teal"]),
    ("Total Profit",   f"${df['profit'].sum():,.0f}",   COLORS["accent"]),
    ("Avg Profit Margin", f"{((df['profit']/df['revenue']).mean()*100):.1f}%", COLORS["gold"]),
    ("Days Analyzed",  str(df['date'].nunique()),        COLORS["mid"]),
]
for i, (label, value, color) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(color + "22")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.text(0.5, 0.65, value, ha="center", va="center", fontsize=22,
            fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha="center", va="center", fontsize=10,
            color=COLORS["light"], transform=ax.transAxes)
    for spine in ax.spines.values(): spine.set_edgecolor(color); spine.set_linewidth(2)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# Revenue trend
ax2 = fig.add_subplot(gs[1, :2])
daily_rev = df.groupby("date")["revenue"].sum().reset_index()
daily_rev["date"] = pd.to_datetime(daily_rev["date"])
ax2.fill_between(daily_rev["date"], daily_rev["revenue"], alpha=0.3, color=COLORS["teal"])
ax2.plot(daily_rev["date"], daily_rev["revenue"], color=COLORS["teal"], lw=2)
ax2.set_title("Daily Revenue Trend (All Products)"); ax2.set_ylabel("Revenue ($)")
ax2.tick_params(axis="x", rotation=30, labelsize=7); ax2.grid(True)

# Profit trend
ax3 = fig.add_subplot(gs[1, 2:])
daily_pft = df.groupby("date")["profit"].sum().reset_index()
daily_pft["date"] = pd.to_datetime(daily_pft["date"])
ax3.fill_between(daily_pft["date"], daily_pft["profit"], alpha=0.3, color=COLORS["accent"])
ax3.plot(daily_pft["date"], daily_pft["profit"], color=COLORS["accent"], lw=2)
ax3.set_title("Daily Profit Trend (All Products)"); ax3.set_ylabel("Profit ($)")
ax3.tick_params(axis="x", rotation=30, labelsize=7); ax3.grid(True)

# Optimal price comparison
ax4 = fig.add_subplot(gs[2, :2])
prod_labels = [r["name"] for r in optimization_results.values()]
base_prices  = [r["base_price"]       for r in optimization_results.values()]
opt_prices   = [r["opt_price_profit"] for r in optimization_results.values()]
x = np.arange(len(prod_labels))
bars1 = ax4.bar(x - 0.2, base_prices, 0.35, label="Current Price", color=COLORS["mid"],    alpha=0.85)
bars2 = ax4.bar(x + 0.2, opt_prices,  0.35, label="Optimal Price", color=COLORS["accent"], alpha=0.85)
ax4.set_xticks(x); ax4.set_xticklabels(prod_labels, fontsize=8)
ax4.set_title("Current vs Profit-Optimal Price"); ax4.set_ylabel("Price ($)")
ax4.legend(fontsize=9); ax4.grid(axis="y")

# Recommendations panel
ax5 = fig.add_subplot(gs[2, 2:])
ax5.set_facecolor("#0A1628"); ax5.set_xlim(0,1); ax5.set_ylim(0,1)
for spine in ax5.spines.values(): spine.set_edgecolor(COLORS["gold"]); spine.set_linewidth(1.5)
ax5.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax5.text(0.5, 0.95, "PRICING RECOMMENDATIONS", ha="center", va="top",
         fontsize=11, fontweight="bold", color=COLORS["gold"], transform=ax5.transAxes)

rec_lines = []
for ins in insights:
    arrow = "▲" if ins["price_change"] > 0 else "▼"
    rec_lines.append(
        f"{arrow} {ins['product']}: ${ins['base_price']} → ${ins['opt_price']:.2f}  "
        f"(+{ins['profit_lift_pct']}% profit)"
    )
rec_lines.append("")
rec_lines.append(f"● Weekend promos lift demand +{weekend_lift:.1f}%")
rec_lines.append("● Run Friday–Sunday discount events for max volume")
rec_lines.append("● Maintain optimal margins Mon–Thu at full price")

y_pos = 0.82
for line in rec_lines:
    color = COLORS["accent"] if line.startswith("▲") else (
            COLORS["teal"]  if line.startswith("▼") else
            COLORS["gold"]  if line.startswith("●") else COLORS["light"])
    ax5.text(0.04, y_pos, line, ha="left", va="top", fontsize=8.5,
             color=color, transform=ax5.transAxes, fontfamily="monospace")
    y_pos -= 0.10

plt.savefig(f"{OUTPUT_DIR}/03_executive_dashboard.png", dpi=150, bbox_inches="tight",
            facecolor=COLORS["bg"])
plt.close()

# ─────────────────────────────────────────
# POWER BI EXPORT DATA
# ─────────────────────────────────────────
# Main dataset
df.to_csv(f"{OUTPUT_DIR}/pricing_dataset.csv", index=False)

# Optimization results table
opt_df = pd.DataFrame(optimization_results).T.reset_index(drop=True)
opt_df.to_csv(f"{OUTPUT_DIR}/optimization_results.csv", index=False)

# Daily aggregated (for Power BI KPI cards and time-series)
daily_agg = df.groupby(["date","product_name"]).agg(
    avg_price=("price","mean"),
    units_sold=("units_sold","sum"),
    revenue=("revenue","sum"),
    profit=("profit","sum"),
    promotion_days=("promotion","sum"),
).reset_index()
daily_agg.to_csv(f"{OUTPUT_DIR}/daily_aggregated.csv", index=False)

# Price-Demand simulation data (for Power BI curve charts)
sim_rows = []
for pid, meta in products.items():
    cost = meta["cost"]
    m = product_models[pid]
    for p in np.linspace(cost+1, meta["base_price"]*2, 80):
        sim_df_row = pd.DataFrame({
            "price": [p], "promotion": [0], "day_of_week": [2], "month": [4]
        })
        units = max(0, m["model"].predict(sim_df_row[m["features"]])[0])
        sim_rows.append({
            "product_id":   pid,
            "product_name": meta["name"],
            "price":        round(p, 2),
            "pred_units":   round(units, 1),
            "pred_revenue": round(p * units, 2),
            "pred_profit":  round((p - cost) * units, 2),
        })
sim_curve_df = pd.DataFrame(sim_rows)
sim_curve_df.to_csv(f"{OUTPUT_DIR}/simulation_curves.csv", index=False)

# Insights summary
insights_df = pd.DataFrame(insights)
insights_df.to_csv(f"{OUTPUT_DIR}/business_insights.csv", index=False)

print("\n  ✓ All CSVs exported for Power BI")
print(f"    - pricing_dataset.csv   ({len(df)} rows)")
print(f"    - optimization_results.csv")
print(f"    - daily_aggregated.csv")
print(f"    - simulation_curves.csv  ({len(sim_curve_df)} rows)")
print(f"    - business_insights.csv")

# ─────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────
print("\n" + "="*60)
print("  SYSTEM COMPLETE — EXECUTIVE SUMMARY")
print("="*60)
print(f"\n  Dataset Period : {df['date'].min()} to {df['date'].max()} (85 days)")
print(f"  Total Revenue  : ${df['revenue'].sum():,.0f}")
print(f"  Total Profit   : ${df['profit'].sum():,.0f}")
print(f"  Avg Margin     : {(df['profit'].sum()/df['revenue'].sum()*100):.1f}%")
print(f"\n  Model Performance:")
for pid, r in model_results.items():
    print(f"    {r['name']}: R²={r['R2']:.3f}, MAE={r['MAE']:.1f} units")
print(f"\n  Optimal Pricing:")
for pid, res in optimization_results.items():
    print(f"    {res['name']}: ${res['base_price']} → ${res['opt_price_profit']:.2f} (profit-optimal)")
print("\n  Deliverables saved to /outputs/")
print("="*60 + "\n")
