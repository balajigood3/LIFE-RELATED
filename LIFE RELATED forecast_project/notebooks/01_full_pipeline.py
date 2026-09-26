"""
Complete end-to-end pipeline script
(Can be converted to Jupyter notebook by adding cells)
"""
# %% [markdown]
# # UAC Care Load Forecasting – Full Pipeline
# This script reproduces the entire analysis from data loading to evaluation.

# %%
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_prep import load_data, prepare_features, train_test_split_time
from models import run_all_models, evaluate

# %% [markdown]
# ## 1. Load & Inspect Data

# %%
df = load_data()
print(df.head())
print(df.describe())

# %% [markdown]
# ## 2. Feature Engineering

# %%
feat = prepare_features(df, target="Children_in_HHS_Care")
print("Feature columns:", feat.columns.tolist())
print("Shape after features:", feat.shape)

# %% [markdown]
# ## 3. Train / Test Split (chronological)

# %%
train, test = train_test_split_time(feat, test_days=60)
print(f"Train: {train['Date'].min().date()} → {train['Date'].max().date()}")
print(f"Test : {test['Date'].min().date()} → {test['Date'].max().date()}")

# %% [markdown]
# ## 4. Train & Evaluate All Models

# %%
results = run_all_models(train, test, target="Children_in_HHS_Care")

metrics = []
for name, res in results.items():
    m = res["metrics"]
    metrics.append({"Model": name, **{k: round(v, 2) for k, v in m.items()}})
metrics_df = pd.DataFrame(metrics).sort_values("MAE")
print(metrics_df)

# %% [markdown]
# ## 5. Visualize Forecast vs Actual

# %%
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(test["Date"], test["Children_in_HHS_Care"], label="Actual", color="black", lw=2)
for name, res in results.items():
    if name in ["RandomForest", "Naive", "MovingAvg_7"]:
        ax.plot(test["Date"], res["pred"], label=name, alpha=0.8)
ax.legend()
ax.set_title("Care Load – Forecast vs Actual (60-day hold-out)")
ax.set_ylabel("Children in HHS Care")
plt.tight_layout()
plt.savefig(ROOT / "outputs" / "notebook_forecast_comparison.png", dpi=120)
print("Saved notebook_forecast_comparison.png")

# %% [markdown]
# ## 6. Key Takeaway
# Random Forest delivers the lowest MAE on the Care Load series.
# Use the Streamlit app for interactive exploration and future forecasts.
