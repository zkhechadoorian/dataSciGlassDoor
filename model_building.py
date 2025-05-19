import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
import pickle
import pathlib

# ─── Data prep ────────────────────────────────────────────────────────────────
print("📥 Loading dataset…")
df = pd.read_csv("eda_data.csv")

cols = [
    "avg_salary", "Rating", "Size", "Type of ownership", "Industry", "Sector",
    "Revenue", "num_comp", "hourly", "employer_provided", "job_state",
    "same_state", "age", "python_yn", "spark", "aws", "excel", "job_simp",
    "seniority", "desc_len",
]
print("🔍 Selecting relevant features…")
df_model = df[cols]

print("🧠 One‑hot encoding categorical columns…")
df_dum = pd.get_dummies(df_model, drop_first=True)

X = df_dum.drop("avg_salary", axis=1)
y = df_dum["avg_salary"]

print("🧪 Train/test split …")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ─── Helper – positive MAE scorer ─────────────────────────────────────────────
pos_mae = make_scorer(mean_absolute_error, greater_is_better=False) 

def cv_mae(model):
    """Return +MAE (so ‘lower is better’ is obvious)."""
    scores = cross_val_score(model, X_train, y_train, scoring=pos_mae, cv=5)
    return -scores.mean()

# ─── (1) OLS for quick interpretability ──────────────────────────────────────
print("📊 statsmodels OLS …")
X_sm = sm.add_constant(X.select_dtypes(include=[np.number]))
ols_res = sm.OLS(y.astype(float), X_sm).fit()
print(ols_res.summary())

# ─── (2) Linear Regression ───────────────────────────────────────────────────
print("📈 Linear Regression (sklearn)…")
lm = LinearRegression()
lm_mae = cv_mae(lm)
print(f"✅ CV MAE: {lm_mae:0.2f}k USD")

# ─── (3) Lasso Regression + alpha scan ───────────────────────────────────────
print("📉 Lasso Regression …")
alphas, mae_vals = [], []

for a in np.linspace(0.01, 1.0, 100):
    lml = Lasso(alpha=a, max_iter=10_000)
    mae_vals.append(cv_mae(lml))
    alphas.append(a)

best_alpha = alphas[int(np.argmin(mae_vals))]
print(f"🏆 Best alpha = {best_alpha:.2f} (CV MAE {min(mae_vals):.2f}k)")

# Plot for reference
plt.figure()
plt.plot(alphas, mae_vals)
plt.xlabel("Alpha")
plt.ylabel("CV MAE (k USD, lower is better)")
plt.title("Lasso alpha tuning")
plt.show()

# Re‑fit best Lasso on full training data
lasso_best = Lasso(alpha=best_alpha, max_iter=10_000).fit(X_train, y_train)

# ─── (4) Random Forest + GridSearch ──────────────────────────────────────────
print("🌲 Random Forest …")
rf = RandomForestRegressor(random_state=42)

grid = {
    "n_estimators": range(40, 241, 40),
    "criterion": ["squared_error", "absolute_error"],
    "max_features": ["sqrt", "log2", None],        # ‘auto’ removed
}

print("🔍 GridSearchCV …")
gs = GridSearchCV(
    rf,
    grid,
    scoring=pos_mae,
    cv=5,
    n_jobs=-1,
    verbose=0,
)
gs.fit(X_train, y_train)
rf_best = gs.best_estimator_
print(f"🏆 Best RF → {rf_best}")
print(f"✅ Best CV MAE: {-gs.best_score_:0.2f}k USD")

# ─── (5) Test‑set evaluation ─────────────────────────────────────────────────
print("\n🔬 Test‑set results")
for name, model in [
    ("Linear Regression", lm.fit(X_train, y_train)),
    ("Lasso",            lasso_best),
    ("Random Forest",    rf_best),
]:
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    print(f"• {name}: MAE = {mae:0.2f}k USD")

# Simple ensemble (average of Linear & RF)
ensemble_pred = (lm.predict(X_test) + rf_best.predict(X_test)) / 2
ensemble_mae  = mean_absolute_error(y_test, ensemble_pred)
print(f"• Ensemble (Linear + RF): MAE = {ensemble_mae:0.2f}k USD")

# ─── (6) Persist best model ──────────────────────────────────────────────────
print("\n💾 Saving best model (Random Forest)…")
model_dir = pathlib.Path("FlaskAPI/models")
model_dir.mkdir(parents=True, exist_ok=True)

with open(model_dir / "model_file.p", "wb") as f:
    pickle.dump({"model": rf_best}, f)

# ─── (7) Quick sanity check ──────────────────────────────────────────────────
sample = X_test.iloc[[0]]          # keep header to avoid warning
print("\n♻️ Reloading & predicting one sample…")
with open(model_dir / "model_file.p", "rb") as f:
    model = pickle.load(f)["model"]

print("📌 Predicted salary:", model.predict(sample)[0])
print("🧾 Features used:", list(sample.iloc[0]))

print("\n✅ Pipeline finished successfully.")
