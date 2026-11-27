"""Customer segmentation: K-Means on behaviour + demographics."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
OUT, CH = ROOT / "outputs", ROOT / "outputs" / "charts"
CH.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

def save(name):
    plt.tight_layout()
    plt.savefig(CH / name, dpi=150)
    plt.close()

# 1. Load and check
cust = pd.read_csv(ROOT / "data" / "customers.csv")
tx = pd.read_csv(ROOT / "data" / "transactions.csv", parse_dates=["order_date"])
assert cust.customer_id.is_unique and not cust.isna().any().any() and not tx.isna().any().any()
assert (tx.amount > 0).all()

# 2. Features per customer (RFM + demographics)
snap = tx.order_date.max() + pd.Timedelta(days=1)
rfm = tx.groupby("customer_id").agg(
    recency_days=("order_date", lambda s: (snap - s.max()).days),
    frequency=("order_id", "count"),
    monetary=("amount", "sum"),
    avg_order_value=("amount", "mean")).reset_index()
df = cust.merge(rfm, on="customer_id", how="inner")

FEATURES = ["age", "annual_income_lakh", "recency_days", "frequency", "monetary", "avg_order_value"]
X = df[FEATURES].copy()
skewed = ["recency_days", "frequency", "monetary", "avg_order_value"]
X[skewed] = np.log1p(X[skewed])
X = StandardScaler().fit_transform(X)

# 3. Choose k
scores = []
for k in range(2, 9):
    m = KMeans(k, n_init=20, random_state=42).fit(X)
    scores.append((k, m.inertia_, silhouette_score(X, m.labels_)))
scores = pd.DataFrame(scores, columns=["k", "inertia", "silhouette"])
scores.to_csv(OUT / "k_selection.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(scores.k, scores.inertia, marker="o"); ax[0].set(title="Elbow method", xlabel="k", ylabel="Inertia")
ax[1].plot(scores.k, scores.silhouette, marker="o"); ax[1].set(title="Silhouette score", xlabel="k", ylabel="Score")
save("01_choose_k.png")

K = 4  # check k_selection.csv and 01_choose_k.png if you change the data
model = KMeans(K, n_init=20, random_state=42).fit(X)
df["cluster"] = model.labels_

# Stability across random seeds
stability = min(adjusted_rand_score(model.labels_, KMeans(K, n_init=20, random_state=s).fit_predict(X))
                for s in range(5))

# 4. Name segments from their own averages
means = df.groupby("cluster")[FEATURES].mean()
names, left = {}, list(means.index)
def take(label, column):
    c = means.loc[left, column].idxmax()
    names[c] = label
    left.remove(c)
take("Premium Loyalists", "monetary")
take("At-Risk / Inactive", "recency_days")
c = means.loc[left, "age"].idxmin(); names[c] = "Young Trend Shoppers"; left.remove(c)
names[left[0]] = "Family Essential Buyers"
df["segment"] = df.cluster.map(names)
tx = tx.merge(df[["customer_id", "segment"]], on="customer_id")
order = df.groupby("segment").monetary.mean().sort_values(ascending=False).index.tolist()

# 5. Profiles
profile = df.groupby("segment")[FEATURES].mean().loc[order]
profile.insert(0, "customers", df.segment.value_counts().loc[order])
profile["customer_share_%"] = 100 * profile.customers / len(df)
profile["revenue_share_%"] = 100 * df.groupby("segment").monetary.sum().loc[order] / df.monetary.sum()
cat = pd.crosstab(tx.segment, tx.category, values=tx.amount, aggfunc="sum", normalize="index").loc[order] * 100
pay = pd.crosstab(tx.segment, tx.payment_method, normalize="index").loc[order] * 100
profile.round(1).to_csv(OUT / "segment_profile.csv")
cat.round(1).to_csv(OUT / "category_preference_pct.csv")
pay.round(1).to_csv(OUT / "payment_preference_pct.csv")
df.to_csv(OUT / "customer_segments.csv", index=False)

# 6. Charts
pal = dict(zip(order, sns.color_palette("colorblind", K)))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, col, title in [(ax[0], "customer_share_%", "Share of customers (%)"),
                      (ax[1], "revenue_share_%", "Share of revenue (%)")]:
    a.barh(order, profile[col], color=[pal[s] for s in order]); a.invert_yaxis(); a.set_title(title)
    a.bar_label(a.containers[0], fmt="%.1f", padding=2); a.margins(x=.12)
ax[1].set_yticklabels([])
save("02_size_and_revenue.png")

pca = PCA(2, random_state=42)
xy = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=df.segment, hue_order=order, palette=pal, s=16)
plt.title(f"Segments in 2D (PCA, {pca.explained_variance_ratio_.sum():.0%} of variance)")
plt.xlabel("PC1"); plt.ylabel("PC2")
save("03_pca_clusters.png")

z = (df.groupby("segment")[FEATURES].mean().loc[order] - df[FEATURES].mean()) / df[FEATURES].std()
plt.figure(figsize=(10, 4.5))
sns.heatmap(z, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Key characteristics (standard deviations from average)"); plt.ylabel("")
save("04_characteristics_heatmap.png")

plt.figure(figsize=(10, 4.5))
sns.heatmap(cat, annot=True, fmt=".1f", cmap="Blues")
plt.title("Category preference (% of segment spend)"); plt.ylabel(""); plt.xlabel("")
save("05_category_preference.png")

monthly = tx.groupby([tx.order_date.dt.to_period("M"), "segment"]).amount.sum().unstack()[order] / 1e5
ax = monthly.plot(figsize=(11, 4.5), marker="o", color=[pal[s] for s in order])
ax.set(title="Monthly revenue by segment (lakh INR)", xlabel="", ylabel="Lakh INR")
save("06_monthly_revenue.png")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
sns.boxplot(df, y="segment", x="age", order=order, hue="segment", palette=pal, legend=False, ax=ax[0])
sns.boxplot(df, y="segment", x="annual_income_lakh", order=order, hue="segment", palette=pal, legend=False, ax=ax[1])
ax[0].set(title="Age", ylabel="", xlabel="Years"); ax[1].set(title="Annual income", ylabel="", xlabel="Lakh INR", yticklabels=[])
save("07_demographics.png")

print(scores.round(3).to_string(index=False))
print(f"\nk={K}, minimum seed-stability ARI={stability:.3f}")
print(profile.round(1).T.to_string())
print("\nCategory %\n", cat.round(1).T.to_string())
print("\nPayment %\n", pay.round(1).T.to_string())
print(f"\nCustomers={len(df)}, orders={len(tx)}, revenue={tx.amount.sum():,.0f}")
