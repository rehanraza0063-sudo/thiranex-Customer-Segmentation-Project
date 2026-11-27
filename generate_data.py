"""Create a SYNTHETIC retail dataset (customers + transactions). Not real customer data."""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

N = 1500
END = pd.Timestamp("2026-10-31")
CATS = ["Electronics", "Fashion", "Grocery", "Home & Kitchen", "Beauty", "Sports"]
# share, mean age, income (lakh INR), orders/yr, basket (INR), days since last order, category mix
PROFILES = [
    (0.18, 42, 18, 22, 4200, 15, [.30, .18, .08, .26, .08, .10]),
    (0.30, 26, 6, 11, 1300, 35, [.12, .40, .08, .08, .22, .10]),
    (0.30, 37, 10, 8, 2100, 60, [.08, .10, .45, .24, .08, .05]),
    (0.22, 33, 9, 2, 1700, 190, [.28, .24, .10, .14, .10, .14]),
]
kind = rng.choice(len(PROFILES), N, p=[p[0] for p in PROFILES])

customers, orders = [], []
for i, k in enumerate(kind):
    _, age, inc, freq, basket, rec, mix = PROFILES[k]
    cid = f"C{i + 1:04d}"
    customers.append((cid, int(np.clip(rng.normal(age, 8), 18, 70)),
                      rng.choice(["Female", "Male"]),
                      rng.choice(["Mumbai", "Pune", "Delhi", "Bengaluru", "Nagpur"],
                                 p=[.3, .18, .2, .22, .1]),
                      round(float(np.clip(rng.normal(inc, inc * .25), 2, 40)), 1)))
    n = max(1, rng.poisson(freq * rng.lognormal(0, .3)))
    last = int(min(364, rng.exponential(rec)))
    days = np.concatenate([[last], rng.integers(last, 365, n - 1)])
    pref = rng.dirichlet(np.array(mix) * 20)
    spend = basket * rng.lognormal(0, .25)
    for d in days:
        orders.append((cid, END - pd.Timedelta(days=int(d)), rng.choice(CATS, p=pref),
                       round(float(spend * rng.lognormal(0, .35)), 2),
                       rng.choice(["UPI", "Card", "COD", "Net Banking"], p=[.45, .3, .18, .07])))

pd.DataFrame(customers, columns=["customer_id", "age", "gender", "city", "annual_income_lakh"]
             ).to_csv(DATA / "customers.csv", index=False)
tx = pd.DataFrame(orders, columns=["customer_id", "order_date", "category", "amount", "payment_method"]
                  ).sort_values("order_date").reset_index(drop=True)
tx.insert(0, "order_id", [f"O{i + 1:06d}" for i in range(len(tx))])
tx.to_csv(DATA / "transactions.csv", index=False)
print(f"{N} customers, {len(tx)} transactions written to {DATA}")
