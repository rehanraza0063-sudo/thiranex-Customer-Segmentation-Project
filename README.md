# Customer Segmentation Project
Thiranex Data Analyst Internship. Segments customers by behaviour and demographics using K-Means.

> **Data note:** the dataset is synthetic (made by `generate_data.py`). The results show the method, not real market findings.

## Run
```bash
pip install -r requirements.txt
python generate_data.py
python segmentation.py
```

## Method
1. 1,500 customers, 16,616 orders (12 months to Oct 2026)
2. Features: age, income, recency, frequency, monetary value, average order value
3. Log transform of skewed features, then standardisation
4. K-Means, checked with elbow and silhouette; PCA for the 2D view

**Choice of k:** silhouette is highest at k=2 (0.40). k=4 (0.28) was chosen because it gives more actionable groups; the elbow flattens after about 3-4. A score of 0.28 means the segments overlap. Results are stable across random seeds (ARI about 1.0).

## Segments
| Segment | Customers | Revenue | Profile | Suggested action |
|---|---|---|---|---|
| Premium Loyalists | 19.7% | 65.9% | Highest income, about 24 orders, bought recently; Electronics, Home | Loyalty tier, early access |
| Family Essential Buyers | 29.4% | 16.7% | Average age 38; Grocery and Home | Bundles, repeat-order reminders |
| Young Trend Shoppers | 29.2% | 15.2% | Average age 25, smaller baskets; Fashion, Beauty | Social campaigns, new arrivals |
| At-Risk / Inactive | 21.7% | 2.2% | About 163 days since last order, about 2 orders | Low-cost win-back offer |

Payment preference is similar across segments (UPI about 45%), so it is not a useful targeting lever here.

## Outputs
- `outputs/charts/`: 7 charts
- `outputs/customer_segments.csv`: one row per customer (can be loaded into Power BI)
- `outputs/segment_profile.csv`, `category_preference_pct.csv`, `payment_preference_pct.csv`, `k_selection.csv`

## Limitations
Synthetic data; K-Means assumes roughly round clusters; suggested actions are untested. Segment names are assigned by rules written for this dataset, so review them if the data changes.
