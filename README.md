# Smart Expense Tracker — Ledger Edition

AI-powered expense tracker with transaction categorization (TF-IDF + Naive Bayes), anomaly detection (Isolation Forest), and spend forecasting (Linear Regression), wrapped in a custom "ledger/receipt" visual identity — not a default Streamlit or AI-generated template look.

## Features & Extensions

### 1. Robust CSV Input Validation
Upload transactions safely. The app checks every uploaded ledger file against rules tailored to prevent pipeline failures:
*   **Column Matching**: Verifies the presence of `date`, `description`, and `amount` (case-insensitive).
*   **Date Formats**: Confirms dates are valid, non-null, and standard parseable formats (e.g. YYYY-MM-DD).
*   **Data Consistency**: Disallows empty cells or descriptions.
*   **Audit Check**: Detects negative amount values and blocks malformed lines.
*   *Validation failures trigger a custom styled error panel that mirrors the app's physical design system without breaking execution.*

### 2. Transaction List Filters
A multi-faceted filter system placed just above the ledger table:
*   **Category Filter**: A dropdown that filters records to show single categories or all categories.
*   **Date Range Filter**: An interactive calendar range selection to filter historical data.
*   *Summary analytics and forecasts remain global to maintain total context while the table filters dynamically.*

### 3. Context-Aware Anomaly Explanations
Instead of simply flagging anomalies, the app displays the specific reasoning behind the outlier status inside the Anomaly Flags panel:
*   **Category Normal Baseline**: Averages are calculated dynamically based strictly on normal historical transactions for that specific category.
*   **Deviation Explanations**: Flags display ratio benchmarks (e.g., `3.2x higher than Food average` or `5.0x lower than Bills average`).
*   **Descriptor Flags**: Identifies rare/unseen merchants (e.g. "Cash Withdrawal") and tags them with description-based alerts.

---

## Machine Learning Pipeline Architecture

```
+-------------------+      +-------------------------+      +--------------------------+
|  Raw Transaction  | ---> |   Categorizer (Naive    | ---> | Anomaly Detector (Isol.  |
|   (Description)   |      |   Bayes + TF-IDF Vector)|      | Forest on Amt/Day-of-Mo) |
+-------------------+      +-------------------------+      +--------------------------+
                                                                         |
                                                                         v
+-------------------+      +-------------------------+      +--------------------------+
|  Forecast (Linear | <--- |   Daily Totals Agg.     | <--- |   Anomalous Transaction  |
| Regression on t)  |      |   (Sum by Date)         |      |   Explanations & Flags   |
+-------------------+      +-------------------------+      +--------------------------+
```

1.  **Stage 1: Text Categorization (TF-IDF + Naive Bayes)**: Uses term frequency-inverse document frequency vectorization paired with a Naive Bayes classifier trained on defined category keyword rules to label descriptions.
2.  **Stage 2: Anomaly Detection (Isolation Forest)**: Trains an Isolation Forest on transaction amounts and days of the month to isolate multivariate outlier records.
3.  **Stage 3: Spend Forecasting (Linear Regression)**: Aggregates daily transaction totals and projects spending trends forward by fitting an OLS linear trend.

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

No CSV needed to try it — click **"Use sample data"** in the sidebar to generate 90 days of synthetic transactions (with injected anomalies) for immediate demo.

## Bring Your Own Data

Upload a CSV file with the following header template:

| column      | type   | example              |
|-------------|--------|----------------------|
| date        | date   | 2026-07-14           |
| description | text   | "Swiggy Order"       |
| amount      | number | 452.50               |

---

## Design System (Lock Restrictions)

*   **Palette**: Deep ledger green-black (`#1B2A26`) background, aged-paper cream (`#F2ECDD`) cards, stamped-ink red (`#C1502E`) for alerts, sage (`#7C9885`) for positive states, muted gold (`#D4AF37`) for highlights.
*   **Typography**: `IBM Plex Mono` for mono numbers/amounts, `Space Grotesk` for titles and labels.
*   **Signature Element**: The torn-edge receipt card showing total spend and category breakdown, built with a CSS `clip-path` zig-zag.
