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

### 2. Daily Entry with Gap Detection & Backfilling
Log everyday expenses while keeping data history clean and seamless:
*   **Interactive Entry Form**: Add expenses (date, description, amount, category) directly via the sidebar.
*   **Gap Detection Engine**: Compares the entry date against the last recorded transaction. If days were missed, the app pauses rendering and prompts you to resolve each skipped date.
*   **Zero-Spend Backfilling**: For each skipped date, choose to record a custom purchase or mark the day as ₹0 (No Spend). This preserves historical continuity for forecasting.

### 3. Ledger History Views (View Mode Toggle)
Switch between three analysis views just above the ledger list:
*   **Table List View**: Default sortable list with interactive category and date-range filters.
*   **Calendar Heatmap View**: A custom CSS contribution heatmap card displaying spend intensity in sage green levels, featuring hover-triggered mini-receipt tooltips.
*   **Spending Trend Chart**: Interactive line and bar charts tracking daily outflow over time.

### 4. Conversational AI Assistant
An invoice-styled chat helper stapled directly to the right side of the dashboard:
*   **Rule-Based NLP Router**: Instantly parses query intents (sums, comparisons, top lists, anomalies) and executes corresponding pandas queries.
*   **Gemini API Integration**: Supply a Gemini API Key in the sidebar to ask open-ended questions (e.g. advice on budgeting, custom analysis).
*   **Local Advisor Fallback**: If no key is set, the system automatically falls back to an offline rule-based advisor generating detailed statistical summaries and budget recommendations.

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
                                                                         |
                                                                         v
                                                            +--------------------------+
                                                            |   Conversational AI      |
                                                            |   Assistant (Gemini/Local) |
                                                            +--------------------------+
```

1.  **Stage 1: Text Categorization (TF-IDF + Naive Bayes)**: Uses term frequency-inverse document frequency vectorization paired with a Naive Bayes classifier trained on defined category keyword rules to label descriptions. Runs model metrics validation using an 80/20 train/test split.
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
