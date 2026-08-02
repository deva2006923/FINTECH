# Smart Expense Tracker — Ledger Edition

AI-powered expense tracker with transaction categorization (TF-IDF + Naive Bayes),
anomaly detection (Isolation Forest), and spend forecasting (Linear Regression),
wrapped in a custom "ledger/receipt" visual identity — not a default Streamlit or
AI-generated template look.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

No CSV needed to try it — click **"Use sample data"** in the sidebar and it
generates 90 days of synthetic transactions (with a few injected anomalies)
so you can demo immediately.

## Bring your own data

Upload a CSV with these columns:

| column      | type   | example              |
|-------------|--------|----------------------|
| date        | date   | 2026-07-14           |
| description | text   | "Swiggy Order"       |
| amount      | number | 452.50               |

## Design system (why it looks the way it does)

- **Palette**: deep ledger green-black (`#1B2A26`) background, aged-paper cream
  (`#F2ECDD`) cards, stamped-ink red (`#C1502E`) for alerts, sage (`#7C9885`)
  for positive states, muted gold (`#D4AF37`) reserved for the signature element.
- **Type**: IBM Plex Mono for every number/amount (receipt feel), Space Grotesk
  for headings and labels.
- **Signature element**: the torn-edge receipt card showing total spend and
  category breakdown, built with a CSS `clip-path` zig-zag — ties directly to
  the "expense tracker" subject rather than a generic KPI card.
- All of this is injected via custom CSS in `app.py` (`CUSTOM_CSS` string) to
  override Streamlit's default theme, which otherwise flattens any design back
  to the standard Streamlit look.

## Deploy

Push this folder to a public GitHub repo, then deploy free on
[Streamlit Community Cloud](https://streamlit.io/cloud) — point it at `app.py`.
