# Smart Expense Tracker — Ledger Journal Edition

AI-powered financial ledger with transaction categorization (TF-IDF + Naive Bayes), anomaly detection (Isolation Forest), spend forecasting (Linear Regression), Google OAuth multi-account data scoping, family groups & invitations, and multi-turn continuous AI assistant — all wrapped in a custom "Ledger Journal" aesthetic.

---

## 🔑 Authentication & Data Isolation

- **Google OAuth Login**: Mandatory authentication gate using Google Identity API.
- **Per-User Data Scoping**: Every user receives a unique 8-character **Ledger ID** upon sign-in. All ledger entries are stored in isolated CSV datastores (`ledger_{user_id}.csv`). No user ever sees another user's personal transactions.

---

## 👨‍👩‍👧 Family Groups & Invitations Dashboard

- **Group Hosting**: Any user can host a Family/Group ledger.
- **Invitation System**: Send invitations by **email**, **username**, or 8-character **Ledger ID**.
- **Accept/Decline Flow**: Invited users receive dashboard notification cards with `Accept` and `Decline` buttons. Data is only shared after explicit acceptance.
- **Multi-Group Switching**: Belong to multiple groups (e.g. Family, Roommates) and switch views dynamically. Group views include a per-member subtotal breakdown alongside merged analytics.

---

## 🤖 Multi-Turn Continuous AI Assistant

- **Conversation Memory**: Maintains continuous multi-turn chat history (`st.session_state.chat_history`).
- **Contextual Pronoun Resolution**: Understands queries referencing prior context, such as *"How much did I spend on food last week?"* followed by *"What about the week before?"* or *"Which of those was the highest?"*.
- **Data Engine + Gemini Fallback**: Directly executes pandas queries on user/group dataframes, with seamless fallback to Gemini API or local analytical engines.

---

## 📝 Manual Daily Entry & Gap Backfilling

- **User-Driven Entry**: Date (defaults to today), amount, quick-select category buttons (Food, Travel, Bills, Shopping, Entertainment, Health, Rent, Other).
- **Gap Detection Engine**: Detects missed transaction dates since last entry and prompts user to backfill or mark as ₹0 (No Spend).

---

## 🎨 Visual Identity & Design Tokens

- **Aesthetic**: Physical financial ledger notebook.
- **Palette**: Deep ledger green-black (`#1B2A26`), aged-paper cream (`#F2ECDD`), stamped-ink red (`#C1502E`), sage (`#7C9885`), gold accent (`#D4AF37`).
- **Typography Scale**: Page title (32px Space Grotesk 700), Section labels (12px IBM Plex Mono 600 UPPERCASE), Body/Data (15px IBM Plex Mono 400), Big numbers (40px IBM Plex Mono 700).

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

