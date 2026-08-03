<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1B2A26,50:2E4A3E,100:C1502E&height=220&section=header&text=Smart%20Expense%20Tracker&fontSize=44&fontColor=F2ECDD&fontAlignY=38&desc=Ledger%20Journal%20Edition&descAlignY=58&descSize=18&animation=fadeIn" width="100%"/>

<img src="https://readme-typing-svg.demolab.com/?font=IBM+Plex+Mono&size=20&duration=3000&pause=800&color=D4AF37&center=true&vCenter=true&width=700&lines=AI-Powered+Personal+Finance+Ledger;TF-IDF+%2B+Naive+Bayes+Categorization;Isolation+Forest+Anomaly+Detection;Google+OAuth+%2B+Family+Ledger+Groups;Multi-Turn+Conversational+AI+Assistant" alt="Typing SVG" />

<br/>

[![Live App](https://img.shields.io/badge/🚀_Live_App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://fintech-dyx8u4tykpcj9flgpn4jnt.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-1B2A26?style=for-the-badge&logo=python&logoColor=D4AF37)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-7C9885?style=for-the-badge)](#license)
[![Made with Love](https://img.shields.io/badge/Made%20with-%E2%9D%A4-C1502E?style=for-the-badge)](#)

<img src="https://capsule-render.vercel.app/api?type=rect&color=1B2A26&height=3&width=100%" width="100%"/>

</div>

<br/>

## 📖 About

**Smart Expense Tracker — Ledger Journal Edition** is an AI-powered personal finance ledger wrapped in the aesthetic of a physical ledger notebook. It combines classical ML (transaction categorization, anomaly detection, spend forecasting) with a conversational AI assistant, multi-account data isolation, and full family/group ledger sharing — all in a single Streamlit app.

<br/>

<div align="center">

### 🌐 Live Demo

**[👉 fintech-dyx8u4tykpcj9flgpn4jnt.streamlit.app](https://fintech-dyx8u4tykpcj9flgpn4jnt.streamlit.app/)**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fintech-dyx8u4tykpcj9flgpn4jnt.streamlit.app/)

</div>

<br/>

## ✨ Feature Highlights

<table>
<tr>
<td width="50%" valign="top">

### 🔑 Authentication & Data Isolation
- Mandatory **Google OAuth** login gate via Google Identity API
- Unique **8-character Ledger ID** per user
- Fully isolated CSV datastores — no cross-user data leakage

</td>
<td width="50%" valign="top">

### 👨‍👩‍👧 Family Groups & Invitations
- Host a Family / Group ledger
- Invite by **email, username, or Ledger ID**
- Accept / Decline flow with dashboard notification cards
- Switch between multiple groups with per-member subtotal breakdowns

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🤖 Multi-Turn AI Assistant
- Persistent conversation memory across turns
- Contextual pronoun resolution (*"what about the week before?"*)
- Direct pandas query engine, with Gemini API fallback

</td>
<td width="50%" valign="top">

### 📝 Manual Entry & Gap Backfilling
- Quick-select category buttons (Food, Travel, Bills, etc.)
- Automatic **gap detection** for missed days
- Backfill entries or mark as ₹0 No-Spend

</td>
</tr>
</table>

### 🧠 Machine Learning Core

| Capability | Technique |
|---|---|
| Transaction Categorization | TF-IDF + Naive Bayes |
| Anomaly Detection | Isolation Forest |
| Spend Forecasting | Linear Regression |

<br/>

## 🎨 Visual Identity

<div align="center">

| Deep Ledger | Aged Paper | Stamped Ink | Sage | Gold Accent |
|:---:|:---:|:---:|:---:|:---:|
| ![#1B2A26](https://placehold.co/80x30/1B2A26/1B2A26.png) `#1B2A26` | ![#F2ECDD](https://placehold.co/80x30/F2ECDD/F2ECDD.png) `#F2ECDD` | ![#C1502E](https://placehold.co/80x30/C1502E/C1502E.png) `#C1502E` | ![#7C9885](https://placehold.co/80x30/7C9885/7C9885.png) `#7C9885` | ![#D4AF37](https://placehold.co/80x30/D4AF37/D4AF37.png) `#D4AF37` |

*Typography: Space Grotesk for titles · IBM Plex Mono for data & labels*

</div>

<br/>

## 🛠️ Tech Stack

<div align="center">
<img src="https://skillicons.dev/icons?i=python,streamlit,sklearn,pandas,gcp,css,git,github&theme=dark" />
</div>

<br/>

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/deva2006923/FINTECH.git
cd FINTECH

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

> 🔗 **Try it live:** [fintech-dyx8u4tykpcj9flgpn4jnt.streamlit.app](https://fintech-dyx8u4tykpcj9flgpn4jnt.streamlit.app/)

<br/>

## 📂 Project Structure

```
FINTECH/
├── app.py              # Main Streamlit application & UI
├── assistant.py        # Multi-turn conversational AI assistant
├── auth.py             # Google OAuth authentication & session handling
├── helpers.py          # Shared utility functions
├── ml_pipeline.py       # Categorization, anomaly detection, forecasting
├── style.css            # Ledger Journal visual theme
├── requirements.txt
└── README.md
```

<br/>

## 🗺️ Roadmap

- [ ] Export ledger to PDF statement
- [ ] Budget goals & alerts
- [ ] Mobile-optimized layout
- [ ] Recurring transaction detection

<br/>

## 📄 License

This project is licensed under the **MIT License**.

<br/>

<div align="center">

### 👤 Author

**Deva Prakassh** · [@deva2006923](https://github.com/deva2006923)

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:C1502E,50:2E4A3E,100:1B2A26&height=120&section=footer" width="100%"/>

</div>
