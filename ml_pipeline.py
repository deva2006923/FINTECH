import pandas as pd
import numpy as np
import streamlit as st
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

# ----------------------------------------------------------------------
# ML: CATEGORIZATION (TF-IDF + Naive Bayes, trained on keyword rules as labels)
# ----------------------------------------------------------------------
@st.cache_resource
def train_categorizer(df):
    labels = df["true_category"] if "true_category" in df.columns else None
    if labels is None:
        return None, None, None
        
    df_clean = df.dropna(subset=["description", "true_category"])
    if len(df_clean) < 10:
        return None, None, None
        
    # We do a train/test split to evaluate our categorizer
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df_clean["description"].astype(str).str.lower(),
        df_clean["true_category"],
        test_size=0.2,
        random_state=42,
        stratify=df_clean["true_category"] if df_clean["true_category"].nunique() > 1 else None
    )
    
    vec = TfidfVectorizer()
    X_train = vec.fit_transform(X_train_text)
    
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    
    # Evaluate
    X_test = vec.transform(X_test_text)
    y_pred = clf.predict(X_test)
    
    unique_labels = sorted(list(set(y_test) | set(y_pred)))
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "labels": unique_labels,
        "confusion_matrix": cm.tolist()
    }
    
    # Retrain on full dataset to maximize classification power for the actual table categorization
    X_full = vec.fit_transform(df_clean["description"].astype(str).str.lower())
    clf_full = MultinomialNB()
    clf_full.fit(X_full, df_clean["true_category"])
    
    return vec, clf_full, metrics

def rule_based_category(desc):
    d = str(desc).lower().strip()
    if any(w in d for w in ["electric", "bill", "water", "wifi", "internet", "recharge", "mobile", "rent", "utility"]):
        return "Bills"
    if any(w in d for w in ["food", "coffee", "starbucks", "dinner", "lunch", "swiggy", "zomato", "restaurant", "grocery", "cafe", "pizza", "burger"]):
        return "Food"
    if any(w in d for w in ["uber", "ola", "cab", "petrol", "fuel", "flight", "train", "bus", "metro", "travel", "irctc", "parking"]):
        return "Travel"
    if any(w in d for w in ["amazon", "flipkart", "clothes", "shopping", "mall", "store", "myntra", "shoes", "purchase"]):
        return "Shopping"
    if any(w in d for w in ["movie", "netflix", "spotify", "game", "cinema", "entertainment", "theatre", "event"]):
        return "Entertainment"
    if any(w in d for w in ["doctor", "hospital", "pharmacy", "medicine", "health", "gym", "clinic", "fitness"]):
        return "Health"
    return None

def categorize(df, vec, clf):
    if df.empty:
        df["category"] = pd.Series(dtype=str)
        return df
        
    categories = []
    for _, row in df.iterrows():
        existing_cat = str(row.get("category", "")).strip()
        desc = str(row.get("description", ""))
        
        # 1. Rule-based keyword matching
        rule_cat = rule_based_category(desc)
        if rule_cat:
            categories.append(rule_cat)
        elif existing_cat and existing_cat not in ["Uncategorized", "nan", "None", ""]:
            categories.append(existing_cat)
        elif vec is not None and clf is not None:
            try:
                X = vec.transform([desc.lower()])
                pred = clf.predict(X)[0]
                categories.append(pred)
            except Exception:
                categories.append("Other")
        else:
            categories.append("Other")
            
    df["category"] = categories
    return df

# ----------------------------------------------------------------------
# ML: ANOMALY DETECTION (Isolation Forest on amount + day-of-month)
# ----------------------------------------------------------------------
def detect_anomalies(df):
    if df.empty:
        df["anomaly"] = pd.Series(dtype=int)
        return df
    feats = pd.DataFrame({
        "amount": df["amount"],
        "day": pd.to_datetime(df["date"]).dt.day,
    })
    model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = model.fit_predict(feats)
    return df

# ----------------------------------------------------------------------
# ML: FORECAST (simple linear trend on daily totals)
# ----------------------------------------------------------------------
def forecast_next_period(df, days_ahead=30):
    if df.empty:
        return pd.DataFrame(columns=["date", "amount"]), pd.DataFrame(columns=["date", "amount"])
        
    daily = df.groupby("date")["amount"].sum().reset_index()
    if len(daily) < 2:
        return daily, pd.DataFrame(columns=["date", "amount"])
        
    daily["t"] = (pd.to_datetime(daily["date"]) - pd.to_datetime(daily["date"]).min()).dt.days
    X = daily[["t"]].values
    y = daily["amount"].values
    model = LinearRegression().fit(X, y)
    future_t = np.arange(daily["t"].max() + 1, daily["t"].max() + 1 + days_ahead).reshape(-1, 1)
    preds = model.predict(future_t)
    last_date = pd.to_datetime(daily["date"]).max()
    future_dates = [last_date + timedelta(days=int(i) + 1) for i in range(days_ahead)]
    forecast_df = pd.DataFrame({"date": future_dates, "amount": np.clip(preds, 0, None)})
    return daily, forecast_df
