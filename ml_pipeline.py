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

def categorize(df, vec, clf):
    if vec is None:
        df["category"] = "Uncategorized"
        return df
    X = vec.transform(df["description"].str.lower())
    df["category"] = clf.predict(X)
    return df

# ----------------------------------------------------------------------
# ML: ANOMALY DETECTION (Isolation Forest on amount + day-of-month)
# ----------------------------------------------------------------------
def detect_anomalies(df):
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
    daily = df.groupby("date")["amount"].sum().reset_index()
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
