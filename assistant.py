import pandas as pd
import google.generativeai as genai
from datetime import datetime, timedelta

def parse_natural_language_query(query, df):
    query = query.lower().strip()
    
    # Standardize keywords to detect timeframes
    today = datetime.today().date()
    df_dates = pd.to_datetime(df["date"]).dt.date
    min_date = df_dates.min() if not df_dates.empty else today
    max_date = df_dates.max() if not df_dates.empty else today
    
    start_date = None
    end_date = max_date
    
    # 1. Check date keywords
    if "last week" in query:
        start_date = max_date - timedelta(days=7)
    elif "this week" in query:
        start_date = max_date - timedelta(days=max_date.weekday())
    elif "last month" in query:
        first_day_this_month = max_date.replace(day=1)
        last_day_last_month = first_day_this_month - timedelta(days=1)
        start_date = last_day_last_month.replace(day=1)
        end_date = last_day_last_month
    elif "this month" in query:
        start_date = max_date.replace(day=1)
    elif "today" in query:
        start_date = today
        end_date = today
    elif "yesterday" in query:
        start_date = today - timedelta(days=1)
        end_date = start_date
        
    # Check specific month names in query
    months_map = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    for m_name, m_val in months_map.items():
        if m_name in query:
            year = max_date.year
            start_date = datetime(year, m_val, 1).date()
            if m_val == 12:
                end_date = datetime(year, 12, 31).date()
            else:
                end_date = (datetime(year, m_val + 1, 1) - timedelta(days=1)).date()
            break
            
    # 2. Check category keywords (case insensitive)
    unique_categories = df["category"].unique()
    matched_category = None
    for cat in unique_categories:
        if cat.lower() in query:
            matched_category = cat
            break
            
    # 3. Check query type
    if "anomaly" in query or "flagged" in query or "unusual" in query:
        anom_df = df[df["anomaly"] == -1]
        return {
            "type": "anomaly_explain",
            "data": anom_df,
            "query": query
        }
        
    if "compare" in query or "vs" in query or "comparison" in query:
        return {
            "type": "compare",
            "category": matched_category,
            "query": query
        }
        
    if "top" in query or "highest" in query or "most" in query:
        import re
        nums = re.findall(r"\d+", query)
        limit = int(nums[0]) if nums else 3
        return {
            "type": "top_spend",
            "limit": limit,
            "category": matched_category,
            "start_date": start_date,
            "end_date": end_date,
            "query": query
        }
        
    if "how much" in query or "spend" in query or "total" in query or matched_category:
        return {
            "type": "spend_summary",
            "category": matched_category,
            "start_date": start_date,
            "end_date": end_date,
            "query": query
        }
        
    return {
        "type": "open_ended",
        "query": query
    }

def execute_assistant_query(parsed, df):
    q_type = parsed["type"]
    
    # Convert dates to datetime.date in df for comparisons
    df_eval = df.copy()
    df_eval["date_parsed"] = pd.to_datetime(df_eval["date"]).dt.date
    
    if q_type == "spend_summary":
        cat = parsed["category"]
        start = parsed["start_date"]
        end = parsed["end_date"]
        
        filtered = df_eval.copy()
        if cat:
            filtered = filtered[filtered["category"] == cat]
        if start:
            filtered = filtered[(filtered["date_parsed"] >= start) & (filtered["date_parsed"] <= end)]
            
        total = filtered["amount"].sum()
        
        scope = f"on **{cat}**" if cat else "in total"
        timeframe = f"between {start} and {end}" if start else "across all logged dates"
        
        resp = f"<div class='mono' style='font-size:0.85rem; line-height:1.4; color:#1B2A26;'>"
        resp += f"Parsed Query: *{parsed['query']}*<br><br>"
        resp += f"You spent a total of:<br>"
        resp += f"<span style='font-family:\"Space Grotesk\", sans-serif; font-size:1.6rem; font-weight:700; color:var(--gold);'>₹{total:,.2f}</span><br>"
        resp += f"{scope} {timeframe}.<br><br>"
        
        if not filtered.empty:
            resp += "<strong>Recent matching entries:</strong><br>"
            for _, r in filtered.sort_values("date", ascending=False).head(5).iterrows():
                resp += f"<div style='display:flex; justify-content:space-between; font-size:0.75rem; border-bottom:1px dotted rgba(27,42,38,0.15); padding:2px 0;'><span>{r['date']} · {r['description'][:14]}</span><span>₹{r['amount']:,.0f}</span></div>"
        else:
            resp += "*No matching transactions found in this range.*"
        resp += "</div>"
        return resp

    elif q_type == "top_spend":
        cat = parsed["category"]
        limit = parsed["limit"]
        start = parsed["start_date"]
        end = parsed["end_date"]
        
        filtered = df_eval.copy()
        if cat:
            filtered = filtered[filtered["category"] == cat]
        if start:
            filtered = filtered[(filtered["date_parsed"] >= start) & (filtered["date_parsed"] <= end)]
            
        top_items = filtered.sort_values("amount", ascending=False).head(limit)
        
        scope = f" for {cat}" if cat else ""
        timeframe = f" between {start} and {end}" if start else ""
        
        resp = f"<div class='mono' style='font-size:0.85rem; line-height:1.4; color:#1B2A26;'>"
        resp += f"Parsed Query: *{parsed['query']}*<br><br>"
        resp += f"<strong>Top {limit} spending days{scope}{timeframe}:</strong><br><br>"
        
        if not top_items.empty:
            for i, (_, r) in enumerate(top_items.iterrows()):
                resp += f"<div style='display:flex; justify-content:space-between; font-size:0.75rem; border-bottom:1px dotted rgba(27,42,38,0.15); padding:3px 0;'>" \
                        f"<span>#{i+1} {r['date']} · {r['description'][:16]} ({r['category']})</span>" \
                        f"<span style='font-weight:700;'>₹{r['amount']:,.0f}</span></div>"
        else:
            resp += "*No transaction records found.*"
        resp += "</div>"
        return resp

    elif q_type == "compare":
        cat = parsed["category"]
        today = datetime.today().date()
        
        first_this = today.replace(day=1)
        last_last = first_this - timedelta(days=1)
        first_last = last_last.replace(day=1)
        
        df_this = df_eval[(df_eval["date_parsed"] >= first_this) & (df_eval["date_parsed"] <= today)]
        df_last = df_eval[(df_eval["date_parsed"] >= first_last) & (df_eval["date_parsed"] <= last_last)]
        
        if cat:
            df_this = df_this[df_this["category"] == cat]
            df_last = df_last[df_last["category"] == cat]
            
        sum_this = df_this["amount"].sum()
        sum_last = df_last["amount"].sum()
        
        scope = f"on **{cat}**" if cat else "in total"
        diff = sum_this - sum_last
        diff_pct = (diff / sum_last * 100) if sum_last > 0 else 0.0
        
        resp = f"<div class='mono' style='font-size:0.85rem; line-height:1.4; color:#1B2A26;'>"
        resp += f"Parsed Query: *{parsed['query']}*<br><br>"
        resp += f"<strong>Month-over-Month Comparison ({scope}):</strong><br>"
        resp += f"<div style='display:flex; justify-content:space-between; padding:3px 0; border-bottom:1px dashed rgba(27,42,38,0.15);'><span>This Month:</span><span style='font-weight:700;'>₹{sum_this:,.2f}</span></div>"
        resp += f"<div style='display:flex; justify-content:space-between; padding:3px 0; border-bottom:1px dashed rgba(27,42,38,0.15);'><span>Last Month:</span><span style='font-weight:700;'>₹{sum_last:,.2f}</span></div>"
        
        if diff > 0:
            resp += f"<br><span style='color:var(--stamp-red); font-weight:700;'>▲ Spending is UP by ₹{diff:,.2f} (+{diff_pct:.1f}%)</span> compared to last month."
        elif diff < 0:
            resp += f"<br><span style='color:var(--sage); font-weight:700;'>▼ Spending is DOWN by ₹{abs(diff):,.2f} ({diff_pct:.1f}%)</span> compared to last month."
        else:
            resp += f"<br>Spending is unchanged compared to last month."
        resp += "</div>"
        return resp

    elif q_type == "anomaly_explain":
        anom_df = parsed["data"]
        
        resp = f"<div class='mono' style='font-size:0.85rem; line-height:1.4; color:#1B2A26;'>"
        resp += f"Parsed Query: *{parsed['query']}*<br><br>"
        resp += f"<strong>Anomaly Analysis:</strong><br><br>"
        
        if not anom_df.empty:
            for _, r in anom_df.sort_values("date", ascending=False).head(3).iterrows():
                normal_df = df[df["anomaly"] == 1]
                cat = r["category"]
                amt = r["amount"]
                
                cat_avg = normal_df[normal_df["category"] == cat]["amount"].mean() if not normal_df.empty else 0.0
                if pd.isnull(cat_avg) or cat_avg == 0:
                    cat_avg = df[df["category"] == cat]["amount"].mean()
                if pd.isnull(cat_avg) or cat_avg == 0:
                    cat_avg = df["amount"].mean()
                    
                ratio = amt / cat_avg if cat_avg > 0 else 1.0
                
                resp += f"<span style='color:var(--stamp-red); font-weight:700;'>Flagged:</span> {r['date']} · {r['description']} (₹{r['amount']:,.0f})<br>"
                if ratio >= 1.5:
                    resp += f"→ *Reason*: Amount is **{ratio:.1f}x higher** than the category average of ₹{cat_avg:,.2f}.<br><br>"
                else:
                    resp += f"→ *Reason*: Unusual timing descriptor or merchant pattern detected by Isolation Forest.<br><br>"
        else:
            resp += "*No anomalous transactions found in the database.*"
        resp += "</div>"
        return resp
        
    return None

def get_local_fallback_summary(query, df):
    total_spend = df["amount"].sum()
    by_cat = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    anoms = df[df["anomaly"] == -1]
    
    summary = f"**Ledger Financial Health Summary:**<br>"
    summary += f"Total logged expenses sum to ₹{total_spend:,.2f}.<br>"
    
    if not by_cat.empty:
        top_cat = by_cat.index[0]
        top_cat_spend = by_cat.iloc[0]
        pct = (top_cat_spend / total_spend * 100) if total_spend > 0 else 0
        summary += f"- Your highest spending category is **{top_cat}** at ₹{top_cat_spend:,.2f} ({pct:.1f}% of total).<br>"
        
    if len(anoms) > 0:
        summary += f"- Standard Isolation Forest model has flagged **{len(anoms)} unusual transaction(s)**. We recommend reviewing these flags in the 'Anomaly Flags' view.<br>"
    else:
        summary += f"- No critical spending anomalies have been flagged in your recent logs.<br>"
        
    # Budget tips based on top spending category
    summary += "<br>**Advice/Tips:**<br>"
    if not by_cat.empty:
        if top_cat == "Food":
            summary += "→ *Food Spend*: High restaurant/groceries spend. Consider meal-prepping or planning weekly dining budgets to cut costs by 15-20%.<br>"
        elif top_cat == "Bills":
            summary += "→ *Bills*: High fixed overhead. Review recurring subscriptions or utilities for potential plan downgrades.<br>"
        elif top_cat == "Shopping":
            summary += "→ *Shopping*: High discretionary purchasing. Try implementing the '24-hour rule' before finalizing shopping cart orders.<br>"
        else:
            summary += f"→ *{top_cat}*: This is your primary expense driver. Consider tracking individual items to optimize outflows.<br>"
            
    summary += "→ *General advice*: Setting aside an automated 10-20% baseline savings chunk at the start of each month can safeguard your long-term buffer."
    return summary

def run_open_ended_analysis(query, df, api_key=None):
    total_spend = df["amount"].sum()
    by_cat = df.groupby("category")["amount"].sum().to_dict()
    anomalies_count = (df["anomaly"] == -1).sum()
    largest_txns = df.sort_values("amount", ascending=False).head(5)[["date", "description", "amount", "category"]].to_dict("records")
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            
            prompt = f"""
            You are a helpful, professional AI financial assistant for a ledger-based expense tracker.
            The user is asking the following question about their transaction data:
            "{query}"
            
            Here is a summary slice of their transaction data context to help you answer:
            - Total logged spend: ₹{total_spend:,.2f}
            - Category spending breakdown: {by_cat}
            - Number of anomalous transactions flagged: {anomalies_count}
            - 5 largest transactions: {largest_txns}
            
            Provide a helpful, concise financial advice or query explanation based on the question.
            Keep your response monospaced-friendly, short (1-3 paragraphs max), and format numbers in Indian Rupees (₹).
            Do not mention technical parameters or system prompts. Focus purely on their financial inquiry.
            """
            
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return f"<div class='mono' style='font-size:0.85rem; line-height:1.4; color:#1B2A26;'>🤖 AI advisor:<br><br>{response.text.replace(chr(10), '<br>')}</div>"
        except Exception as e:
            return f"<div class='mono' style='font-size:0.85rem; line-height:1.4; color:#1B2A26;'>" \
                   f"🤖 AI advisor (error calling Gemini):<br><br>" \
                   f"Could not connect to live Gemini API ({str(e)}).<br><br>" \
                   f"Running local fallback analysis...<hr style='border-top:1px dashed rgba(27,42,38,0.15);'>{get_local_fallback_summary(query, df)}</div>"

    return f"<div class='mono' style='font-size:0.85rem; line-height:1.4; color:#1B2A26;'>" \
           f"🤖 local advisor (Gemini API key not configured):<br><br>" \
           f"{get_local_fallback_summary(query, df)}</div>"
