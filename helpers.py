import pandas as pd
from datetime import datetime, timedelta

def validate_csv(uploaded_file):
    try:
        # Read file without parsing dates yet to prevent initial crash
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return False, None, f"❌ PARSE ERROR: The uploaded file is not a valid CSV format or is corrupted. Details: {str(e)}"
    
    if df.empty:
        return False, None, "❌ LEDGER ERROR: The uploaded ledger file is empty. Please enter valid transaction lines."
    
    # Normalize column names to lowercase to be user friendly
    df.columns = [c.lower() for c in df.columns]
    
    required_cols = ["date", "description", "amount"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, None, f"❌ LEDGER ERROR: Missing required column(s): {', '.join(missing_cols)}. Please verify your column headers."
    
    # Validate date
    if df["date"].isna().any():
        return False, None, "❌ ENTRY ERROR: Missing date(s) found. All ledger lines must have dates."
    try:
        parsed_dates = pd.to_datetime(df["date"], errors="coerce")
        if parsed_dates.isna().any():
            return False, None, "❌ ENTRY ERROR: Invalid or unparseable date format(s) found in the ledger. Dates must be in YYYY-MM-DD or standard parseable format."
        # Assign parsed datetime
        df["date"] = parsed_dates.dt.date
    except Exception as e:
        return False, None, f"❌ ENTRY ERROR: Date validation failed. Details: {str(e)}"
    
    # Validate description
    if df["description"].isna().any() or (df["description"].astype(str).str.strip() == "").any():
        return False, None, "❌ ENTRY ERROR: Missing transaction description(s) found. All ledger lines must have descriptions."
    
    # Validate amount
    if df["amount"].isna().any():
        return False, None, "❌ ENTRY ERROR: Missing transaction amount(s) found. All ledger lines must have amounts."
    
    # Ensure numeric
    if not pd.api.types.is_numeric_dtype(df["amount"]):
        try:
            df["amount"] = pd.to_numeric(df["amount"], errors="raise")
        except Exception:
            return False, None, "❌ VALUE ERROR: Non-numeric values found in the amount column."
    
    # Check for negative amounts
    negative_count = (df["amount"] < 0).sum()
    if negative_count > 0:
        return False, None, f"❌ AUDIT ERROR: Negative amount(s) detected ({negative_count} occurrences). Only positive expense records are accepted in this ledger."
    
    return True, df, None

def generate_heatmap_html(df):
    df_heatmap = df.copy()
    df_heatmap["date_parsed"] = pd.to_datetime(df_heatmap["date"]).dt.date
    
    daily_spend = df_heatmap.groupby("date_parsed")["amount"].sum().to_dict()
    
    daily_txns = {}
    for date_obj, group in df_heatmap.groupby("date_parsed"):
        daily_txns[date_obj] = group[["description", "amount"]].to_dict("records")
        
    if not daily_spend:
        return "<div class='mono' style='color:var(--paper-cream); opacity:0.6;'>No transaction data available for heatmap.</div>"
        
    max_date = max(daily_spend.keys())
    min_date = min(daily_spend.keys())
    
    # Start of 90-day window aligned to Monday of the week containing the date 90 days ago
    start_date = max_date - timedelta(days=90)
    start_date = start_date - timedelta(days=start_date.weekday())
    
    # End of the window aligned to Sunday of the week containing max_date
    end_date = max_date + timedelta(days=6 - max_date.weekday())
    
    # Build list of weeks
    current = start_date
    weeks = []
    while current <= end_date:
        week_days = []
        for i in range(7):
            week_days.append(current + timedelta(days=i))
        weeks.append(week_days)
        current += timedelta(days=7)
        
    # Transpose matrix to group by weekday rows (Monday to Sunday)
    rows_html = ""
    weekday_labels = ["M", "T", "W", "T", "F", "S", "S"]
    for day_idx in range(7):
        row_cells = f'<td style="font-family:\'IBM Plex Mono\', monospace; font-size:12px; font-weight:600; color:var(--paper-cream); opacity:0.5; padding-right:8px; text-align:right; vertical-align:middle; line-height:1;">{weekday_labels[day_idx]}</td>'
        for week in weeks:
            date_obj = week[day_idx]
            amt = daily_spend.get(date_obj, 0.0)
            txns = daily_txns.get(date_obj, [])
            
            # Determine color class/opacity based on amount spend levels
            if amt == 0:
                bg_color = "#2c3b37" # Dark forest cell for zero spending
            elif amt < 500:
                bg_color = "rgba(124, 152, 133, 0.25)"
            elif amt < 2000:
                bg_color = "rgba(124, 152, 133, 0.5)"
            elif amt < 10000:
                bg_color = "rgba(124, 152, 133, 0.75)"
            else:
                bg_color = "rgb(124, 152, 133)" # Bright solid sage
                
            txn_rows_html = ""
            for t in txns[:5]:
                txn_rows_html += f'<div class="mini-receipt-row"><span>{t["description"][:16]}</span><span>₹{t["amount"]:,.0f}</span></div>'
            if len(txns) > 5:
                txn_rows_html += f'<div class="mini-receipt-row" style="opacity:0.6;"><span>... +{len(txns)-5} more</span></div>'
                
            tooltip_html = f"""
            <span class="tooltip">
                <div class="mini-receipt-title">{date_obj.strftime('%b %d, %Y')}</div>
                {txn_rows_html if txns else '<div class="mini-receipt-row" style="opacity:0.6;">No activity</div>'}
                <div class="mini-receipt-total"><span>Total Spend</span><span>₹{amt:,.2f}</span></div>
            </span>
            """
            
            if date_obj < min_date or date_obj > max_date:
                row_cells += f'<td style="width:16px; height:16px; background:transparent; border-radius:2px;"></td>'
            else:
                row_cells += f'<td class="heatmap-cell" style="width:16px; height:16px; background:{bg_color}; border-radius:2px; position:relative;">{tooltip_html}</td>'
        rows_html += f'<tr>{row_cells}</tr>'
        
    heatmap_table = f"""
    <div class="heatmap-scroll-container">
        <table style="border-collapse:separate; border-spacing:3px; margin:0 auto;">
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    return heatmap_table
