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
        
    # Group dates by (year, month)
    dates = sorted(list(daily_spend.keys()))
    months_dict = {}
    for d in dates:
        ym = (d.year, d.month)
        if ym not in months_dict:
            months_dict[ym] = []
        months_dict[ym].append(d)
        
    months_html = ""
    weekday_labels = ["M", "T", "W", "T", "F", "S", "S"]
    
    for (year, month) in sorted(months_dict.keys()):
        import calendar
        _, num_days = calendar.monthrange(year, month)
        month_name = datetime(year, month, 1).strftime("%B %Y")
        
        m_start = datetime(year, month, 1).date()
        m_end = datetime(year, month, num_days).date()
        
        # Align start date to Monday of that week
        week_start = m_start - timedelta(days=m_start.weekday())
        # Align end date to Sunday of that week
        week_end = m_end + timedelta(days=6 - m_end.weekday())
        
        weeks = []
        curr = week_start
        while curr <= week_end:
            w_days = [curr + timedelta(days=i) for i in range(7)]
            weeks.append(w_days)
            curr += timedelta(days=7)
            
        rows_html = ""
        for day_idx in range(7):
            row_cells = f'<td style="font-family:\'IBM Plex Mono\', monospace; font-size:11px; font-weight:600; color:var(--paper-cream); opacity:0.5; padding-right:4px; text-align:right; vertical-align:middle;">{weekday_labels[day_idx]}</td>'
            for week in weeks:
                date_obj = week[day_idx]
                
                # Check if cell belongs to this month
                if date_obj.year == year and date_obj.month == month:
                    amt = daily_spend.get(date_obj, 0.0)
                    txns = daily_txns.get(date_obj, [])
                    
                    if amt == 0:
                        bg_color = "#2c3b37"
                    elif amt < 500:
                        bg_color = "rgba(124, 152, 133, 0.35)"
                    elif amt < 2000:
                        bg_color = "rgba(124, 152, 133, 0.6)"
                    elif amt < 10000:
                        bg_color = "rgba(124, 152, 133, 0.85)"
                    else:
                        bg_color = "rgb(124, 152, 133)"
                        
                    txn_rows_html = ""
                    for t in txns[:5]:
                        txn_rows_html += f'<div class="mini-receipt-row"><span>{t["description"][:16]}</span><span>₹{t["amount"]:,.0f}</span></div>'
                    if len(txns) > 5:
                        txn_rows_html += f'<div class="mini-receipt-row" style="opacity:0.6;"><span>... +{len(txns)-5} more</span></div>'
                        
                    tooltip_html = f'<span class="tooltip"><div class="mini-receipt-title">{date_obj.strftime("%b %d, %Y")}</div>{txn_rows_html if txns else "<div class=\'mini-receipt-row\' style=\'opacity:0.6;\'>No activity</div>"}<div class="mini-receipt-total"><span>Total Spend</span><span>₹{amt:,.2f}</span></div></span>'
                    row_cells += f'<td class="heatmap-cell" style="width:16px; height:16px; background:{bg_color}; border-radius:2px; position:relative;">{tooltip_html}</td>'
                else:
                    row_cells += f'<td style="width:16px; height:16px; background:transparent;"></td>'
            rows_html += f'<tr>{row_cells}</tr>'
            
        month_spend = sum(daily_spend.get(d, 0.0) for d in months_dict[(year, month)])
        months_html += f'<div style="display:inline-block; margin:0 12px 16px 0; vertical-align:top; background:rgba(27,42,38,0.4); padding:12px; border-radius:8px; border:1px solid rgba(242,236,221,0.1);"><div style="font-family:\'Space Grotesk\', sans-serif; font-size:13px; font-weight:700; color:var(--gold); margin-bottom:8px; display:flex; justify-content:space-between;"><span>📅 {month_name}</span><span style="color:var(--paper-cream); font-size:12px; margin-left:12px;">₹{month_spend:,.2f}</span></div><table style="border-collapse:separate; border-spacing:3px;"><tbody>{rows_html}</tbody></table></div>'
        
    return f'<div class="heatmap-scroll-container" style="white-space:nowrap; overflow-x:auto; padding:4px 0;">{months_html}</div>'
