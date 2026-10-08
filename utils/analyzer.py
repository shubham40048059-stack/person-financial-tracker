"""
Financial analytics, aggregation, and smart insights engine.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def apply_filters(
    df: pd.DataFrame,
    start_date: Optional[Any] = None,
    end_date: Optional[Any] = None,
    type_filter: str = "All",
    category: str = "All",
    search: str = "",
    min_amount: float = 0.0,
    max_amount: float = float("inf"),
) -> pd.DataFrame:
    """
    Filter the financial dataframe based on user sidebar controls.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date", "Description", "Category", "Type", "Income", "Expense", "Balance"])

    filtered = df.copy()

    # Ensure Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(filtered["Date"]):
        filtered["Date"] = pd.to_datetime(filtered["Date"], errors="coerce")
    filtered = filtered.dropna(subset=["Date"])

    # 1. Date Range
    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        filtered = filtered[filtered["Date"] >= start_ts]
    if end_date is not None:
        # Include full end-day
        end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        filtered = filtered[filtered["Date"] <= end_ts]

    # 2. Transaction Type
    if type_filter and type_filter != "All":
        filtered = filtered[filtered["Type"].str.strip().str.lower() == type_filter.strip().lower()]

    # 3. Category
    if category and category != "All":
        filtered = filtered[filtered["Category"].str.strip().str.lower() == category.strip().lower()]

    # 4. Search description
    if search and str(search).strip():
        term = str(search).strip().lower()
        filtered = filtered[filtered["Description"].str.lower().str.contains(term, na=False)]

    # 5. Amount Range (max of Income or Expense for each row)
    row_amount = np.maximum(filtered["Income"], filtered["Expense"])
    if min_amount > 0:
        filtered = filtered[row_amount >= min_amount]
    if np.isfinite(max_amount):
        filtered = filtered[row_amount <= max_amount]

    return filtered.reset_index(drop=True)


def financial_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute core metrics for the dashboard cards:
    Current Balance, Total Income, Total Expenses, Net Cash Flow,
    Number of Transactions, Average Expense.
    """
    defaults = {
        "current_balance": 0.0,
        "total_income": 0.0,
        "total_expenses": 0.0,
        "net_cash_flow": 0.0,
        "transaction_count": 0,
        "average_expense": 0.0,
    }

    if df is None or df.empty:
        return defaults

    total_income = float(df["Income"].sum())
    total_expenses = float(df["Expense"].sum())
    net_cash_flow = total_income - total_expenses
    transaction_count = len(df)

    # Current balance is the last transaction's balance if chronological, else net cash flow
    if "Balance" in df.columns and not df["Balance"].empty:
        current_balance = float(df["Balance"].iloc[-1])
    else:
        current_balance = net_cash_flow

    # Average expense considering expense transactions
    expense_mask = (df["Type"] == "Expense") & (df["Expense"] > 0)
    if expense_mask.any():
        avg_expense = float(df.loc[expense_mask, "Expense"].mean())
    elif total_expenses > 0:
        avg_expense = total_expenses / transaction_count
    else:
        avg_expense = 0.0

    return {
        "current_balance": current_balance,
        "total_income": total_income,
        "total_expenses": total_expenses,
        "net_cash_flow": net_cash_flow,
        "transaction_count": transaction_count,
        "average_expense": avg_expense,
    }


def balance_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return daily closing balance trend dataframe.
    """
    if df is None or df.empty or "Balance" not in df.columns:
        return pd.DataFrame(columns=["Date", "Balance"])

    trend = df[["Date", "Balance"]].copy()
    trend["Date"] = pd.to_datetime(trend["Date"]).dt.date
    # Take the last balance of each day
    trend = trend.groupby("Date", as_index=False)["Balance"].last()
    trend["Date"] = pd.to_datetime(trend["Date"])
    return trend.sort_values("Date").reset_index(drop=True)


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build monthly summary with Income, Expenses, and Net Cash Flow.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Month", "Income", "Expenses", "Net Cash Flow", "Period"])

    frame = df.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Period"] = frame["Date"].dt.to_period("M")

    grouped = frame.groupby("Period").agg(
        Income=("Income", "sum"),
        Expenses=("Expense", "sum"),
        Transactions=("Description", "count")
    ).reset_index()

    grouped["Net Cash Flow"] = grouped["Income"] - grouped["Expenses"]
    grouped["Month"] = grouped["Period"].apply(lambda p: p.strftime("%b %Y"))
    grouped = grouped.sort_values("Period").reset_index(drop=True)

    return grouped


def expense_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate expenses by category for pie/donut charts.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Category", "Amount"])

    expense_df = df[df["Expense"] > 0].copy()
    if expense_df.empty:
        return pd.DataFrame(columns=["Category", "Amount"])

    summary = expense_df.groupby("Category")["Expense"].sum().reset_index()
    summary.columns = ["Category", "Amount"]
    summary = summary.sort_values("Amount", ascending=False).reset_index(drop=True)
    return summary


def monthly_expenses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return total monthly expenses for bar/line charts.
    """
    summary = monthly_summary(df)
    if summary.empty:
        return pd.DataFrame(columns=["Month", "Expenses"])
    return summary[["Month", "Expenses"]].reset_index(drop=True)


def spending_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily spending aggregate over time.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date", "Expense"])

    expense_df = df[df["Expense"] > 0].copy()
    if expense_df.empty:
        return pd.DataFrame(columns=["Date", "Expense"])

    expense_df["Date"] = pd.to_datetime(expense_df["Date"]).dt.date
    daily = expense_df.groupby("Date")["Expense"].sum().reset_index()
    daily["Date"] = pd.to_datetime(daily["Date"])
    return daily.sort_values("Date").reset_index(drop=True)


def top_expenses(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Return top N largest expense transactions.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date", "Description", "Category", "Expense"])

    expense_df = df[df["Expense"] > 0].copy()
    if expense_df.empty:
        return pd.DataFrame(columns=["Date", "Description", "Category", "Expense"])

    top = expense_df.sort_values("Expense", ascending=False).head(top_n).copy()
    return top[["Date", "Description", "Category", "Expense"]].reset_index(drop=True)


def transaction_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return transaction counts grouped by Type (Income vs Expense).
    """
    if df is None or df.empty:
        return pd.DataFrame({"Type": ["Income", "Expense"], "Count": [0, 0]})

    counts = df["Type"].value_counts().reset_index()
    counts.columns = ["Type", "Count"]

    # Ensure both types exist in the result
    for t in ["Income", "Expense"]:
        if t not in counts["Type"].values:
            counts = pd.concat([counts, pd.DataFrame({"Type": [t], "Count": [0]})], ignore_index=True)

    return counts


def monthly_comparison(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare monthly metrics and identify standout months:
    Best cash-flow month, Highest spending month, Highest income month.
    """
    summary = monthly_summary(df)
    if summary.empty:
        return {
            "table": summary,
            "best_cash_flow_month": None,
            "highest_spending_month": None,
            "highest_income_month": None,
        }

    best_cf_idx = summary["Net Cash Flow"].idxmax()
    highest_exp_idx = summary["Expenses"].idxmax()
    highest_inc_idx = summary["Income"].idxmax()

    return {
        "table": summary,
        "best_cash_flow_month": summary.loc[best_cf_idx].to_dict(),
        "highest_spending_month": summary.loc[highest_exp_idx].to_dict(),
        "highest_income_month": summary.loc[highest_inc_idx].to_dict(),
    }


def income_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    In-depth metrics and breakdowns for Income transactions.
    """
    if df is None or df.empty:
        return {
            "total_income": 0.0,
            "count": 0,
            "average": 0.0,
            "largest": 0.0,
            "by_month": pd.DataFrame(columns=["Month", "Income"]),
            "by_category": pd.DataFrame(columns=["Category", "Income"]),
        }

    income_df = df[df["Income"] > 0].copy()
    total_income = float(df["Income"].sum())
    count = len(income_df)
    average = float(income_df["Income"].mean()) if count > 0 else 0.0
    largest = float(income_df["Income"].max()) if count > 0 else 0.0

    # Income by month
    monthly = monthly_summary(df)
    by_month = monthly[["Month", "Income"]].copy() if not monthly.empty else pd.DataFrame(columns=["Month", "Income"])

    # Income by category
    if not income_df.empty:
        by_category = income_df.groupby("Category")["Income"].sum().reset_index().sort_values("Income", ascending=False)
    else:
        by_category = pd.DataFrame(columns=["Category", "Income"])

    return {
        "total_income": total_income,
        "count": count,
        "average": average,
        "largest": largest,
        "by_month": by_month,
        "by_category": by_category,
    }


def expense_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    In-depth metrics and breakdowns for Expense transactions.
    """
    if df is None or df.empty:
        return {
            "total_expenses": 0.0,
            "count": 0,
            "average": 0.0,
            "largest": 0.0,
            "by_month": pd.DataFrame(columns=["Month", "Expenses"]),
            "by_category": pd.DataFrame(columns=["Category", "Amount"]),
            "top_10": pd.DataFrame(columns=["Date", "Description", "Category", "Expense"]),
        }

    expense_df = df[df["Expense"] > 0].copy()
    total_expenses = float(df["Expense"].sum())
    count = len(expense_df)
    average = float(expense_df["Expense"].mean()) if count > 0 else 0.0
    largest = float(expense_df["Expense"].max()) if count > 0 else 0.0

    monthly = monthly_summary(df)
    by_month = monthly[["Month", "Expenses"]].copy() if not monthly.empty else pd.DataFrame(columns=["Month", "Expenses"])
    by_cat = expense_by_category(df)
    top_10 = top_expenses(df, 10)

    return {
        "total_expenses": total_expenses,
        "count": count,
        "average": average,
        "largest": largest,
        "by_month": by_month,
        "by_category": by_cat,
        "top_10": top_10,
    }


def cash_flow_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute cash flow stats and savings rate.
    """
    overview = financial_overview(df)
    total_income = overview["total_income"]
    total_expenses = overview["total_expenses"]
    net_cash_flow = overview["net_cash_flow"]

    rate: Optional[float] = None
    if total_income > 0:
        rate = (net_cash_flow / total_income) * 100.0

    return {
        "total_income": total_income,
        "total_expenses": total_expenses,
        "net_cash_flow": net_cash_flow,
        "cash_flow_rate": rate,
    }


def generate_insights(df: pd.DataFrame) -> List[str]:
    """
    Generate factual, simple English insights derived from transaction data.
    Strictly descriptive; no financial advice or predictions.
    """
    from .helpers import format_inr

    if df is None or df.empty:
        return ["No transaction records are currently available for insight generation."]

    insights: List[str] = []
    overview = financial_overview(df)
    income = overview["total_income"]
    expenses = overview["total_expenses"]
    net_cf = overview["net_cash_flow"]

    # 1. Total income statement
    insights.append(f"Your total income during this period was {format_inr(income)}.")

    # 2. Total expenses statement
    insights.append(f"Your total expenses were {format_inr(expenses)}.")

    # 3. Net cash flow statement
    if net_cf >= 0:
        insights.append(f"Your net cash flow was positive at {format_inr(net_cf)}.")
    else:
        insights.append(f"Your net cash flow was negative at {format_inr(net_cf)}.")

    # 4. Largest spending category
    cat_df = expense_by_category(df)
    if not cat_df.empty:
        top_cat = cat_df.iloc[0]["Category"]
        top_cat_amt = cat_df.iloc[0]["Amount"]
        insights.append(f"Your largest spending category was {top_cat} ({format_inr(top_cat_amt)}).")

    # 5. Average expense
    if overview["average_expense"] > 0:
        insights.append(f"Your average expense was {format_inr(overview['average_expense'])}.")

    # 6. Highest individual expense
    top_exp = top_expenses(df, 1)
    if not top_exp.empty:
        highest_amt = top_exp.iloc[0]["Expense"]
        highest_desc = top_exp.iloc[0]["Description"]
        insights.append(f"Your highest individual expense was {format_inr(highest_amt)} ({highest_desc}).")

    # 7. Expense transaction count
    exp_count = (df["Expense"] > 0).sum()
    insights.append(f"You made {exp_count} expense transactions across this dataset.")

    # 8. Monthly trend insight if multiple months exist
    monthly = monthly_summary(df)
    if len(monthly) >= 2:
        last_m = monthly.iloc[-1]
        prev_m = monthly.iloc[-2]
        diff = last_m["Expenses"] - prev_m["Expenses"]
        if prev_m["Expenses"] > 0:
            pct_change = (abs(diff) / prev_m["Expenses"]) * 100
            direction = "increased" if diff > 0 else "decreased"
            insights.append(
                f"Your spending {direction} by {pct_change:.1f}% in {last_m['Month']} compared with {prev_m['Month']}."
            )

    return insights
