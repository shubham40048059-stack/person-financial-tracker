"""
Data loading and normalization module for CSV and Excel statements.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from .categorizer import categorize_transaction
from .helpers import safe_float

DEFAULT_DEMO_PATH = Path("data/demo_transactions.csv")

# Column alias dictionary for intelligent header detection
COLUMN_ALIASES = {
    "date": [
        "date", "transaction date", "transaction_date", "txn date", "txn_date",
        "value date", "value_date", "posting date", "posting_date",
        "booking date", "booking_date", "trans date", "trans_date"
    ],
    "description": [
        "description", "narration", "particulars", "transaction particulars",
        "transaction_particulars", "details", "memo", "remarks",
        "narrative", "transaction description", "payee"
    ],
    "income": [
        "income", "deposit", "deposits", "credit", "credits", "credit amount",
        "credit_amount", "cr", "cr amount", "cr_amount", "inflow", "received"
    ],
    "expense": [
        "expense", "expenses", "withdrawal", "withdrawals", "debit", "debits",
        "debit amount", "debit_amount", "dr", "dr amount", "dr_amount",
        "outflow", "spent", "paid", "payment"
    ],
    "balance": [
        "balance", "closing balance", "closing_balance", "running balance",
        "running_balance", "available balance", "available_balance",
        "net balance", "bal", "account balance"
    ],
    "type": [
        "type", "transaction type", "transaction_type", "entry type",
        "entry_type", "cr/dr", "dr/cr", "cr_dr", "dr_cr"
    ],
    "category": [
        "category", "spending category", "category name", "sub category", "tag"
    ],
    "amount": [
        "amount", "transaction amount", "transaction_amount", "txn amount",
        "value", "net amount"
    ]
}


def _clean_header(name: Any) -> str:
    """Normalize column header for fuzzy comparison."""
    if name is None:
        return ""
    text = str(name).strip().lower()
    return re.sub(r"[\s_\-\.]+", " ", text).strip()


def _match_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """Find the first matching column name using aliases."""
    normalized_cols = {col: _clean_header(col) for col in columns}
    for candidate in candidates:
        clean_candidate = _clean_header(candidate)
        for original_col, clean_col in normalized_cols.items():
            if clean_col == clean_candidate:
                return original_col
    return None


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intelligently map, clean, and standardize an uploaded or demo dataframe.
    Ensures standard columns: ['Date', 'Description', 'Category', 'Type', 'Income', 'Expense', 'Balance'].
    """
    if df is None or df.empty:
        raise ValueError("The provided file is empty. Please upload a file containing financial records.")

    # Remove completely empty rows or columns
    raw_df = df.dropna(how="all").dropna(axis=1, how="all").copy()
    if raw_df.empty:
        raise ValueError("The provided file does not contain any valid transaction data.")

    cols = list(raw_df.columns)

    # Match standard columns
    date_col = _match_column(cols, COLUMN_ALIASES["date"])
    desc_col = _match_column(cols, COLUMN_ALIASES["description"])
    income_col = _match_column(cols, COLUMN_ALIASES["income"])
    expense_col = _match_column(cols, COLUMN_ALIASES["expense"])
    balance_col = _match_column(cols, COLUMN_ALIASES["balance"])
    type_col = _match_column(cols, COLUMN_ALIASES["type"])
    category_col = _match_column(cols, COLUMN_ALIASES["category"])
    amount_col = _match_column(cols, COLUMN_ALIASES["amount"])

    # Basic requirement: must have at least Date and either (Description or Amount/Income/Expense)
    if date_col is None:
        raise ValueError(
            "We couldn't identify the Date column in this file. "
            "Please ensure your file has a column named Date, Txn Date, or Transaction Date."
        )

    clean_df = pd.DataFrame()

    # 1. Parse Dates
    clean_df["Date"] = pd.to_datetime(raw_df[date_col], errors="coerce")
    clean_df = clean_df.dropna(subset=["Date"]).copy()
    if clean_df.empty:
        raise ValueError(
            "No valid transaction dates could be parsed from the Date column. "
            "Please check the date format in your file."
        )

    # Align raw_df index with clean_df index
    raw_df = raw_df.loc[clean_df.index].copy()

    # 2. Parse Description
    if desc_col is not None:
        clean_df["Description"] = raw_df[desc_col].fillna("Transaction").astype(str).str.strip()
    else:
        clean_df["Description"] = "Transaction"

    # 3. Parse Income and Expense
    has_explicit_income = income_col is not None
    has_explicit_expense = expense_col is not None

    if has_explicit_income and has_explicit_expense:
        clean_df["Income"] = raw_df[income_col].apply(safe_float)
        clean_df["Expense"] = raw_df[expense_col].apply(safe_float)
    elif amount_col is not None:
        raw_amounts = raw_df[amount_col].apply(safe_float)
        if type_col is not None:
            raw_types = raw_df[type_col].astype(str).str.strip().str.upper()
            is_income = raw_types.str.contains("CR|CREDIT|INCOME|DEPOSIT|INFLOW", regex=True)
            clean_df["Income"] = np.where(is_income, raw_amounts.abs(), 0.0)
            clean_df["Expense"] = np.where(~is_income, raw_amounts.abs(), 0.0)
        else:
            # Positive amounts as Income, Negative as Expense
            clean_df["Income"] = np.where(raw_amounts > 0, raw_amounts, 0.0)
            clean_df["Expense"] = np.where(raw_amounts < 0, raw_amounts.abs(), 0.0)
    elif has_explicit_income:
        clean_df["Income"] = raw_df[income_col].apply(safe_float)
        clean_df["Expense"] = 0.0
    elif has_explicit_expense:
        clean_df["Income"] = 0.0
        clean_df["Expense"] = raw_df[expense_col].apply(safe_float)
    else:
        raise ValueError(
            "We couldn't identify the transaction columns in this file. "
            "Please ensure columns for Income/Expense, Credit/Debit, or Amount are present."
        )

    # 4. Determine Type
    if type_col is not None:
        types = raw_df[type_col].astype(str).str.strip().str.capitalize()
        # Normalize to 'Income' or 'Expense'
        clean_df["Type"] = np.where(
            types.str.contains("Cr|Credit|Income|Deposit", case=False, regex=True),
            "Income",
            np.where(clean_df["Income"] > clean_df["Expense"], "Income", "Expense")
        )
    else:
        clean_df["Type"] = np.where(clean_df["Income"] > 0, "Income", "Expense")

    # 5. Determine Category
    if category_col is not None:
        clean_df["Category"] = raw_df[category_col].fillna("").astype(str).str.strip()
        # Fallback to auto-categorizer if empty or 'Other'
        mask_empty = clean_df["Category"].isin(["", "Other", "Unknown", "None", "nan"])
        if mask_empty.any():
            clean_df.loc[mask_empty, "Category"] = [
                categorize_transaction(desc, txn_type)
                for desc, txn_type in zip(clean_df.loc[mask_empty, "Description"], clean_df.loc[mask_empty, "Type"])
            ]
    else:
        clean_df["Category"] = [
            categorize_transaction(desc, txn_type)
            for desc, txn_type in zip(clean_df["Description"], clean_df["Type"])
        ]

    # Ensure Category is never blank
    clean_df["Category"] = clean_df["Category"].replace("", "Other").fillna("Other")

    # 6. Parse or Compute Balance
    # Sort chronologically to ensure accurate balance progression
    clean_df = clean_df.sort_values("Date").reset_index(drop=True)

    if balance_col is not None:
        clean_df["Balance"] = raw_df.loc[clean_df.index, balance_col].apply(safe_float)
        # If balance column was completely zero or empty, calculate cumulative cash flow
        if (clean_df["Balance"] == 0).all() and len(clean_df) > 1:
            clean_df["Balance"] = (clean_df["Income"] - clean_df["Expense"]).cumsum()
    else:
        # Calculate running balance starting from cumulative difference
        clean_df["Balance"] = (clean_df["Income"] - clean_df["Expense"]).cumsum()

    # Reorder standard columns
    final_cols = ["Date", "Description", "Category", "Type", "Income", "Expense", "Balance"]
    return clean_df[final_cols]


def load_demo_data(path: str | Path = DEFAULT_DEMO_PATH) -> pd.DataFrame:
    """Load and normalize the default demo dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Demo transaction dataset not found at: {path}")

    df = pd.read_csv(path)
    return normalize_dataframe(df)


def load_uploaded_data(uploaded_file: Any) -> pd.DataFrame:
    """Load user uploaded CSV or Excel file safely."""
    if uploaded_file is None:
        raise ValueError("Please upload a CSV or Excel file.")

    filename = str(uploaded_file.name).lower()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV, XLSX, or XLS file.")
    except UnicodeDecodeError:
        # Retry CSV with latin1 encoding if utf-8 fails
        if filename.endswith(".csv"):
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin1")
        else:
            raise ValueError("Unable to read the file with standard text encoding.")
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Error reading file: {str(e)}")

    return normalize_dataframe(df)
