"""
Personal Financial Tracker — FinTrack
A modern, portfolio-quality personal financial dashboard built with Streamlit, Pandas, and Plotly.
"""
from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.analyzer import (
    apply_filters,
    balance_trend,
    cash_flow_metrics,
    expense_analysis,
    expense_by_category,
    financial_overview,
    generate_insights,
    income_analysis,
    monthly_comparison,
    monthly_expenses,
    monthly_summary,
    spending_trend,
    top_expenses,
    transaction_distribution,
)
from utils.data_loader import DEFAULT_DEMO_PATH, load_demo_data, load_uploaded_data
from utils.helpers import format_inr

# -----------------------------------------------------------------------------
# Asset Paths
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "assets" / "fintrack-logo.png"
ICON_PATH = APP_DIR / "assets" / "fintrack-icon.png"

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FinTrack | Personal Financial Tracker",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else "💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Visual Design System & Color Palette
# -----------------------------------------------------------------------------
COLOR_PALETTE = {
    "income": "#10B981",       # Emerald Green
    "expense": "#F43F5E",      # Rose Red
    "balance": "#38BDF8",      # Sky Blue
    "neutral": "#64748B",      # Slate Gray
    "accent1": "#8B5CF6",      # Violet
    "accent2": "#F59E0B",      # Amber
    "accent3": "#06B6D4",      # Cyan
    "accent4": "#EC4899",      # Pink
    "category_colors": [
        "#10B981", "#38BDF8", "#F59E0B", "#F43F5E", "#8B5CF6",
        "#06B6D4", "#EC4899", "#14B8A6", "#6366F1", "#F97316",
        "#84CC16", "#A855F7", "#64748B", "#E11D48", "#0EA5E9"
    ]
}


def apply_chart_style(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """Apply SaaS-grade, clean, high-contrast styling to Plotly charts."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=15, color="#F8FAFC", family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
            x=0.01,
            y=0.96,
        ),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color="#94A3B8", size=12, family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
        height=height,
        margin=dict(l=15, r=15, t=55 if title else 25, b=25),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#CBD5E1"),
            bgcolor="rgba(0, 0, 0, 0)",
        ),
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_size=12,
            font_color="#F8FAFC",
            bordercolor="rgba(255, 255, 255, 0.15)",
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(255, 255, 255, 0.05)",
        zerolinecolor="rgba(255, 255, 255, 0.1)",
        showline=True,
        linecolor="rgba(255, 255, 255, 0.12)",
        tickfont=dict(color="#94A3B8", size=11),
    )
    fig.update_yaxes(
        gridcolor="rgba(255, 255, 255, 0.05)",
        zerolinecolor="rgba(255, 255, 255, 0.1)",
        showline=True,
        linecolor="rgba(255, 255, 255, 0.12)",
        tickfont=dict(color="#94A3B8", size=11),
    )
    return fig


# -----------------------------------------------------------------------------
# Data Caching & Session State Management
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_demo_data() -> pd.DataFrame:
    """Load and cache the synthetic demo transactions dataset."""
    return load_demo_data(DEFAULT_DEMO_PATH)


@st.cache_data(show_spinner=False)
def get_logo_base64() -> str:
    """Read and cache base64 representation of the brand logo."""
    if LOGO_PATH.exists():
        try:
            return base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
        except Exception:
            return ""
    return ""


def init_session_state() -> None:
    """Initialize session state parameters for navigation and dataset handling."""
    if "data_source" not in st.session_state:
        st.session_state.data_source = "demo"
    if "data_source_name" not in st.session_state:
        st.session_state.data_source_name = "demo_transactions.csv"
    if "uploaded_df" not in st.session_state:
        st.session_state.uploaded_df = None
    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Dashboard"


def get_active_dataframe() -> pd.DataFrame:
    """Retrieve the active dataframe (either uploaded session data or demo data)."""
    if st.session_state.data_source == "uploaded" and st.session_state.uploaded_df is not None:
        return st.session_state.uploaded_df
    return get_cached_demo_data()


# -----------------------------------------------------------------------------
# High-End SaaS Custom Styling (CSS)
# -----------------------------------------------------------------------------
def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        /* Base page layout & typography */
        html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
            background-color: #0B0F17 !important;
            color: #F8FAFC;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
        }

        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 3.5rem !important;
            max-width: 95rem;
        }

        /* Top header bar */
        .saas-page-header {
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        }
        .saas-page-title {
            font-size: 1.85rem;
            font-weight: 700;
            color: #F8FAFC;
            letter-spacing: -0.025em;
            margin: 0;
            line-height: 1.2;
        }
        .saas-page-desc {
            font-size: 0.92rem;
            color: #94A3B8;
            margin: 0.35rem 0 0 0;
        }

        /* SIDEBAR STYLING */
        [data-testid="stSidebar"] {
            background-color: #101726 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1rem;
            padding-left: 1.1rem;
            padding-right: 1.1rem;
        }

        /* 1. SEPARATE BRAND LOGO CONTAINER */
        .fintrack-brand-container {
            margin-bottom: 1.5rem;
            padding-bottom: 1.25rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }
        .fintrack-brand-card {
            background: #FFFFFF;
            border-radius: 12px;
            padding: 12px 14px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.28);
            border: 1px solid rgba(255, 255, 255, 0.15);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .fintrack-brand-logo-img {
            width: 100%;
            max-width: 220px;
            height: auto;
            display: block;
            object-fit: contain;
        }

        /* Sidebar Section Headers */
        .sidebar-section-title {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #64748B;
            margin-top: 1.25rem;
            margin-bottom: 0.5rem;
        }
        .sidebar-section-title:first-of-type {
            margin-top: 0;
        }

        /* 2. SIDEBAR NAVIGATION MENU */
        [data-testid="stSidebar"] [data-testid="stRadioGroup"] {
            gap: 0.35rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadioOption"] {
            position: relative;
            min-height: 2.6rem;
            padding: 0.65rem 0.85rem !important;
            border-radius: 8px;
            border: 1px solid transparent;
            background: transparent;
            color: #CBD5E1 !important;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        /* Hide default round radio dots */
        [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {
            display: none !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadioOption"] p {
            margin: 0;
            font-size: 0.92rem;
            font-weight: 500;
            line-height: 1.35;
        }
        [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {
            background: #172234 !important;
            color: #F8FAFC !important;
            border-color: rgba(255, 255, 255, 0.06);
        }
        [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {
            background: rgba(16, 185, 129, 0.14) !important;
            border-color: rgba(16, 185, 129, 0.35) !important;
            color: #34D399 !important;
            font-weight: 600;
        }
        [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {
            color: #34D399 !important;
            font-weight: 600;
        }
        [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"]::before {
            position: absolute;
            top: 0.5rem;
            bottom: 0.5rem;
            left: 0;
            width: 3px;
            border-radius: 0 3px 3px 0;
            background: #10B981;
            content: "";
        }

        /* Sidebar Dataset Status Card */
        .dataset-status-card {
            background: #141E30;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.85rem;
            display: flex;
            align-items: center;
            gap: 0.65rem;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .status-dot.demo {
            background: #38BDF8;
            box-shadow: 0 0 8px rgba(56, 189, 248, 0.6);
        }
        .status-dot.user {
            background: #10B981;
            box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
        }
        .dataset-title {
            font-size: 0.82rem;
            font-weight: 600;
            color: #F1F5F9;
            line-height: 1.2;
            word-break: break-all;
        }
        .dataset-sub {
            font-size: 0.72rem;
            color: #94A3B8;
            margin-top: 0.15rem;
        }

        /* Sidebar Footer */
        .sidebar-footer-card {
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            color: #64748B;
            font-size: 0.78rem;
            line-height: 1.45;
        }
        .footer-brand {
            font-weight: 600;
            color: #94A3B8;
        }

        /* METRIC CARDS (Clean SaaS KPI Cards) */
        .kpi-card {
            background: #131B2A;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
            transition: transform 0.15s ease, border-color 0.15s ease;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 108px;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.16);
        }
        .kpi-header {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            color: #94A3B8;
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .kpi-value {
            font-size: 1.45rem;
            font-weight: 700;
            color: #F8FAFC;
            letter-spacing: -0.02em;
            margin: 0.35rem 0 0.15rem 0;
            line-height: 1.2;
        }
        .kpi-value.balance {
            color: #38BDF8;
        }
        .kpi-value.income {
            color: #10B981;
        }
        .kpi-value.expense {
            color: #F43F5E;
        }
        .kpi-value.positive {
            color: #10B981;
        }
        .kpi-value.negative {
            color: #F43F5E;
        }
        .kpi-sub {
            font-size: 0.72rem;
            color: #64748B;
        }

        /* SUMMARY CALLOUT BOX */
        .summary-card {
            background: #131B2A;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-left: 4px solid #10B981;
            border-radius: 10px;
            padding: 1.2rem 1.4rem;
            margin: 1.25rem 0 1.75rem 0;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.18);
        }
        .summary-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.05rem;
            font-weight: 700;
            color: #F8FAFC;
            margin-bottom: 0.75rem;
        }
        .summary-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 0.55rem 1.5rem;
        }
        .summary-item {
            color: #CBD5E1;
            font-size: 0.92rem;
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
            line-height: 1.45;
        }
        .summary-bullet {
            color: #10B981;
            font-weight: bold;
            font-size: 1rem;
        }

        /* SAAS DEMO BANNER */
        .saas-banner {
            background: rgba(56, 189, 248, 0.06);
            border: 1px solid rgba(56, 189, 248, 0.22);
            border-radius: 10px;
            padding: 0.85rem 1.15rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.85rem;
        }
        .saas-banner-left {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            flex-wrap: wrap;
        }
        .saas-banner-pill {
            background: rgba(56, 189, 248, 0.18);
            color: #38BDF8;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            padding: 0.2rem 0.55rem;
            border-radius: 4px;
            text-transform: uppercase;
        }
        .saas-banner-text {
            color: #E2E8F0;
            font-size: 0.9rem;
        }

        /* CHART CONTAINER CARD */
        .chart-card {
            background: #131B2A;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
        }

        /* STANDOUT MONTHLY CARDS */
        .standout-card {
            background: #131B2A;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 1.1rem 1.25rem;
            text-align: center;
        }
        .standout-card.best {
            border-top: 3px solid #10B981;
        }
        .standout-card.spending {
            border-top: 3px solid #F43F5E;
        }
        .standout-card.income {
            border-top: 3px solid #38BDF8;
        }
        .standout-label {
            font-size: 0.75rem;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }
        .standout-month {
            font-size: 1.25rem;
            font-weight: 700;
            color: #F8FAFC;
        }
        .standout-val {
            font-size: 0.95rem;
            font-weight: 600;
            margin-top: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Metric Cards Renderer
# -----------------------------------------------------------------------------
def render_metric_cards(metrics: Dict[str, Any]) -> None:
    """Render the 6 responsive metric cards for the financial overview."""
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-header">🏦 Current Balance</div>
                <div class="kpi-value balance">{format_inr(metrics["current_balance"])}</div>
                <div class="kpi-sub">Closing position</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-header">💰 Total Income</div>
                <div class="kpi-value income">{format_inr(metrics["total_income"])}</div>
                <div class="kpi-sub">Cumulative credits</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-header">💸 Total Expenses</div>
                <div class="kpi-value expense">{format_inr(metrics["total_expenses"])}</div>
                <div class="kpi-sub">Cumulative debits</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        net_cf = metrics["net_cash_flow"]
        cf_class = "positive" if net_cf >= 0 else "negative"
        sub_text = "Net surplus" if net_cf >= 0 else "Net deficit"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-header">📈 Net Cash Flow</div>
                <div class="kpi-value {cf_class}">{format_inr(net_cf)}</div>
                <div class="kpi-sub">{sub_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-header">🧾 Transactions</div>
                <div class="kpi-value">{metrics["transaction_count"]}</div>
                <div class="kpi-sub">Total records</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col6:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-header">📉 Average Expense</div>
                <div class="kpi-value">{format_inr(metrics["average_expense"])}</div>
                <div class="kpi-sub">Per transaction</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Sidebar Navigation and Filters
# -----------------------------------------------------------------------------
def render_sidebar(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Render sidebar with completely separate brand card, navigation menu, and filters."""
    # 1. SEPARATE BRAND LOGO SECTION (Distinct container with breathing room)
    logo_b64 = get_logo_base64()
    if logo_b64:
        st.sidebar.markdown(
            f"""
            <div class="fintrack-brand-container">
                <div class="fintrack-brand-card">
                    <img src="data:image/png;base64,{logo_b64}" alt="FinTrack - Personal Finance, Simplified" class="fintrack-brand-logo-img" />
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            """
            <div class="fintrack-brand-container">
                <h2 style="margin: 0; color: #10B981; font-weight: 800; letter-spacing: -0.03em;">FinTrack</h2>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.2rem;">Personal Finance, Simplified</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. SEPARATE NAVIGATION SECTION
    st.sidebar.markdown('<div class="sidebar-section-title">NAVIGATION</div>', unsafe_allow_html=True)
    pages = ["🏠 Dashboard", "💳 Transactions", "📊 Analytics", "📁 Upload Data", "ℹ️ About"]
    selected_page = st.sidebar.radio(
        "Navigation",
        pages,
        index=pages.index(st.session_state.current_page),
        label_visibility="collapsed",
    )
    st.session_state.current_page = selected_page

    # 3. DATASET STATUS CARD
    st.sidebar.markdown('<div class="sidebar-section-title">DATA SOURCE</div>', unsafe_allow_html=True)
    if st.session_state.data_source == "demo":
        st.sidebar.markdown(
            f"""
            <div class="dataset-status-card">
                <span class="status-dot demo"></span>
                <div>
                    <div class="dataset-title">Synthetic Demo Database</div>
                    <div class="dataset-sub">{len(df)} verified transactions</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            f"""
            <div class="dataset-status-card">
                <span class="status-dot user"></span>
                <div>
                    <div class="dataset-title">{st.session_state.data_source_name}</div>
                    <div class="dataset-sub">Active session dataset ({len(df)} records)</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.sidebar.button("🔄 Reset to Demo Data", use_container_width=True):
            st.session_state.data_source = "demo"
            st.session_state.data_source_name = "demo_transactions.csv"
            st.session_state.uploaded_df = None
            st.rerun()

    # 4. FILTERS SECTION (Dashboard, Transactions, Analytics)
    filter_values = {}
    if selected_page in ["🏠 Dashboard", "💳 Transactions", "📊 Analytics"]:
        st.sidebar.markdown('<div class="sidebar-section-title">FILTERS</div>', unsafe_allow_html=True)

        if df.empty:
            st.sidebar.info("No records available to filter.")
            filtered_df = df
        else:
            # Date Range Filter
            dates = pd.to_datetime(df["Date"]).dropna()
            min_date = dates.min().date() if not dates.empty else datetime.today().date()
            max_date = dates.max().date() if not dates.empty else datetime.today().date()

            date_range = st.sidebar.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )

            start_date = None
            end_date = None
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
            elif isinstance(date_range, tuple) and len(date_range) == 1:
                start_date = date_range[0]
                end_date = max_date

            # Transaction Type Filter
            type_options = ["All", "Income", "Expense"]
            type_filter = st.sidebar.selectbox("Transaction Type", type_options, index=0)

            # Category Filter
            available_categories = ["All"] + sorted(df["Category"].dropna().unique().tolist())
            category_filter = st.sidebar.selectbox("Category", available_categories, index=0)

            # Search Description Filter
            search_query = st.sidebar.text_input("Search Description", placeholder="e.g. Zomato, Salary...")

            # Amount Range Slider
            max_row_amt = float(np.maximum(df["Income"], df["Expense"]).max()) if not df.empty else 100000.0
            amount_range = st.sidebar.slider(
                "Amount Range (₹)",
                min_value=0.0,
                max_value=float(np.ceil(max_row_amt / 1000.0) * 1000.0 if max_row_amt > 0 else 100000.0),
                value=(0.0, float(np.ceil(max_row_amt / 1000.0) * 1000.0 if max_row_amt > 0 else 100000.0)),
                step=100.0,
            )

            min_amt, max_amt = amount_range

            filter_values = {
                "start_date": start_date,
                "end_date": end_date,
                "type_filter": type_filter,
                "category": category_filter,
                "search": search_query,
                "min_amount": min_amt,
                "max_amount": max_amt,
            }

            filtered_df = apply_filters(
                df,
                start_date=start_date,
                end_date=end_date,
                type_filter=type_filter,
                category=category_filter,
                search=search_query,
                min_amount=min_amt,
                max_amount=max_amt,
            )
    else:
        filtered_df = df

    # 5. SIDEBAR FOOTER
    st.sidebar.markdown(
        """
        <div class="sidebar-footer-card">
            <div class="footer-brand">FinTrack Personal Finance</div>
            <div>Portfolio-Ready SaaS Dashboard</div>
            <div style="margin-top: 0.35rem; color: #475569;">Synthetic demo data • Zero tracking</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return filtered_df, filter_values


# -----------------------------------------------------------------------------
# PAGE 1: Dashboard
# -----------------------------------------------------------------------------
def render_dashboard(df: pd.DataFrame) -> None:
    """Render the clean SaaS financial dashboard."""
    # Page Header
    st.markdown(
        """
        <div class="saas-page-header">
            <h1 class="saas-page-title">Financial Overview</h1>
            <p class="saas-page-desc">A consolidated snapshot of your balance trends, cash flows, and category spending.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Demo Banner with Action Button
    if st.session_state.data_source == "demo":
        banner_col_l, banner_col_r = st.columns([4, 1.2])
        with banner_col_l:
            st.markdown(
                """
                <div class="saas-banner">
                    <div class="saas-banner-left">
                        <span class="saas-banner-pill">DEMO DATA</span>
                        <span class="saas-banner-text">Viewing 3-month synthetic transactions for demonstration.</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with banner_col_r:
            if st.button("📁 Upload Your Own Data", use_container_width=True):
                st.session_state.current_page = "📁 Upload Data"
                st.rerun()

    if df.empty:
        st.warning("⚠️ No transactions match your current filters. Adjust your sidebar filters to see data.")
        return

    # 1. 6 KPI Metric Cards
    metrics = financial_overview(df)
    render_metric_cards(metrics)

    # 2. "Your Financial Summary" Card
    insights = generate_insights(df)
    summary_html = """
    <div class="summary-card">
        <div class="summary-header">
            <span>💡</span>
            <span>Your Financial Summary</span>
        </div>
        <div class="summary-list">
    """
    for ins in insights:
        summary_html += f'<div class="summary-item"><span class="summary-bullet">•</span><span>{ins}</span></div>'
    summary_html += """
        </div>
    </div>
    """
    st.markdown(summary_html, unsafe_allow_html=True)

    # 3. Interactive Charts
    st.markdown("<h3 style='color: #F8FAFC; margin: 1.5rem 0 1rem 0;'>Financial Trends & Breakdowns</h3>", unsafe_allow_html=True)

    # Row 1: Balance Trend & Monthly Income vs Expenses
    r1_col1, r1_col2 = st.columns(2)

    with r1_col1:
        # Chart 1: Balance Trend
        b_df = balance_trend(df)
        if not b_df.empty:
            fig_bal = px.line(
                b_df,
                x="Date",
                y="Balance",
                title="Balance Trend",
                color_discrete_sequence=[COLOR_PALETTE["balance"]],
            )
            fig_bal.update_traces(
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(width=2.5),
                hovertemplate="<b>Date:</b> %{x|%d %b %Y}<br><b>Balance:</b> ₹%{y:,.2f}<extra></extra>",
            )
            fig_bal.update_xaxes(title_text="Date")
            fig_bal.update_yaxes(title_text="Balance (₹)")
            apply_chart_style(fig_bal, "Balance Trend")
            st.plotly_chart(fig_bal, use_container_width=True)
        else:
            st.info("No balance history available for this selection.")

    with r1_col2:
        # Chart 2: Monthly Income vs Expenses
        m_df = monthly_summary(df)
        if not m_df.empty:
            fig_inc_exp = px.bar(
                m_df,
                x="Month",
                y=["Income", "Expenses"],
                barmode="group",
                title="Monthly Income vs Expenses",
                color_discrete_map={
                    "Income": COLOR_PALETTE["income"],
                    "Expenses": COLOR_PALETTE["expense"],
                },
            )
            fig_inc_exp.update_traces(
                hovertemplate="<b>%{x}</b><br>%{data.name}: ₹%{y:,.2f}<extra></extra>"
            )
            fig_inc_exp.update_xaxes(title_text="Month")
            fig_inc_exp.update_yaxes(title_text="Amount (₹)")
            apply_chart_style(fig_inc_exp, "Monthly Income vs Expenses")
            st.plotly_chart(fig_inc_exp, use_container_width=True)
        else:
            st.info("No monthly data available for the selected range.")

    # Row 2: Expense by Category (Donut) & Monthly Expenses
    r2_col1, r2_col2 = st.columns(2)

    with r2_col1:
        # Chart 3: Expense by Category
        cat_df = expense_by_category(df)
        if not cat_df.empty:
            fig_cat = px.pie(
                cat_df,
                names="Category",
                values="Amount",
                title="Expense by Category",
                hole=0.45,
                color_discrete_sequence=COLOR_PALETTE["category_colors"],
            )
            fig_cat.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Amount: ₹%{value:,.2f}<br>Share: %{percent}<extra></extra>",
            )
            apply_chart_style(fig_cat, "Expense by Category")
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("No expense transactions recorded in this selection.")

    with r2_col2:
        # Chart 4: Monthly Expenses
        m_exp = monthly_expenses(df)
        if not m_exp.empty:
            fig_monthly_exp = px.bar(
                m_exp,
                x="Month",
                y="Expenses",
                title="Monthly Expenses",
                color_discrete_sequence=[COLOR_PALETTE["expense"]],
            )
            fig_monthly_exp.update_traces(
                hovertemplate="<b>%{x}</b><br>Total Expense: ₹%{y:,.2f}<extra></extra>"
            )
            fig_monthly_exp.update_xaxes(title_text="Month")
            fig_monthly_exp.update_yaxes(title_text="Expenses (₹)")
            apply_chart_style(fig_monthly_exp, "Monthly Expenses")
            st.plotly_chart(fig_monthly_exp, use_container_width=True)
        else:
            st.info("No expense data available.")

    # Row 3: Spending Trend & Top Expenses
    r3_col1, r3_col2 = st.columns(2)

    with r3_col1:
        # Chart 5: Spending Trend
        spend_df = spending_trend(df)
        if not spend_df.empty:
            fig_spend = px.line(
                spend_df,
                x="Date",
                y="Expense",
                title="Spending Trend",
                color_discrete_sequence=[COLOR_PALETTE["expense"]],
            )
            fig_spend.update_traces(
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(width=2),
                hovertemplate="<b>Date:</b> %{x|%d %b %Y}<br><b>Spent:</b> ₹%{y:,.2f}<extra></extra>",
            )
            fig_spend.update_xaxes(title_text="Date")
            fig_spend.update_yaxes(title_text="Daily Spending (₹)")
            apply_chart_style(fig_spend, "Spending Trend")
            st.plotly_chart(fig_spend, use_container_width=True)
        else:
            st.info("No daily spending data available.")

    with r3_col2:
        # Chart 6: Top Expenses (Horizontal Bar)
        top_exp_df = top_expenses(df, top_n=10)
        if not top_exp_df.empty:
            top_exp_sorted = top_exp_df.sort_values("Expense", ascending=True)
            fig_top = px.bar(
                top_exp_sorted,
                x="Expense",
                y="Description",
                orientation="h",
                title="Top Expenses (Largest 10)",
                color="Category",
                color_discrete_sequence=COLOR_PALETTE["category_colors"],
            )
            fig_top.update_traces(
                hovertemplate="<b>%{y}</b><br>Expense: ₹%{x:,.2f}<extra></extra>"
            )
            fig_top.update_xaxes(title_text="Amount (₹)")
            fig_top.update_yaxes(title_text="Description")
            apply_chart_style(fig_top, "Top Expenses")
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info("No expenses found.")

    # Row 4: Transaction Distribution
    dist_df = transaction_distribution(df)
    if not dist_df.empty:
        fig_dist = px.bar(
            dist_df,
            x="Type",
            y="Count",
            title="Transaction Distribution",
            color="Type",
            color_discrete_map={
                "Income": COLOR_PALETTE["income"],
                "Expense": COLOR_PALETTE["expense"],
            },
            text="Count",
        )
        fig_dist.update_traces(
            textposition="outside",
            hovertemplate="<b>%{x}:</b> %{y} transactions<extra></extra>",
        )
        fig_dist.update_xaxes(title_text="Transaction Type")
        fig_dist.update_yaxes(title_text="Number of Transactions")
        apply_chart_style(fig_dist, "Transaction Distribution", height=320)
        st.plotly_chart(fig_dist, use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE 2: Transactions
# -----------------------------------------------------------------------------
def render_transactions(df: pd.DataFrame) -> None:
    """Render the detailed transactions page with formatted tables and CSV export."""
    st.markdown(
        """
        <div class="saas-page-header">
            <h1 class="saas-page-title">Transactions</h1>
            <p class="saas-page-desc">Search, inspect, and export your individual financial records.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df.empty:
        st.warning("⚠️ No transactions match your current filters. Try resetting the filters in the sidebar.")
        return

    # Transaction Summary Pill
    total_inc = df["Income"].sum()
    total_exp = df["Expense"].sum()
    total_count = len(df)

    sum_col1, sum_col2, sum_col3 = st.columns(3)
    sum_col1.metric("Filtered Records", f"{total_count} transactions")
    sum_col2.metric("Filtered Income", format_inr(total_inc))
    sum_col3.metric("Filtered Expenses", format_inr(total_exp))

    st.markdown("<br>", unsafe_allow_html=True)

    # Prepare Display DataFrame with Indian Rupee formatting
    display_df = df.copy()
    display_df["Date"] = pd.to_datetime(display_df["Date"]).dt.strftime("%d %b %Y")
    display_df["Income"] = display_df["Income"].apply(lambda v: format_inr(v) if v > 0 else "-")
    display_df["Expense"] = display_df["Expense"].apply(lambda v: format_inr(v) if v > 0 else "-")
    display_df["Balance"] = display_df["Balance"].apply(format_inr)

    # Interactive Table
    st.dataframe(
        display_df[["Date", "Description", "Category", "Type", "Income", "Expense", "Balance"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.TextColumn("Date", width="medium"),
            "Description": st.column_config.TextColumn("Description", width="large"),
            "Category": st.column_config.TextColumn("Category", width="medium"),
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Income": st.column_config.TextColumn("Income", width="medium"),
            "Expense": st.column_config.TextColumn("Expense", width="medium"),
            "Balance": st.column_config.TextColumn("Balance", width="medium"),
        },
    )

    # Download Button
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Transactions",
        data=csv_bytes,
        file_name="filtered_transactions.csv",
        mime="text/csv",
        use_container_width=False,
    )


# -----------------------------------------------------------------------------
# PAGE 3: Analytics
# -----------------------------------------------------------------------------
def render_analytics(df: pd.DataFrame) -> None:
    """Render the deep financial analytics page."""
    st.markdown(
        """
        <div class="saas-page-header">
            <h1 class="saas-page-title">Analytics</h1>
            <p class="saas-page-desc">In-depth breakdown of income streams, expense behaviors, and monthly performance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df.empty:
        st.warning("⚠️ No transactions match your current filters. Adjust your sidebar filters to see analytics.")
        return

    # Section 1: Cash Flow
    st.markdown("<h3 style='color: #F8FAFC; margin-bottom: 0.75rem;'>Cash Flow Health</h3>", unsafe_allow_html=True)
    cf = cash_flow_metrics(df)

    cf_col1, cf_col2, cf_col3, cf_col4 = st.columns(4)
    cf_col1.metric("Total Income", format_inr(cf["total_income"]))
    cf_col2.metric("Total Expenses", format_inr(cf["total_expenses"]))
    cf_col3.metric(
        "Net Cash Flow",
        format_inr(cf["net_cash_flow"]),
        delta=f"{format_inr(cf['net_cash_flow'])}",
        delta_color="normal" if cf["net_cash_flow"] >= 0 else "inverse",
    )
    if cf["cash_flow_rate"] is not None:
        cf_col4.metric(
            "Cash Flow / Savings Rate",
            f"{cf['cash_flow_rate']:.1f}%",
            help="Net Cash Flow divided by Total Income × 100",
        )
    else:
        cf_col4.metric("Cash Flow Rate", "N/A (No Income)")

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 2: Income Analysis vs Expense Analysis tabs
    tab_income, tab_expense = st.tabs(["💰 Income Analysis", "💸 Expense Analysis"])

    with tab_income:
        inc_data = income_analysis(df)
        inc_c1, inc_c2, inc_c3, inc_c4 = st.columns(4)
        inc_c1.metric("Total Income", format_inr(inc_data["total_income"]))
        inc_c2.metric("Income Transactions", f"{inc_data['count']}")
        inc_c3.metric("Average Income", format_inr(inc_data["average"]))
        inc_c4.metric("Largest Income", format_inr(inc_data["largest"]))

        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            if not inc_data["by_month"].empty:
                fig_inc_m = px.bar(
                    inc_data["by_month"],
                    x="Month",
                    y="Income",
                    title="Income by Month",
                    color_discrete_sequence=[COLOR_PALETTE["income"]],
                )
                fig_inc_m.update_traces(hovertemplate="<b>%{x}</b>: ₹%{y:,.2f}<extra></extra>")
                apply_chart_style(fig_inc_m, "Income by Month")
                st.plotly_chart(fig_inc_m, use_container_width=True)
            else:
                st.info("No monthly income records.")

        with ch_col2:
            if not inc_data["by_category"].empty:
                fig_inc_cat = px.pie(
                    inc_data["by_category"],
                    names="Category",
                    values="Income",
                    title="Income by Category",
                    color_discrete_sequence=COLOR_PALETTE["category_colors"],
                    hole=0.4,
                )
                fig_inc_cat.update_traces(hovertemplate="<b>%{label}</b>: ₹%{value:,.2f}<extra></extra>")
                apply_chart_style(fig_inc_cat, "Income by Category")
                st.plotly_chart(fig_inc_cat, use_container_width=True)
            else:
                st.info("No categorized income records.")

    with tab_expense:
        exp_data = expense_analysis(df)
        exp_c1, exp_c2, exp_c3, exp_c4 = st.columns(4)
        exp_c1.metric("Total Expenses", format_inr(exp_data["total_expenses"]))
        exp_c2.metric("Expense Transactions", f"{exp_data['count']}")
        exp_c3.metric("Average Expense", format_inr(exp_data["average"]))
        exp_c4.metric("Largest Expense", format_inr(exp_data["largest"]))

        ech_col1, ech_col2 = st.columns(2)
        with ech_col1:
            if not exp_data["by_month"].empty:
                fig_exp_m = px.bar(
                    exp_data["by_month"],
                    x="Month",
                    y="Expenses",
                    title="Expense by Month",
                    color_discrete_sequence=[COLOR_PALETTE["expense"]],
                )
                fig_exp_m.update_traces(hovertemplate="<b>%{x}</b>: ₹%{y:,.2f}<extra></extra>")
                apply_chart_style(fig_exp_m, "Expense by Month")
                st.plotly_chart(fig_exp_m, use_container_width=True)
            else:
                st.info("No monthly expense records.")

        with ech_col2:
            if not exp_data["by_category"].empty:
                fig_exp_cat = px.bar(
                    exp_data["by_category"],
                    x="Category",
                    y="Amount",
                    title="Expense by Category",
                    color="Category",
                    color_discrete_sequence=COLOR_PALETTE["category_colors"],
                )
                fig_exp_cat.update_traces(hovertemplate="<b>%{x}</b>: ₹%{y:,.2f}<extra></extra>")
                apply_chart_style(fig_exp_cat, "Expense by Category")
                st.plotly_chart(fig_exp_cat, use_container_width=True)
            else:
                st.info("No categorized expense records.")

        if not exp_data["top_10"].empty:
            st.markdown("<h4 style='color: #F8FAFC; margin-top: 1rem;'>Top 10 Expenses</h4>", unsafe_allow_html=True)
            top_sorted = exp_data["top_10"].sort_values("Expense", ascending=True)
            fig_top10 = px.bar(
                top_sorted,
                x="Expense",
                y="Description",
                orientation="h",
                color="Category",
                color_discrete_sequence=COLOR_PALETTE["category_colors"],
            )
            fig_top10.update_traces(hovertemplate="<b>%{y}</b>: ₹%{x:,.2f}<extra></extra>")
            apply_chart_style(fig_top10, "Top 10 Expenses", height=380)
            st.plotly_chart(fig_top10, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 3: Monthly Comparison
    st.markdown("<h3 style='color: #F8FAFC; margin-bottom: 0.75rem;'>Monthly Comparison</h3>", unsafe_allow_html=True)
    comp = monthly_comparison(df)

    if comp["table"] is not None and not comp["table"].empty:
        h1, h2, h3 = st.columns(3)

        if comp["best_cash_flow_month"]:
            b_m = comp["best_cash_flow_month"]
            with h1:
                st.markdown(
                    f"""
                    <div class="standout-card best">
                        <div class="standout-label">🌟 Best Cash-Flow Month</div>
                        <div class="standout-month">{b_m['Month']}</div>
                        <div class="standout-val" style="color: #10B981;">{format_inr(b_m['Net Cash Flow'])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if comp["highest_spending_month"]:
            s_m = comp["highest_spending_month"]
            with h2:
                st.markdown(
                    f"""
                    <div class="standout-card spending">
                        <div class="standout-label">💸 Highest Spending Month</div>
                        <div class="standout-month">{s_m['Month']}</div>
                        <div class="standout-val" style="color: #F43F5E;">{format_inr(s_m['Expenses'])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if comp["highest_income_month"]:
            i_m = comp["highest_income_month"]
            with h3:
                st.markdown(
                    f"""
                    <div class="standout-card income">
                        <div class="standout-label">💰 Highest Income Month</div>
                        <div class="standout-month">{i_m['Month']}</div>
                        <div class="standout-val" style="color: #38BDF8;">{format_inr(i_m['Income'])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        fig_comp = px.bar(
            comp["table"],
            x="Month",
            y=["Income", "Expenses", "Net Cash Flow"],
            barmode="group",
            title="Monthly Comparison: Income, Expenses & Net Cash Flow",
            color_discrete_map={
                "Income": COLOR_PALETTE["income"],
                "Expenses": COLOR_PALETTE["expense"],
                "Net Cash Flow": COLOR_PALETTE["balance"],
            },
        )
        fig_comp.update_traces(hovertemplate="<b>%{x}</b><br>%{data.name}: ₹%{y:,.2f}<extra></extra>")
        apply_chart_style(fig_comp, "Monthly Comparison", height=380)
        st.plotly_chart(fig_comp, use_container_width=True)

        table_view = comp["table"].copy()
        table_view["Income"] = table_view["Income"].apply(format_inr)
        table_view["Expenses"] = table_view["Expenses"].apply(format_inr)
        table_view["Net Cash Flow"] = table_view["Net Cash Flow"].apply(format_inr)
        st.dataframe(
            table_view[["Month", "Income", "Expenses", "Net Cash Flow", "Transactions"]],
            use_container_width=True,
            hide_index=True,
        )


# -----------------------------------------------------------------------------
# PAGE 4: Upload Data
# -----------------------------------------------------------------------------
def render_upload_data() -> None:
    """Render the upload personal dataset page with privacy protections."""
    st.markdown(
        """
        <div class="saas-page-header">
            <h1 class="saas-page-title">Upload Data</h1>
            <p class="saas-page-desc">Analyze your own bank statement or spreadsheet securely in-session.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Privacy Guarantee Banners
    st.info(
        "🔒 **Privacy Guarantee:** Your personal financial data is processed in-memory for this session only "
        "and is never written to disk or included in the public demo database."
    )
    st.warning(
        "⚠️ **Security Notice:** Do not upload sensitive banking credentials. "
        "Never share passwords, PINs, OTPs, CVVs, or internet banking logins."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload your transaction register (CSV, XLSX, or XLS)",
        type=["csv", "xlsx", "xls"],
        help="Upload standard bank export files or spreadsheets.",
    )

    if uploaded_file is not None:
        try:
            with st.spinner("Processing and normalizing transaction columns..."):
                processed_df = load_uploaded_data(uploaded_file)

            st.session_state.uploaded_df = processed_df
            st.session_state.data_source = "uploaded"
            st.session_state.data_source_name = uploaded_file.name

            st.success(
                f"✅ Successfully loaded **{uploaded_file.name}** with **{len(processed_df)} transactions**! "
                "The dashboard and analytics now reflect your data."
            )

            st.markdown("### Data Preview (First 5 Transactions)")
            preview_df = processed_df.head(5).copy()
            preview_df["Date"] = pd.to_datetime(preview_df["Date"]).dt.strftime("%d %b %Y")
            preview_df["Income"] = preview_df["Income"].apply(lambda v: format_inr(v) if v > 0 else "-")
            preview_df["Expense"] = preview_df["Expense"].apply(lambda v: format_inr(v) if v > 0 else "-")
            preview_df["Balance"] = preview_df["Balance"].apply(format_inr)
            st.dataframe(preview_df, use_container_width=True, hide_index=True)

            col_btn1, col_btn2 = st.columns([1, 4])
            with col_btn1:
                if st.button("🚀 View Dashboard", type="primary"):
                    st.session_state.current_page = "🏠 Dashboard"
                    st.rerun()

        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.markdown(
                """
                **Tips for resolving format issues:**
                - Verify your file contains a date column (e.g., `Date`, `Txn Date`, `Transaction Date`).
                - Ensure either separate `Income` & `Expense` columns or an `Amount` column is present.
                - Confirm the file is not password-protected.
                """
            )
        except Exception as e:
            st.error(f"❌ Unexpected error reading file: {str(e)}")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("ℹ️ Supported Column Formats & Intelligent Recognition"):
        st.markdown(
            """
            The tracker automatically recognizes common column naming variants:
            * **Date:** `Date`, `Transaction Date`, `Txn Date`, `Value Date`, `Posting Date`
            * **Description:** `Description`, `Narration`, `Particulars`, `Details`, `Memo`, `Remarks`
            * **Income:** `Income`, `Deposit`, `Credit`, `Credit Amount`, `Cr`, `Inflow`
            * **Expense:** `Expense`, `Withdrawal`, `Debit`, `Debit Amount`, `Dr`, `Outflow`
            * **Balance:** `Balance`, `Closing Balance`, `Running Balance`, `Available Balance`
            * **Single Amount:** An `Amount` column with a `Type` or `Dr/Cr` indicator is also supported.
            * **Category:** If omitted, transactions are automatically categorized based on description keywords.
            """
        )

    if st.session_state.data_source == "uploaded":
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Switch Back to Demo Data"):
            st.session_state.data_source = "demo"
            st.session_state.data_source_name = "demo_transactions.csv"
            st.session_state.uploaded_df = None
            st.session_state.current_page = "🏠 Dashboard"
            st.rerun()


# -----------------------------------------------------------------------------
# PAGE 5: About
# -----------------------------------------------------------------------------
def render_about() -> None:
    """Render the project about page, feature checklist, and privacy declarations."""
    logo_b64 = get_logo_base64()
    if logo_b64:
        st.markdown(
            f"""
            <div style="background: #FFFFFF; border-radius: 12px; padding: 14px 20px; display: inline-block; margin-bottom: 1.5rem; box-shadow: 0 4px 14px rgba(0,0,0,0.25);">
                <img src="data:image/png;base64,{logo_b64}" alt="FinTrack" style="max-width: 260px; height: auto; display: block;" />
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="saas-page-header">
            <h1 class="saas-page-title">About FinTrack</h1>
            <p class="saas-page-desc">Personal Finance, Simplified — A Portfolio-Ready SaaS Application.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### ✨ Core Features
            * 💰 **Income & Expense Tracking:** Real-time visibility into balances and cash flows.
            * 🍩 **Category Classification:** Donut and bar breakdowns across lifestyle categories.
            * 📅 **Monthly Comparative Analytics:** Head-to-head cash flow evaluation and peak detection.
            * 📈 **Interactive Plotly Visualizations:** Responsive line, bar, donut, and distribution charts.
            * 🧠 **Smart English Insights:** Descriptive observations derived directly from transaction history.
            * 🔍 **Multi-dimensional Filtering:** Filter dynamically by Date Range, Type, Category, Description, and Amount.
            * 📥 **CSV Export:** Download filtered datasets with a single click.
            * 📁 **Custom File Upload:** Support for CSV and Excel statements with auto column detection.
            """
        )

    with col2:
        st.markdown(
            """
            ### 🛠️ Technology Stack
            * **Python 3.10+**
            * **Streamlit:** Modern reactive web framework
            * **Pandas:** Tabular data manipulation & aggregation
            * **Plotly:** Interactive, responsive charting engine
            * **OpenPyXL:** Excel file processing (.xlsx/.xls)
            * **NumPy:** Numerical computing utilities

            ### 🔒 Privacy & Safety Declarations
            * **Synthetic Demo Data:** The default database uses synthetic transactions with no real banking records.
            * **Zero Server Storage:** Uploaded statements exist strictly in browser session memory and are never persisted to disk.
            * **No Credentials Required:** Never upload or share passwords, OTPs, or banking PINs.
            """
        )

    st.markdown("<br><hr style='border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 1rem 0;">
            FinTrack • Personal Finance, Simplified • Designed for GitHub & Streamlit Community Cloud
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Main Application Entrypoint
# -----------------------------------------------------------------------------
def main() -> None:
    # 1. Initialize State & Inject CSS
    init_session_state()
    inject_custom_css()

    # 2. Retrieve Active Dataframe
    active_df = get_active_dataframe()

    # 3. Render Sidebar Controls & Filter Data
    filtered_df, _ = render_sidebar(active_df)

    # 4. Route to Active Page
    current_page = st.session_state.current_page

    if current_page == "🏠 Dashboard":
        render_dashboard(filtered_df)
    elif current_page == "💳 Transactions":
        render_transactions(filtered_df)
    elif current_page == "📊 Analytics":
        render_analytics(filtered_df)
    elif current_page == "📁 Upload Data":
        render_upload_data()
    elif current_page == "ℹ️ About":
        render_about()


if __name__ == "__main__":
    main()
