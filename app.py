"""
Personal Financial Tracker
A modern, portfolio-quality personal financial dashboard built with Streamlit, Pandas, and Plotly.
"""
from __future__ import annotations

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
# App Configuration & Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Personal Financial Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Color Palette & Chart Theming
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
    """Apply consistent, professional styling to Plotly charts."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=16, color="#F1F5F9", family="sans-serif"),
            x=0.01,
            y=0.96,
        ),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color="#CBD5E1", size=12, family="sans-serif"),
        height=height,
        margin=dict(l=20, r=20, t=50 if title else 20, b=20),
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
            bordercolor="#475569"
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(255, 255, 255, 0.07)",
        zerolinecolor="rgba(255, 255, 255, 0.12)",
        showline=True,
        linecolor="rgba(255, 255, 255, 0.15)",
        tickfont=dict(color="#94A3B8"),
    )
    fig.update_yaxes(
        gridcolor="rgba(255, 255, 255, 0.07)",
        zerolinecolor="rgba(255, 255, 255, 0.12)",
        showline=True,
        linecolor="rgba(255, 255, 255, 0.15)",
        tickfont=dict(color="#94A3B8"),
    )
    return fig


# -----------------------------------------------------------------------------
# Data Caching & Session State Management
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_demo_data() -> pd.DataFrame:
    """Load and cache the synthetic demo transactions dataset."""
    return load_demo_data(DEFAULT_DEMO_PATH)


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
# Custom Styling (CSS)
# -----------------------------------------------------------------------------
def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        /* Modern fintech layout and background styling */
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3.5rem;
            max-width: 95rem;
        }

        /* Metric cards styling */
        .metric-card-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(16, 185, 129, 0.35);
        }
        .metric-label {
            font-size: 0.82rem;
            color: #94A3B8;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.45rem;
            margin-bottom: 0.4rem;
        }
        .metric-value {
            font-size: 1.45rem;
            font-weight: 700;
            color: #F8FAFC;
            letter-spacing: -0.02em;
        }
        .metric-value.income {
            color: #10B981;
        }
        .metric-value.expense {
            color: #F43F5E;
        }
        .metric-value.balance {
            color: #38BDF8;
        }
        .metric-value.positive {
            color: #10B981;
        }
        .metric-value.negative {
            color: #F43F5E;
        }

        /* Insights Card */
        .insight-box {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-left: 4px solid #10B981;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-bottom: 1.5rem;
        }
        .insight-item {
            color: #E2E8F0;
            font-size: 0.95rem;
            margin: 0.4rem 0;
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
        }
        .insight-bullet {
            color: #10B981;
            font-weight: bold;
        }

        /* Standout comparison cards */
        .highlight-card {
            background: rgba(30, 41, 59, 0.7);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 1rem 1.2rem;
            text-align: center;
        }
        .highlight-card.best {
            border-top: 3px solid #10B981;
        }
        .highlight-card.spending {
            border-top: 3px solid #F43F5E;
        }
        .highlight-card.income {
            border-top: 3px solid #38BDF8;
        }
        .highlight-title {
            font-size: 0.8rem;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.3rem;
        }
        .highlight-month {
            font-size: 1.15rem;
            font-weight: 700;
            color: #F8FAFC;
        }
        .highlight-amount {
            font-size: 0.92rem;
            font-weight: 600;
            margin-top: 0.2rem;
        }

        /* Banner styling */
        .demo-badge-bar {
            background: rgba(56, 189, 248, 0.08);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.75rem;
        }
        .demo-badge-text {
            color: #BAE6FD;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Chart container */
        .chart-box {
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        /* Sidebar footer styling */
        .sidebar-footer {
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            color: #64748B;
            font-size: 0.78rem;
            line-height: 1.4;
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
            <div class="metric-card">
                <div class="metric-label">🏦 Current Balance</div>
                <div class="metric-value balance">{format_inr(metrics["current_balance"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">💰 Total Income</div>
                <div class="metric-value income">{format_inr(metrics["total_income"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">💸 Total Expenses</div>
                <div class="metric-value expense">{format_inr(metrics["total_expenses"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        net_cf = metrics["net_cash_flow"]
        cf_class = "positive" if net_cf >= 0 else "negative"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">📈 Net Cash Flow</div>
                <div class="metric-value {cf_class}">{format_inr(net_cf)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">🧾 Transactions</div>
                <div class="metric-value">{metrics["transaction_count"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col6:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">📉 Average Expense</div>
                <div class="metric-value">{format_inr(metrics["average_expense"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Sidebar Navigation and Filters
# -----------------------------------------------------------------------------
def render_sidebar(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Render sidebar navigation, filter controls, data status, and footer."""
    st.sidebar.markdown("## 💰 FinTrack")
    st.sidebar.caption("Personal Financial Tracker")

    # Navigation Menu
    pages = ["🏠 Dashboard", "💳 Transactions", "📊 Analytics", "📁 Upload Data", "ℹ️ About"]
    selected_page = st.sidebar.radio(
        "Navigation",
        pages,
        index=pages.index(st.session_state.current_page),
        label_visibility="collapsed",
    )
    st.session_state.current_page = selected_page

    st.sidebar.markdown("---")

    # Data Source Indicator
    if st.session_state.data_source == "demo":
        st.sidebar.success("📊 Dataset: Synthetic Demo Data")
    else:
        st.sidebar.info(f"📁 Dataset: {st.session_state.data_source_name}")
        if st.sidebar.button("🔄 Reset to Demo Data", use_container_width=True):
            st.session_state.data_source = "demo"
            st.session_state.data_source_name = "demo_transactions.csv"
            st.session_state.uploaded_df = None
            st.rerun()

    # Filters Section (applicable on Dashboard, Transactions, Analytics)
    filter_values = {}
    if selected_page in ["🏠 Dashboard", "💳 Transactions", "📊 Analytics"]:
        st.sidebar.markdown("### 🔍 Filters")

        if df.empty:
            st.sidebar.info("No data available to filter.")
            filtered_df = df
        else:
            # 1. Date Range Filter
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

            # 2. Transaction Type Filter
            type_options = ["All", "Income", "Expense"]
            type_filter = st.sidebar.selectbox("Transaction Type", type_options, index=0)

            # 3. Category Filter
            available_categories = ["All"] + sorted(df["Category"].dropna().unique().tolist())
            category_filter = st.sidebar.selectbox("Category", available_categories, index=0)

            # 4. Search Filter
            search_query = st.sidebar.text_input("Search Description", placeholder="e.g. Zomato, Salary...")

            # 5. Amount Range Filter
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

    # Sidebar Footer
    st.sidebar.markdown(
        """
        <div class="sidebar-footer">
            <b>Personal Financial Tracker</b><br>
            Portfolio-Ready Project<br><br>
            <i>Demo data is synthetic and contains no real banking information.</i>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return filtered_df, filter_values


# -----------------------------------------------------------------------------
# PAGE 1: Dashboard
# -----------------------------------------------------------------------------
def render_dashboard(df: pd.DataFrame) -> None:
    """Render the primary dashboard page."""
    # Demo notice banner with Upload shortcut button
    if st.session_state.data_source == "demo":
        st.markdown(
            """
            <div class="demo-badge-bar">
                <div class="demo-badge-text">
                    <span>💡</span>
                    <span><b>Demo Mode:</b> You're viewing synthetic financial transactions for demonstration purposes.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_banner_l, col_banner_r = st.columns([4, 1])
        with col_banner_r:
            if st.button("📁 Upload Your Own Data", use_container_width=True):
                st.session_state.current_page = "📁 Upload Data"
                st.rerun()

    st.markdown("## Financial Overview")
    st.caption("A consolidated snapshot of your balances, cash flows, and expense distributions.")

    if df.empty:
        st.warning("⚠️ No transactions match your current filters. Adjust your sidebar filters to see data.")
        return

    # 1. Metric Cards
    metrics = financial_overview(df)
    render_metric_cards(metrics)

    # 2. Your Financial Summary (Smart English Insights)
    st.markdown("### Your Financial Summary")
    insights = generate_insights(df)
    with st.container():
        insight_html = '<div class="insight-box">'
        for insight in insights:
            insight_html += f'<div class="insight-item"><span class="insight-bullet">•</span> {insight}</div>'
        insight_html += "</div>"
        st.markdown(insight_html, unsafe_allow_html=True)

    # 3. Interactive Charts
    st.markdown("### Financial Trends & Visualizations")

    # Row 1: Balance Trend & Monthly Income vs Expenses
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
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

    with row1_col2:
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
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        # Chart 3: Expense by Category (Donut Chart)
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

    with row2_col2:
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
    row3_col1, row3_col2 = st.columns(2)

    with row3_col1:
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

    with row3_col2:
        # Chart 6: Top Expenses (Horizontal Bar)
        top_exp_df = top_expenses(df, top_n=10)
        if not top_exp_df.empty:
            # Sort ascending for clean horizontal bar presentation
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
            fig_top.update_yaxes(title_text="Transaction Description")
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
    st.markdown("## 💳 Transactions")
    st.caption("Search, inspect, and export your individual financial transactions.")

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

    st.markdown("---")

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
    st.markdown("## 📊 Analytics")
    st.caption("Detailed breakdown of income streams, expense behaviors, and monthly performance.")

    if df.empty:
        st.warning("⚠️ No transactions match your current filters. Adjust your sidebar filters to see analytics.")
        return

    # Section 1: Cash Flow
    st.markdown("### 📈 Cash Flow Analysis")
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

    st.markdown("---")

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

        # Top 10 Expenses in Expense Tab
        if not exp_data["top_10"].empty:
            st.markdown("#### Top 10 Expenses")
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

    st.markdown("---")

    # Section 3: Monthly Comparison
    st.markdown("### 📅 Monthly Comparison")
    comp = monthly_comparison(df)

    if comp["table"] is not None and not comp["table"].empty:
        # Highlight Cards for Standout Months
        h1, h2, h3 = st.columns(3)

        if comp["best_cash_flow_month"]:
            b_m = comp["best_cash_flow_month"]
            with h1:
                st.markdown(
                    f"""
                    <div class="highlight-card best">
                        <div class="highlight-title">🌟 Best Cash-Flow Month</div>
                        <div class="highlight-month">{b_m['Month']}</div>
                        <div class="highlight-amount" style="color: #10B981;">{format_inr(b_m['Net Cash Flow'])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if comp["highest_spending_month"]:
            s_m = comp["highest_spending_month"]
            with h2:
                st.markdown(
                    f"""
                    <div class="highlight-card spending">
                        <div class="highlight-title">💸 Highest Spending Month</div>
                        <div class="highlight-month">{s_m['Month']}</div>
                        <div class="highlight-amount" style="color: #F43F5E;">{format_inr(s_m['Expenses'])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if comp["highest_income_month"]:
            i_m = comp["highest_income_month"]
            with h3:
                st.markdown(
                    f"""
                    <div class="highlight-card income">
                        <div class="highlight-title">💰 Highest Income Month</div>
                        <div class="highlight-month">{i_m['Month']}</div>
                        <div class="highlight-amount" style="color: #38BDF8;">{format_inr(i_m['Income'])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Monthly Comparison Grouped Chart
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

        # Formatted Data Table for Monthly Comparison
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
    st.markdown("## 📁 Upload Data")
    st.caption("Analyze your own bank statement or transaction register securely within this session.")

    # Privacy Guarantee Banners
    st.info(
        "🔒 **Privacy Guarantee:** Your personal financial data is processed in-memory for this session only "
        "and is never written to disk or included in the public demo database."
    )
    st.warning(
        "⚠️ **Security Notice:** Do not upload sensitive banking credentials. "
        "Never share your bank passwords, PINs, OTPs, CVVs, or login credentials."
    )

    st.markdown("---")

    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload your own transaction data (CSV, XLSX, or XLS)",
        type=["csv", "xlsx", "xls"],
        help="Upload standard bank export files or spreadsheets.",
    )

    if uploaded_file is not None:
        try:
            with st.spinner("Processing and normalizing transaction columns..."):
                processed_df = load_uploaded_data(uploaded_file)

            # Store in session state
            st.session_state.uploaded_df = processed_df
            st.session_state.data_source = "uploaded"
            st.session_state.data_source_name = uploaded_file.name

            st.success(
                f"✅ Successfully loaded **{uploaded_file.name}** with **{len(processed_df)} transactions**! "
                "The dashboard and analytics now reflect your data."
            )

            # Preview Table
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

    st.markdown("---")

    # File Format Guidance
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

    # Reset Option
    if st.session_state.data_source == "uploaded":
        st.markdown("### Reset Data")
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
    st.markdown("## ℹ️ About Personal Financial Tracker")
    st.markdown(
        """
        **Personal Financial Tracker** is an interactive, privacy-first web application designed
        to help users understand their income patterns, expense behavior, spending categories,
        and cash-flow health with intuitive visualizations and automated insights.
        """
    )

    st.markdown("---")

    # Core Capabilities
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### ✨ Core Features
            * 💰 **Income & Expense Tracking:** Real-time visibility into overall balances and cash flows.
            * 🍩 **Category Classification:** Donut and bar breakdowns across standard lifestyle categories.
            * 📅 **Monthly Comparative Analytics:** Head-to-head monthly cash-flow evaluation and peak detection.
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
            * **Pandas:** High-performance data manipulation & aggregation
            * **Plotly:** Interactive, responsive charting engine
            * **OpenPyXL:** Excel file processing (.xlsx/.xls)
            * **NumPy:** Numerical computing utilities

            ### 🔒 Privacy & Safety Declarations
            * **Synthetic Demo Data:** The default database uses synthetic demo data and contains no real banking records.
            * **Zero Server Storage:** Uploaded statements exist strictly in your active browser session memory and are never persisted to disk or databases.
            * **No Credentials Required:** Never upload or share passwords, OTPs, or banking PINs.
            """
        )

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #94A3B8; font-size: 0.85rem; padding: 1rem 0;">
            Personal Financial Tracker • Designed for GitHub & Streamlit Community Cloud Deployment
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
