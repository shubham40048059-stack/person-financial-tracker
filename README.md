# Personal Financial Tracker 💰

A complete, modern, portfolio-quality **Personal Financial Tracker** web application built with **Python**, **Streamlit**, **Pandas**, and **Plotly**.

Designed with an intuitive fintech dashboard interface, dynamic multi-filter controls, interactive charts, auto-categorization, Indian Rupee (`₹`) currency formatting, and privacy-first session handling. Ready to publish on GitHub and deploy directly to Streamlit Community Cloud.

---

## 🚀 Live Demo & Repository
* **GitHub Repository:** [https://github.com/shubham40048059-stack/person-financial-tracker.git](https://github.com/shubham40048059-stack/person-financial-tracker.git)
* **Default Database:** Synthetic 3-month demo transactions (`data/demo_transactions.csv`)

---

## ✨ Features

### 🏠 1. Financial Overview (Dashboard)
* **Metric Cards:** Current Balance, Total Income, Total Expenses, Net Cash Flow, Number of Transactions, and Average Expense.
* **Your Financial Summary:** Automatically generated descriptive English insights highlighting cash-flow status, top spending categories, and peak expense records without speculative financial advice.
* **Interactive Visualizations:**
  * **Balance Trend:** Interactive timeline of account balance over time.
  * **Monthly Income vs Expenses:** Side-by-side comparative bar chart grouped by month.
  * **Expense by Category:** Donut chart breakdown across lifestyle categories (Food, Groceries, Travel, Transport, Shopping, Healthcare, Utilities, Fuel, etc.).
  * **Monthly Expenses:** Aggregate monthly outflow trends.
  * **Spending Trend:** Daily spending trajectory over time.
  * **Top Expenses:** Horizontal bar chart ranking the 10 largest expenses.
  * **Transaction Distribution:** Proportional breakdown of income vs. expense transactions.

### 💳 2. Transaction Explorer
* Search, sort, and inspect individual records.
* Formatted currency in Indian Rupees (`₹`).
* Real-time transaction counter and filtered summary metrics.
* **CSV Export:** Download filtered transactions with a single click.

### 📊 3. Deep Analytics
* **Income Analysis:** In-depth metrics, income by month, and income sources by category.
* **Expense Analysis:** Average and peak outflows, category distribution, and top 10 expense rankings.
* **Cash Flow Health:** Net cash flow calculation and savings/cash-flow rate percentage.
* **Monthly Comparison:** Standout cards highlighting Best Cash-Flow Month, Highest Spending Month, and Highest Income Month.

### 📁 4. Upload Your Own Data
* Seamlessly upload **CSV**, **XLSX**, or **XLS** bank statements or personal registers.
* **Intelligent Header Detection:** Automatically recognizes aliases for Date, Description/Narration, Income/Deposit/Credit, Expense/Withdrawal/Debit, Balance, and Amount.
* **Automatic Category Detection:** Keyword-based classification engine for travel, food delivery, utilities, fuel, medical, and shopping.
* **Instant Session Reset:** Easily switch back to the demo dataset anytime.

### ℹ️ 5. Privacy & Security
* Zero permanent server storage: uploaded personal bank statements reside strictly in browser session memory.
* The repository contains **only synthetic demo data**; no real banking details, account numbers, or credentials are used.

---

## 🛠️ Technology Stack

* **Python 3.10+**
* **Streamlit:** Fast, reactive web app interface
* **Pandas:** High-performance tabular data manipulation and aggregation
* **Plotly:** Interactive, responsive, production-ready charting
* **OpenPyXL:** Excel file parser (`.xlsx`, `.xls`)
* **NumPy:** Mathematical and vector operations

---

## 📁 Project Structure

```text
personal-financial-tracker/
├── app.py                     # Main Streamlit web application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation & portfolio guide
├── .gitignore                 # Git ignore rules for privacy & bytecode
│
├── data/
│   └── demo_transactions.csv  # 3-month synthetic demo dataset
│
├── utils/
│   ├── __init__.py            # Package exports
│   ├── data_loader.py         # CSV & Excel loading, normalization, aliases
│   ├── analyzer.py            # Financial metrics, aggregations & insights
│   ├── categorizer.py         # Keyword-based auto-categorization
│   └── helpers.py             # Indian Rupee formatting (₹) & safe parsers
│
└── .streamlit/
    └── config.toml            # UI theme & server configuration
```

---

## 💻 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/shubham40048059-stack/person-financial-tracker.git
cd person-financial-tracker
```

### 2. Create a virtual environment (optional but recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## 🐙 GitHub Setup & Push

To push the project to your GitHub repository:

```bash
# 1. Initialize git (if not already initialized)
git init

# 2. Add remote origin
git remote add origin https://github.com/shubham40048059-stack/person-financial-tracker.git

# 3. Stage project files
git add .

# 4. Commit changes
git commit -m "Initial commit: portfolio-ready Personal Financial Tracker"

# 5. Push to main branch
git branch -M main
git push -u origin main
```

> [!NOTE]
> The `.gitignore` is pre-configured to ensure only `data/demo_transactions.csv` is committed. Personal spreadsheets, `.xlsx`, `.xls`, and environment files are strictly ignored.

---

## ☁️ Streamlit Community Cloud Deployment

Deploy the project for free in under 2 minutes:

1. Push the code to your GitHub repository: `https://github.com/shubham40048059-stack/person-financial-tracker.git`.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New app"**.
4. In **Repository**, choose `shubham40048059-stack/person-financial-tracker`.
5. In **Branch**, select `main`.
6. In **Main file path**, enter `app.py`.
7. Click **"Deploy!"**.

---

## 🔒 Privacy & Safety

* **Synthetic Demo Data:** The default dataset contains 3 months of synthetic transactions. It does not include any real bank account numbers, UPI IDs, passwords, phone numbers, or confidential financial credentials.
* **Session In-Memory Processing:** When a user uploads a statement, it is processed only during the active browser session. Files are never stored permanently on the host machine.
* **Security Advice:** Never upload private account passwords, internet banking credentials, PINs, or OTPs.

---

## 📄 License
MIT License. Free for educational, commercial, and personal portfolio use.
