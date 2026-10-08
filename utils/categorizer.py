"""
Automatic transaction categorization rules and engine.
"""
from __future__ import annotations

from typing import Dict, List

# Keyword-to-category mapping for financial transactions
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Travel": [
        "IRCTC", "RAILWAY", "TRAIN", "FLIGHT", "AIRLINE", "INDIGO", "AIR INDIA",
        "SPICEJET", "VISTARA", "MAKEMYTRIP", "YATRA", "GOIBIBO", "HOTEL",
        "BOOKING.COM", "TRIP", "AIRBNB", "CLEARTRIP", "BUS TICKET", "RED BUS"
    ],
    "Transport": [
        "RAPIDO", "UBER", "OLA", "METRO", "CAB", "TAXI", "AUTO", "BUS",
        "FASTAG", "TOLL", "PARKING"
    ],
    "Mobile & Internet": [
        "AIRTEL", "JIO", "VODAFONE", "VI PREPAID", "VI POSTPAID", "BSNL",
        "RECHARGE", "MOBILE", "INTERNET", "BROADBAND", "ACT FIBERNET",
        "TATA PLAY", "DTH", "WIFI"
    ],
    "Fuel": [
        "PETROL", "DIESEL", "FUEL", "CNG", "HPCL", "BPCL", "IOCL",
        "INDIAN OIL", "BHARAT PETROLEUM", "HINDUSTAN PETROLEUM", "SHELL"
    ],
    "Groceries": [
        "GROCERY", "SUPERMARKET", "MART", "BIGBASKET", "BLINKIT", "ZEPTO",
        "INSTAMART", "DMART", "RELIANCE FRESH", "NATURES BASKET", "VEGETABLE",
        "FRUIT", "MILK", "SPICE", "PROVISION", "KIRANA"
    ],
    "Food": [
        "ZOMATO", "SWIGGY", "RESTAURANT", "CAFE", "FOOD", "PIZZA", "BURGER",
        "DOMINOS", "MCDONALD", "STARBUCKS", "KFC", "SUBWAY", "BIRAYANI",
        "BAKERY", "DINING", "EATERY", "SWEETS", "CHAI"
    ],
    "Healthcare": [
        "PHARMACY", "HOSPITAL", "CLINIC", "MEDICAL", "MEDICINE", "HEALTH",
        "APOLLO", "1MG", "NETMEDS", "MEDPLUS", "DOCTOR", "DENTAL",
        "DIAGNOSTIC", "LAB", "TEST", "OPTICAL"
    ],
    "Shopping": [
        "FASHION", "CLOTHING", "SHOPPING", "AMAZON", "FLIPKART", "MYNTRA",
        "MEESHO", "AJIO", "ZARA", "H&M", "MAX FASHION", "LIFESTYLE",
        "WESTSIDE", "APPAREL", "SHOES", "ELECTRONICS", "CROMA", "RELIANCE DIGITAL"
    ],
    "Utilities": [
        "ELECTRICITY", "POWER", "WATER", "GAS", "BESCOM", "MSEB", "TNEB",
        "UPPCL", "MAHAGENCO", "CYLINDER", "INDANE", "BHARAT GAS", "HP GAS",
        "PIPED GAS", "MAINTENANCE", "MUNICIPAL"
    ],
    "Cash Withdrawal": [
        "ATM", "CASH WITHDRAWAL", "SELF WITHDRAWAL", "CASH WDL", "CWDR"
    ],
    "Entertainment": [
        "NETFLIX", "SPOTIFY", "CINEMA", "PVR", "INOX", "BOOKMYSHOW",
        "PRIME VIDEO", "HOTSTAR", "DISNEY+", "YOUTUBE PREMIUM", "GAMING",
        "STEAM", "PLAYSTATION"
    ],
    "Salary": [
        "SALARY", "PAYROLL", "STIPEND", "WAGES", "MONTHLY SALARY"
    ],
    "Freelance": [
        "FREELANCE", "CLIENT", "UPWORK", "FIVERR", "CONSULTING", "CONTRACT"
    ],
    "Investment": [
        "ZERODHA", "GROWW", "UPSTOX", "MUTUAL FUND", "SIP", "DIVIDEND",
        "INTEREST", "FIXED DEPOSIT", "FD", "BONDS", "SHARES"
    ],
    "Other": []
}


def categorize_transaction(description: str, txn_type: str = "") -> str:
    """
    Classify a transaction description using keyword rules.
    If the transaction type is Income and mentions salary/credit, categorize appropriately.
    Returns the recognized category name, defaulting to 'Other'.
    """
    if not description:
        return "Salary" if txn_type == "Income" else "Other"

    desc_upper = str(description).upper()

    # Prioritize income keywords if type is Income
    if txn_type == "Income":
        for cat in ["Salary", "Freelance", "Investment"]:
            for keyword in CATEGORY_KEYWORDS.get(cat, []):
                if keyword in desc_upper:
                    return cat

    # General check across categories (excluding 'Other')
    for category, keywords in CATEGORY_KEYWORDS.items():
        if category == "Other":
            continue
        for keyword in keywords:
            if keyword in desc_upper:
                return category

    # Default fallback based on type
    if txn_type == "Income":
        return "Other Income"
    return "Other"
