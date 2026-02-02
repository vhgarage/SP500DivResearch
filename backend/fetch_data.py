import os
import json
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import yfinance as yf


WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


# -----------------------------
# Wikipedia: S&P 500 constituents
# -----------------------------
def fetch_sp500_wikipedia() -> pd.DataFrame:
    """
    Fetch S&P 500 constituents from Wikipedia using a browser-like request
    and gracefully handle column name changes.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()

    # NOTE: FutureWarning from pandas about literal HTML can be ignored for now
    tables = pd.read_html(resp.text)
    df = tables[0]

    # Debug: see actual columns
    print("Wikipedia columns:", df.columns.tolist())

    # Map current Wikipedia columns to standardized names.
    # As of your last run, columns were:
    # ['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry',
    #  'Headquarters Location', 'Date added', 'CIK', 'Founded']
    rename_map = {
        "Symbol": "Symbol",
        "Security": "Company",
        "GICS Sector": "Sector",
        "GICS Sub-Industry": "SubIndustry",
        "Headquarters Location": "Headquarters",
        "Date added": "DateAdded",  # note: not "Date first added" anymore
        "CIK": "CIK",
        "Founded": "Founded",
    }

    existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing_renames)

    selected_cols = list(existing_renames.values())
    df = df[selected_cols]

    return df


# -----------------------------
# Helpers for TTM & recent values
# -----------------------------
def _sum_last_n_columns(df: pd.DataFrame, label: str, n: int = 4) -> Optional[float]:
    """
    Sum the last n columns (most recent periods) for a given row label in a
    wide-format DataFrame (columns = dates, index = line items).
    Returns None if label or data not available.
    """
    if df is None or df.empty:
        return None
    if label not in df.index:
        return None

    # Columns are typically dates from oldest to newest or vice versa; we want most recent n
    cols = df.columns[-n:]
    values = df.loc[label, cols].dropna()
    if values.empty:
        return None
    return float(values.sum())


def _last_annual_value(df: pd.DataFrame, label: str) -> Optional[float]:
    """
    Get the most recent annual value for a label from a wide-format DataFrame.
    """
    if df is None or df.empty:
        return None
    if label not in df.index:
        return None
    # Most recent column
    col = df.columns[-1]
    val = df.loc[label, col]
    if pd.isna(val):
        return None
    return float(val)


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


# -----------------------------
# Yahoo Finance per-symbol fetch
# -----------------------------
def fetch_yf_for_symbol(symbol: str) -> Dict[str, Any]:
    """
    Fetch extended metrics for a single symbol using yfinance
    with a hybrid approach:
      - TTM (sum of last 4 quarters) for income/cash-flow metrics
      - Latest annual for balance-sheet metrics
    """
    print(f"Fetching Yahoo Finance data for {symbol}...")
    ticker = yf.Ticker(symbol)

    # Basic info
    info = ticker.info or {}

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    market_cap = info.get("marketCap")
    beta = info.get("beta")
    trailing_pe = info.get("trailingPE")

    # Dividends (TTM)
    dividends = ticker.dividends
    dividend_ttm = None
    dividend_yield = None
    payout_ratio = None

    if dividends is not None and not dividends.empty:
        last_year = dividends[dividends.index >= (dividends.index.max() - pd.DateOffset(years=1))]
        if not last_year.empty:
            dividend_ttm = float(last_year.sum())
            if price not in (None, 0):
                dividend_yield = dividend_ttm / price

    # Financial statements (yfinance DataFrames)
    q_income = ticker.quarterly_financials
    a_income = ticker.financials
    q_cf = ticker.quarterly_cashflow
    a_cf = ticker.cashflow
    a_bs = ticker.balance_sheet

    # TTM metrics from quarterly income statement
    revenue_ttm = _sum_last_n_columns(q_income, "Total Revenue")
    ebit_ttm = _sum_last_n_columns(q_income, "Ebit")
    operating_income_ttm = _sum_last_n_columns(q_income, "Operating Income")
    net_income_ttm = _sum_last_n_columns(q_income, "Net Income")

    # EPS TTM
    shares_out = info.get("sharesOutstanding")
    eps_ttm = None
    if net_income_ttm is not None and shares_out not in (None, 0):
        eps_ttm = net_income_ttm / shares_out

    # EBITDA TTM
    ebitda_ttm = _sum_last_n_columns(q_income, "Ebitda")

    # Free cash flow TTM
    ocf_ttm = _sum_last_n_columns(q_cf, "Total Cash From Operating Activities")
    capex_ttm = _sum_last_n_columns(q_cf, "Capital Expenditures")
    fcf_ttm = None
    if ocf_ttm is not None and capex_ttm is not None:
        fcf_ttm = ocf_ttm - capex_ttm

    # Annual balance sheet metrics
    total_debt = None
    cash = None
    total_equity = None
    total_assets = None

    if a_bs is not None and not a_bs.empty:
        total_debt = _last_annual_value(a_bs, "Total Debt")
        if total_debt is None:
            short_debt = _last_annual_value(a_bs, "Short Long Term Debt") or 0.0
            long_debt = _last_annual_value(a_bs, "Long Term Debt") or 0.0
            total_debt = short_debt + long_debt if (short_debt or long_debt) else None

        cash = _last_annual_value(a_bs, "Cash And Cash Equivalents")
        if cash is None:
            cash = _last_annual_value(a_bs, "Cash")

        total_equity = _last_annual_value(a_bs, "Total Stockholder Equity")
        total_assets = _last_annual_value(a_bs, "Total Assets")

    # Interest expense (hybrid)
    interest_expense_ttm = _sum_last_n_columns(q_income, "Interest Expense")
    interest_expense_annual = _last_annual_value(a_income, "Interest Expense")
    interest_expense = interest_expense_ttm if interest_expense_ttm is not None else interest_expense_annual

    # Hybrid / derived metrics
    pe_ttm = None
    if price not in (None, 0) and eps_ttm not in (None, 0):
        pe_ttm = price / eps_ttm

    if dividend_ttm is not None and eps_ttm not in (None, 0):
        payout_ratio = dividend_ttm / eps_ttm

    net_debt = None
    if total_debt is not None:
        net_debt = total_debt - (cash or 0.0)

    net_debt_to_ebitda = _safe_div(net_debt, ebitda_ttm)
    debt_to_equity = _safe_div(total_debt, total_equity)
    interest_coverage = _safe_div(ebit_ttm, interest_expense)

    # Margins & returns
    operating_margin = _safe_div(operating_income_ttm, revenue_ttm)
    profit_margin = _safe_div(net_income_ttm, revenue_ttm)
    roe = _safe_div(net_income_ttm, total_equity)
    roa = _safe_div(net_income_ttm, total_assets)

    # Build result dictionary
    result: Dict[str, Any] = {
        "Symbol": symbol,
        "Price": price,
        "MarketCap": market_cap,
        "Beta": beta,
        "PE_TTM": pe_ttm,
        "EPS_TTM": eps_ttm,
        "Dividend_TTM": dividend_ttm,
        "DividendYield_TTM": dividend_yield,
        "PayoutRatio_TTM": payout_ratio,
        "Revenue_TTM": revenue_ttm,
        "EBIT_TTM": ebit_ttm,
        "EBITDA_TTM": ebitda_ttm,
        "NetIncome_TTM": net_income_ttm,
        "FreeCashFlow_TTM": fcf_ttm,
        "OperatingMargin_TTM": operating_margin,
        "ProfitMargin_TTM": profit_margin,
        "ROE_TTM": roe,
        "ROA_TTM": roa,
        "TotalDebt_Annual": total_debt,
        "Cash_Annual": cash,
        "TotalEquity_Annual": total_equity,
        "TotalAssets_Annual": total_assets,
        "NetDebt": net_debt,
        "NetDebtToEBITDA_TTM": net_debt_to_ebitda,
        "DebtToEquity_Annual": debt_to_equity,
        "InterestExpense_Hybrid": interest_expense,
        "InterestCoverage_TTM": interest_coverage,
    }

    return result
    
# -----------------------------
# Parallel fetching orchestration
# -----------------------------
def choose_batch_size(n_symbols: int) -> int:
    """
    Simple 'auto-tuning' of batch size based on number of symbols.
    This avoids overloading the runner while still being fast.
    """
    if n_symbols > 400:
        return 25
    if n_symbols > 250:
        return 20
    if n_symbols > 100:
        return 15
    return 10


def fetch_all_yf_data(symbols: List[str]) -> pd.DataFrame:
    """
    Fetch Yahoo Finance metrics for all symbols using yfinance,
    with:
      - automatic resume from partial results
      - caching of previously fetched symbols
      - randomized throttling to avoid Yahoo rate limits
      - periodic cooldowns
      - robust error handling
    """

    import random

    symbols = [s.upper() for s in symbols]
    n = len(symbols)
    print(f"Fetching Yahoo Finance data for {n} symbols using yfinance...")

    # ------------------------------------------------------------
    # 1. Load existing cache (if any)
    # ------------------------------------------------------------
    cache_path = "data/yf_financials.json"
    cache = {}

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached_list = json.load(f)
                cache = {item["Symbol"]: item for item in cached_list}
            print(f"Loaded {len(cache)} cached symbols. Will resume from there.")
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")

    results = []

    # ------------------------------------------------------------
    # 2. Iterate through symbols with resume logic
    # ------------------------------------------------------------
    for idx, symbol in enumerate(symbols):
        print(f"[{idx+1}/{n}] {symbol}")

        # Skip if already cached
        if symbol in cache:
            print("  → Using cached data")
            results.append(cache[symbol])
            continue

        # Fetch fresh data
        try:
            data = fetch_yf_for_symbol(symbol)
            results.append(data)

            # Update cache immediately
            cache[symbol] = data
            with open(cache_path, "w") as f:
                json.dump(list(cache.values()), f, indent=2)

        except Exception as e:
            print(f"  → Failed to fetch {symbol}: {e}")
            error_entry = {"Symbol": symbol, "Error": str(e)}
            results.append(error_entry)

            # Cache the failure too
            cache[symbol] = error_entry
            with open(cache_path, "w") as f:
                json.dump(list(cache.values()), f, indent=2)

        # --------------------------------------------------------
        # Randomized delay between 0.8s and 3.0s
        # --------------------------------------------------------
        sleep_time = random.uniform(0.8, 3.0)
        time.sleep(sleep_time)

        # --------------------------------------------------------
        # Every 50 tickers, take a long cooldown
        # --------------------------------------------------------
        if (idx + 1) % 50 == 0:
            print("Reached 50‑ticker batch. Cooling down for 60 seconds...")
            time.sleep(60)

    # ------------------------------------------------------------
    # 3. Return DataFrame
    # ------------------------------------------------------------
    return pd.DataFrame(results)    
    
# -----------------------------
# Main orchestration
# -----------------------------
def main():
    # Paths relative to repo root
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    # 1) Fetch S&P 500 metadata
    print("Fetching S&P 500 metadata from Wikipedia...")
    sp_df = fetch_sp500_wikipedia()

    # 2) Fetch Yahoo Finance data
    symbols = sp_df["Symbol"].dropna().astype(str).tolist()
    print(f"Fetching Yahoo Finance financial data for {len(symbols)} symbols...")
    yf_df = fetch_all_yf_data(symbols)

    # 3) Merge
    print("Merging datasets...")
    combined = sp_df.merge(yf_df, on="Symbol", how="left")

    # 4) Save to JSON
    sp_path = os.path.join(data_dir, "sp500_metadata.json")
    yf_path = os.path.join(data_dir, "yf_financials.json")
    combined_path = os.path.join(data_dir, "combined.json")

    sp_df.to_json(sp_path, orient="records")
    yf_df.to_json(yf_path, orient="records")
    combined.to_json(combined_path, orient="records")

    print("Saved:")
    print(f"  {sp_path}")
    print(f"  {yf_path}")
    print(f"  {combined_path}")


if __name__ == "__main__":
    main()
