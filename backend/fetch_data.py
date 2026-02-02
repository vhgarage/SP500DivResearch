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
    Two-phase adaptive Yahoo Finance fetcher using yfinance.

    Phase 1:
      - Free run (no batch limit), fetch symbol by symbol.
      - On first 429 (explicit or silent or low-quality data): switch to batch mode, sleep, restart.

    Phase 2 (batch mode):
      - Start with batch size = 75.
      - On 429 (explicit or silent or low-quality data): sleep, reduce batch size by 5 (min 5), restart.
      - Continue until all symbols are fetched.

    Includes:
      - caching
      - randomized pacing
      - cooldowns
      - strict + soft silent-429 detection
      - data quality threshold detection
      - end-of-run summary
    """

    import random
    import subprocess
    import sys

    mode_path = "data/yf_mode.txt"          # "free" or "batch"
    batch_size_path = "data/yf_batch_size.txt"
    cache_path = "data/yf_financials.json"

    # -----------------------------
    # Determine mode
    # -----------------------------
    if os.path.exists(mode_path):
        with open(mode_path, "r") as f:
            mode = f.read().strip()
    else:
        mode = "free"
        os.makedirs("data", exist_ok=True)
        with open(mode_path, "w") as f:
            f.write(mode)

    # -----------------------------
    # Load cache
    # -----------------------------
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached_list = json.load(f)
                cache = {item["Symbol"]: item for item in cached_list}
            print(f"Loaded {len(cache)} cached symbols. Will resume.")
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")

    symbols = [s.upper() for s in symbols]
    n = len(symbols)
    remaining = [s for s in symbols if s not in cache]
    print(f"Mode: {mode.upper()}")
    print(f"Total symbols: {n}, remaining: {len(remaining)}")

    # Helper to persist cache
    def save_cache():
        with open(cache_path, "w") as f:
            json.dump(list(cache.values()), f, indent=2)

    # Strict silent-429 detector
    def is_strict_silent_429(data: Dict[str, Any]) -> bool:
        return (
            data.get("Revenue_TTM") is None and
            data.get("EBIT_TTM") is None and
            data.get("NetIncome_TTM") is None and
            data.get("TotalAssets_Annual") is None
        )

    # Soft silent-429 detector (5 or more missing)
    def is_soft_silent_429(data: Dict[str, Any]) -> bool:
        missing = 0

        if data.get("Price") is None:
            missing += 1
        if data.get("Revenue_TTM") is None:
            missing += 1
        if data.get("EBIT_TTM") is None:
            missing += 1
        if data.get("EBITDA_TTM") is None:
            missing += 1
        if data.get("NetIncome_TTM") is None:
            missing += 1
        if data.get("FreeCashFlow_TTM") is None:
            missing += 1
        if data.get("TotalAssets_Annual") is None:
            missing += 1

        if data.get("quarterly_financials_empty", False):
            missing += 1
        if data.get("quarterly_cashflow_empty", False):
            missing += 1

        return missing >= 5

    # Data Quality Threshold (DQT): too few meaningful fields present
    def is_low_quality(data: Dict[str, Any]) -> bool:
        present = 0

        if data.get("Price") not in (None, 0):
            present += 1
        if data.get("Revenue_TTM") is not None:
            present += 1
        if data.get("EBIT_TTM") is not None:
            present += 1
        if data.get("EBITDA_TTM") is not None:
            present += 1
        if data.get("NetIncome_TTM") is not None:
            present += 1
        if data.get("FreeCashFlow_TTM") is not None:
            present += 1
        if data.get("TotalAssets_Annual") is not None:
            present += 1
        if not data.get("quarterly_financials_empty", False):
            present += 1
        if not data.get("quarterly_cashflow_empty", False):
            present += 1

        # If fewer than 3 meaningful signals, treat as throttled/junk
        return present < 3

    # Unified throttling check
    def is_throttled_data(data: Dict[str, Any]) -> bool:
        return (
            is_strict_silent_429(data) or
            is_soft_silent_429(data) or
            is_low_quality(data)
        )

    # Helper to handle 429 / throttling
    def handle_429(current_batch_size: Optional[int] = None):
        nonlocal mode
        print("  → Throttling detected (explicit or silent or low-quality). Entering cooldown...")
        time.sleep(600)  # 10 minutes

        if mode == "free":
            mode = "batch"
            with open(mode_path, "w") as f:
                f.write(mode)
            batch_size = 75
            with open(batch_size_path, "w") as f:
                f.write(str(batch_size))
            print("  → Switching to BATCH mode with batch size 75.")
        else:
            if current_batch_size is None:
                current_batch_size = 75
            new_batch_size = max(5, current_batch_size - 5)
            with open(batch_size_path, "w") as f:
                f.write(str(new_batch_size))
            print(f"  → Reducing batch size from {current_batch_size} to {new_batch_size}.")

        print("  → Restarting fetcher...")
        subprocess.Popen([sys.executable, sys.argv[0]])
        sys.exit(0)

    # -----------------------------
    # PHASE 1: FREE RUN
    # -----------------------------
    if mode == "free":
        print("Starting FREE RUN (no batch limit)...")
        for idx, symbol in enumerate(remaining):
            print(f"[{idx+1}/{len(remaining)}] {symbol}")
            try:
                data = fetch_yf_for_symbol(symbol)

                # Flags for emptiness (used by detectors)
                data["quarterly_financials_empty"] = (
                    data.get("Revenue_TTM") is None and
                    data.get("EBIT_TTM") is None
                )
                data["quarterly_cashflow_empty"] = (
                    data.get("FreeCashFlow_TTM") is None
                )

                if is_throttled_data(data):
                    print("  → Throttling detected from data quality.")
                    handle_429()

                cache[symbol] = data
                save_cache()

            except Exception as e:
                msg = str(e)
                if "429" in msg or "Too Many Requests" in msg:
                    handle_429()
                print(f"  → Failed to fetch {symbol}: {msg}")
                cache[symbol] = {"Symbol": symbol, "Error": msg}
                save_cache()

            time.sleep(random.uniform(0.8, 3.0))

        successes = sum(1 for v in cache.values() if "Error" not in v)
        failures = sum(1 for v in cache.values() if "Error" in v)
        print("\n===== SUMMARY =====")
        print(f"Mode: FREE RUN")
        print(f"Total symbols: {n}")
        print(f"Fetched successfully: {successes}")
        print(f"Failed: {failures}")
        print(f"Remaining: {n - len(cache)}")
        print("===================\n")

        return pd.DataFrame(list(cache.values()))

    # -----------------------------
    # PHASE 2: BATCH MODE
    # -----------------------------
    if os.path.exists(batch_size_path):
        with open(batch_size_path, "r") as f:
            batch_size = int(f.read().strip())
    else:
        batch_size = 75
        with open(batch_size_path, "w") as f:
            f.write(str(batch_size))

    print(f"Starting BATCH MODE with batch size {batch_size}...")
    remaining = [s for s in symbols if s not in cache]
    print(f"{len(remaining)} symbols remaining in batch mode.")

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start : batch_start + batch_size]
        print(f"\nProcessing batch of {len(batch)} symbols (batch size {batch_size})...")

        for symbol in batch:
            print(f"  Fetching {symbol}...")
            try:
                data = fetch_yf_for_symbol(symbol)

                data["quarterly_financials_empty"] = (
                    data.get("Revenue_TTM") is None and
                    data.get("EBIT_TTM") is None
                )
                data["quarterly_cashflow_empty"] = (
                    data.get("FreeCashFlow_TTM") is None
                )

                if is_throttled_data(data):
                    print("  → Throttling detected from data quality.")
                    handle_429(current_batch_size=batch_size)

                cache[symbol] = data
                save_cache()

            except Exception as e:
                msg = str(e)
                if "429" in msg or "Too Many Requests" in msg:
                    handle_429(current_batch_size=batch_size)
                print(f"  → Failed to fetch {symbol}: {msg}")
                cache[symbol] = {"Symbol": symbol, "Error": msg}
                save_cache()

            time.sleep(random.uniform(0.8, 3.0))

        print("Batch complete. Cooling down for 3 minutes...")
        time.sleep(180)

    successes = sum(1 for v in cache.values() if "Error" not in v)
    failures = sum(1 for v in cache.values() if "Error" in v)
    print("\n===== SUMMARY =====")
    print(f"Mode: BATCH MODE")
    print(f"Batch size used: {batch_size}")
    print(f"Total symbols: {n}")
    print(f"Fetched successfully: {successes}")
    print(f"Failed: {failures}")
    print(f"Remaining: {n - len(cache)}")
    print("===================\n")

    return pd.DataFrame(list(cache.values()))



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
