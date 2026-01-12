import os
import json
import time
from typing import List, Dict

import pandas as pd
import requests


WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FMP_BASE = "https://financialmodelingprep.com/api/v3"
FMP_API_KEY = os.environ.get("FMP_API_KEY")


def fetch_sp500_wikipedia() -> pd.DataFrame:
    """
    Fetch S&P 500 constituents from Wikipedia.
    """
    tables = pd.read_html(WIKI_URL)
    # The first table is the S&P 500 list on the current page structure
    df = tables[0]

    # Standardize column names
    df = df.rename(
        columns={
            "Symbol": "Symbol",
            "Security": "Company",
            "GICS Sector": "Sector",
            "GICS Sub-Industry": "SubIndustry",
            "Headquarters Location": "Headquarters",
            "Date first added": "DateFirstAdded",
            "CIK": "CIK",
            "Founded": "Founded",
        }
    )

    # Keep only the columns we care about
    df = df[
        [
            "Symbol",
            "Company",
            "Sector",
            "SubIndustry",
            "Headquarters",
            "DateFirstAdded",
            "CIK",
            "Founded",
        ]
    ]

    return df


def chunk_list(lst: List[str], size: int) -> List[List[str]]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def fetch_fmp_endpoint(endpoint: str, symbols: List[str]) -> List[Dict]:
    """
    Fetch data from an FMP endpoint for a list of symbols, in batches.
    """
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY environment variable is not set.")

    results: List[Dict] = []

    # FMP supports comma-separated tickers, but we keep batch size modest
    for batch in chunk_list(symbols, 50):
        tickers_str = ",".join(batch)
        url = f"{FMP_BASE}/{endpoint}/{tickers_str}"
        params = {"apikey": FMP_API_KEY}

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Some endpoints return dict, some list
        if isinstance(data, dict):
            # Normalize into list
            data = [data]

        results.extend(data)

        # Be gentle on the API
        time.sleep(1)

    return results


def to_symbol_keyed(df: pd.DataFrame, key_col: str = "symbol") -> Dict[str, Dict]:
    """
    Convert a DataFrame with a 'symbol'-like column into dict keyed by uppercased symbol.
    """
    out: Dict[str, Dict] = {}
    for _, row in df.iterrows():
        symbol = str(row[key_col]).upper()
        out[symbol] = row.to_dict()
    return out


def fetch_fmp_data(symbols: List[str]) -> pd.DataFrame:
    """
    Fetch and combine financial data from FMP for the given symbols.
    We use:
      - /profile/ for basic info (exchange, country, industry, sector)
      - /key-metrics-ttm/ for valuation/leverage metrics
      - /ratios-ttm/ for coverage etc.
    """
    symbols = [s.upper() for s in symbols]

    # Profile
    profile_raw = fetch_fmp_endpoint("profile", symbols)
    profile_df = pd.DataFrame(profile_raw)
    if not profile_df.empty:
        profile_df = profile_df.rename(
            columns={
                "symbol": "Symbol",
                "companyName": "FMP_CompanyName",
                "exchangeShortName": "Exchange",
                "industry": "FMP_Industry",
                "sector": "FMP_Sector",
                "price": "Price",
                "mktCap": "MarketCap",
                "beta": "Beta",
            }
        )
        profile_df = profile_df[
            [
                "Symbol",
                "FMP_CompanyName",
                "Exchange",
                "FMP_Industry",
                "FMP_Sector",
                "Price",
                "MarketCap",
                "Beta",
            ]
        ]

    # Key metrics TTM
    km_raw = fetch_fmp_endpoint("key-metrics-ttm", symbols)
    km_df = pd.DataFrame(km_raw)
    if not km_df.empty:
        km_df = km_df.rename(
            columns={
                "symbol": "Symbol",
                "peTTM": "PE_TTM",
                "dividendYieldTTM": "DividendYield_TTM",
                "dividendPerShareTTM": "DividendRate_TTM",
                "payoutRatioTTM": "PayoutRatio_TTM",
                "netDebtToEBITDATTM": "NetDebtToEBITDA_TTM",
                "debtToEquityTTM": "DebtToEquity_TTM",
            }
        )
        km_df = km_df[
            [
                "Symbol",
                "PE_TTM",
                "DividendYield_TTM",
                "DividendRate_TTM",
                "PayoutRatio_TTM",
                "NetDebtToEBITDA_TTM",
                "DebtToEquity_TTM",
            ]
        ]

    # Ratios TTM (for interest coverage etc., where available)
    ratios_raw = fetch_fmp_endpoint("ratios-ttm", symbols)
    ratios_df = pd.DataFrame(ratios_raw)
    if not ratios_df.empty:
        ratios_df = ratios_df.rename(
            columns={
                "symbol": "Symbol",
                "interestCoverageTTM": "InterestCoverage_TTM",
            }
        )
        ratios_df = ratios_df[["Symbol", "InterestCoverage_TTM"]]

    # Merge all FMP pieces
    df = profile_df

    if not km_df.empty:
        df = df.merge(km_df, on="Symbol", how="left")

    if not ratios_df.empty:
        df = df.merge(ratios_df, on="Symbol", how="left")

    return df


def main():
    # Paths relative to repo root
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    # 1) Fetch S&P 500 metadata
    print("Fetching S&P 500 metadata from Wikipedia...")
    sp_df = fetch_sp500_wikipedia()

    # 2) Fetch FMP data
    symbols = sp_df["Symbol"].dropna().astype(str).tolist()
    print(f"Fetching FMP financial data for {len(symbols)} symbols...")
    fmp_df = fetch_fmp_data(symbols)

    # 3) Merge
    print("Merging datasets...")
    combined = sp_df.merge(fmp_df, on="Symbol", how="left")

    # 4) Save to JSON
    sp_path = os.path.join(data_dir, "sp500_metadata.json")
    fmp_path = os.path.join(data_dir, "fmp_financials.json")
    combined_path = os.path.join(data_dir, "combined.json")

    sp_df.to_json(sp_path, orient="records")
    fmp_df.to_json(fmp_path, orient="records")
    combined.to_json(combined_path, orient="records")

    print("Saved:")
    print(f"  {sp_path}")
    print(f"  {fmp_path}")
    print(f"  {combined_path}")


if __name__ == "__main__":
    main()
