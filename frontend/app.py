import json
import os

import numpy as np
import pandas as pd
import streamlit as st


def load_combined_data() -> pd.DataFrame:
    """Load pre-fetched combined dataset from /data/combined.json."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "data", "combined.json")

    if not os.path.exists(data_path):
        st.error("Data file 'combined.json' not found in /data. "
                 "Make sure the GitHub Action has run at least once.")
        return pd.DataFrame()

    with open(data_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Ensure numeric types where needed
    numeric_cols = [
        "Price",
        "MarketCap",
        "Beta",
        "PE_TTM",
        "DividendYield_TTM",
        "DividendRate_TTM",
        "PayoutRatio_TTM",
        "NetDebtToEBITDA_TTM",
        "DebtToEquity_TTM",
        "InterestCoverage_TTM",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def kpi_card(label: str, value, fmt: str = None):
    if fmt and isinstance(value, (int, float, np.number)) and not np.isnan(value):
        text = fmt.format(value)
    else:
        text = str(value)
    st.metric(label, text)


def main():
    st.set_page_config(
        page_title="SP500 Research Dashboard",
        layout="wide",
    )

    st.title("SP500 Research Dashboard")
    st.caption("Real-time S&P 500 market, valuation, and income profile")

    df = load_combined_data()
    if df.empty:
        return

    # Sidebar filters
    st.sidebar.header("Filters")

    sectors = sorted(df["Sector"].dropna().unique().tolist())
    selected_sectors = st.sidebar.multiselect(
        "Sector", options=sectors, default=sectors
    )

    # Basic valuation filter
    pe_min, pe_max = st.sidebar.slider(
        "PE (TTM) range",
        float(np.nanmin(df["PE_TTM"])) if df["PE_TTM"].notna().any() else 0.0,
        float(np.nanmax(df["PE_TTM"])) if df["PE_TTM"].notna().any() else 50.0,
        (0.0, 50.0),
    )

    # Dividend filter
    yield_min, yield_max = st.sidebar.slider(
        "Dividend Yield (TTM, %)",
        0.0,
        float(
            np.nanmax(df["DividendYield_TTM"]) * 100.0
            if df["DividendYield_TTM"].notna().any()
            else 10.0
        ),
        (0.0, 10.0),
    )

    # Apply filters
    filtered = df.copy()
    filtered = filtered[filtered["Sector"].isin(selected_sectors)]

    if "PE_TTM" in filtered.columns:
        filtered = filtered[
            (filtered["PE_TTM"].isna())
            | ((filtered["PE_TTM"] >= pe_min) & (filtered["PE_TTM"] <= pe_max))
        ]

    if "DividendYield_TTM" in filtered.columns:
        dy = filtered["DividendYield_TTM"] * 100.0
        mask = dy.isna() | ((dy >= yield_min) & (dy <= yield_max))
        filtered = filtered[mask]

    st.subheader("Market overview")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_mcap = filtered["MarketCap"].sum()
        kpi_card("Total Market Cap", total_mcap, fmt="${:,.0f}")

    with col2:
        avg_pe = filtered["PE_TTM"].mean()
        kpi_card("Average PE (TTM)", avg_pe, fmt="{:.1f}x")

    with col3:
        avg_yield = (filtered["DividendYield_TTM"] * 100.0).mean()
        kpi_card("Average Dividend Yield (TTM)", avg_yield, fmt="{:.2f}%")

    with col4:
        avg_ndebt_ebitda = filtered["NetDebtToEBITDA_TTM"].mean()
        kpi_card("Avg Net Debt / EBITDA (TTM)", avg_ndebt_ebitda, fmt="{:.2f}x")

    # Sector distribution
    st.subheader("Sector distribution")

    sector_group = (
        filtered.groupby("Sector")
        .agg(
            MarketCap=("MarketCap", "sum"),
            Count=("Symbol", "count"),
            AvgPE=("PE_TTM", "mean"),
            AvgYield=("DividendYield_TTM", lambda x: (x * 100.0).mean()),
        )
        .sort_values("MarketCap", ascending=False)
    )

    st.dataframe(
        sector_group.style.format(
            {
                "MarketCap": "${:,.0f}".format,
                "AvgPE": "{:.1f}x".format,
                "AvgYield": "{:.2f}%".format,
            }
        ),
        use_container_width=True,
    )

    # Charts
    st.subheader("Charts")

    c1, c2 = st.columns(2)

    with c1:
        if "Sector" in filtered.columns and "MarketCap" in filtered.columns:
            st.markdown("**Market Cap by Sector**")
            st.bar_chart(
                sector_group["MarketCap"],
                use_container_width=True,
            )

    with c2:
        if "PE_TTM" in filtered.columns:
            st.markdown("**PE (TTM) Distribution**")
            pe_series = filtered["PE_TTM"].dropna()
            if not pe_series.empty:
                st.histogram(pe_series, bins=30)

    c3, c4 = st.columns(2)

    with c3:
        if "DividendYield_TTM" in filtered.columns:
            st.markdown("**Dividend Yield (TTM) Distribution**")
            dy_series = (filtered["DividendYield_TTM"] * 100.0).dropna()
            if not dy_series.empty:
                st.histogram(dy_series, bins=30)

    with c4:
        if "NetDebtToEBITDA_TTM" in filtered.columns:
            st.markdown("**Net Debt / EBITDA (TTM) Distribution**")
            nde_series = filtered["NetDebtToEBITDA_TTM"].dropna()
            if not nde_series.empty:
                st.histogram(nde_series, bins=30)

    # Detailed table
    st.subheader("Company-level detail")

    display_cols = [
        "Symbol",
        "Company",
        "Sector",
        "SubIndustry",
        "Headquarters",
        "Price",
        "MarketCap",
        "PE_TTM",
        "DividendYield_TTM",
        "DividendRate_TTM",
        "PayoutRatio_TTM",
        "NetDebtToEBITDA_TTM",
        "DebtToEquity_TTM",
        "InterestCoverage_TTM",
        "DateFirstAdded",
        "Founded",
    ]
    available_cols = [c for c in display_cols if c in filtered.columns]

    table_df = filtered[available_cols].copy()
    if "DividendYield_TTM" in table_df.columns:
        table_df["DividendYield_TTM"] = table_df["DividendYield_TTM"] * 100.0

    st.dataframe(
        table_df.style.format(
            {
                "Price": "${:,.2f}".format,
                "MarketCap": "${:,.0f}".format,
                "PE_TTM": "{:.1f}x".format,
                "DividendYield_TTM": "{:.2f}%".format,
                "DividendRate_TTM": "${:,.2f}".format,
                "PayoutRatio_TTM": "{:.1f}%".format,
                "NetDebtToEBITDA_TTM": "{:.2f}x".format,
                "DebtToEquity_TTM": "{:.2f}x".format,
                "InterestCoverage_TTM": "{:.1f}x".format,
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Download
    st.download_button(
        "Download filtered data as CSV",
        data=table_df.to_csv(index=False),
        file_name="sp500_research_filtered.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
