# SP500-Research

Real-time S&P 500 financial research dashboard built with:

- Wikipedia S&P 500 constituents
- FinancialModelingPrep (FMP) API
- Python + pandas
- Streamlit
- GitHub Actions for daily refresh

## Structure

- `backend/fetch_data.py` – fetches and merges S&P 500 + FMP data, writes JSON to `data/`
- `frontend/app.py` – Streamlit dashboard UI
- `data/` – JSON files used by the app
- `.github/workflows/refresh.yml` – daily data refresh automation

## Setup

1. Create a **public** GitHub repo and add these files.
2. In GitHub, add a repository secret `FMP_API_KEY` with your FinancialModelingPrep API key.
3. The GitHub Action will run daily and populate `data/combined.json`.

## Deploy to Streamlit Cloud

1. Go to https://streamlit.io/cloud and sign in with GitHub.
2. Click **New app**.
3. Select this repo.
4. Set main file to `frontend/app.py`.
5. Deploy.

The dashboard will use the pre-fetched data from `/data/combined.json`.
