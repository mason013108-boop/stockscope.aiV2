import io
import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_OK = True
except Exception:
    VADER_OK = False

st.set_page_config(
    page_title="StockScope AI Pro V3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: .8rem; padding-bottom: 2rem; max-width: 1320px;}
    h1 {margin-bottom: .1rem;}
    div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.18);padding:.72rem .85rem;border-radius:14px;}
    .small-note {font-size:.84rem;opacity:.72;}
    @media (max-width:700px){
        .block-container{padding-top:.35rem;padding-left:.7rem;padding-right:.7rem;}
        h1{font-size:1.75rem!important;} h2{font-size:1.35rem!important;} h3{font-size:1.1rem!important;}
        div[data-testid="stMetric"]{padding:.52rem .62rem;}
        .stButton button{min-height:46px;} input{font-size:16px!important;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 StockScope AI Pro V3")
st.caption("Stocks + crypto · multi-source fallbacks · data-quality checks · multi-factor ensemble forecast")

with st.form("asset_lookup", border=False):
    q1, q2, q3 = st.columns([1.35, 1, 1])
    with q1:
        ticker_input = st.text_input(
            "Stock or crypto symbol",
            value="AAPL",
            placeholder="AAPL, NVDA, BTC, ETH, SOL...",
        ).upper().strip()
    with q2:
        period = st.selectbox("Chart history", ["3mo", "6mo", "1y", "2y", "5y", "10y"], index=2)
    with q3:
        forecast_days = st.select_slider("Forecast horizon", options=[5, 10, 15, 20, 30, 45, 60], value=20)
    run = st.form_submit_button("Analyze Everything", type="primary", use_container_width=True)

st.caption("Stocks: AAPL · NVDA · TSLA · MSFT   |   Crypto: BTC · ETH · SOL · XRP · DOGE · ADA · BTC-USD")

CRYPTO_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "ripple",
    "DOGE": "dogecoin", "ADA": "cardano", "AVAX": "avalanche-2", "LINK": "chainlink",
    "LTC": "litecoin", "BCH": "bitcoin-cash", "DOT": "polkadot", "TRX": "tron",
    "SHIB": "shiba-inu", "UNI": "uniswap", "ATOM": "cosmos", "XLM": "stellar",
    "ETC": "ethereum-classic", "NEAR": "near", "APT": "aptos", "FIL": "filecoin",
    "ICP": "internet-computer", "HBAR": "hedera-hashgraph", "ALGO": "algorand",
    "AAVE": "aave", "MKR": "maker", "ARB": "arbitrum", "OP": "optimism",
    "SUI": "sui", "PEPE": "pepe", "POL": "polygon-ecosystem-token",
}

POSITIVE_WORDS = {
    "beat", "beats", "surge", "surges", "growth", "strong", "record", "upgrade", "upgraded",
    "profit", "profits", "bullish", "outperform", "approval", "approved", "partnership", "launch",
    "rally", "rallies", "gain", "gains", "positive", "buy", "breakthrough", "expands", "expansion",
}
NEGATIVE_WORDS = {
    "miss", "misses", "drop", "drops", "decline", "weak", "downgrade", "downgraded", "loss",
    "losses", "bearish", "underperform", "lawsuit", "probe", "investigation", "fraud", "recall",
    "cut", "cuts", "warning", "negative", "sell", "crash", "bankruptcy", "hack", "hacked",
}


def safe_float(x, default=np.nan):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def first_valid(*values):
    for value in values:
        v = safe_float(value)
        if not np.isnan(v):
            return v
    return np.nan


def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, float(x)))


def fmt_money(x):
    x = safe_float(x)
    return "Data unavailable" if np.isnan(x) else f"${x:,.2f}"


def fmt_pct(x, scale=100, signed=True):
    x = safe_float(x)
    if np.isnan(x):
        return "Data unavailable"
    sign = "+" if signed else ""
    return f"{x * scale:{sign}.2f}%"


def fmt_big(x):
    x = safe_float(x)
    if np.isnan(x):
        return "Data unavailable"
    for unit, scale in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(x) >= scale:
            return f"{x / scale:.2f}{unit}"
    return f"{x:,.0f}"


def period_days(period):
    return {"3mo": 95, "6mo": 190, "1y": 370, "2y": 740, "5y": 1830, "10y": 3660}.get(period, 370)


def normalize_input(raw):
    raw = raw.strip().upper().replace(" ", "")
    if not raw:
        return raw, "unknown", None, raw
    if raw in CRYPTO_IDS:
        return f"{raw}-USD", "crypto", CRYPTO_IDS[raw], raw
    if raw.endswith("-USD"):
        base = raw[:-4]
        return raw, "crypto", CRYPTO_IDS.get(base), base
    return raw, "equity", None, raw


def standardize_history(df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        if len(x.columns.levels) >= 2:
            # yfinance download() can place field names on either level.
            fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
            if any(c in fields for c in x.columns.get_level_values(0)):
                x.columns = x.columns.get_level_values(0)
            else:
                x.columns = x.columns.get_level_values(-1)
    rename = {c: str(c).title() for c in x.columns}
    x = x.rename(columns=rename)
    if "Adj Close" in x.columns and "Close" not in x.columns:
        x["Close"] = x["Adj Close"]
    if "Close" not in x.columns:
        return pd.DataFrame()
    for c in ["Open", "High", "Low"]:
        if c not in x.columns:
            x[c] = x["Close"]
    if "Volume" not in x.columns:
        x["Volume"] = 0.0
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["Close"]).sort_index()
    if isinstance(x.index, pd.DatetimeIndex) and x.index.tz is not None:
        x.index = x.index.tz_localize(None)
    return x[["Open", "High", "Low", "Close", "Volume"]]


def yf_history_ticker(symbol, period):
    try:
        return standardize_history(yf.Ticker(symbol).history(period=period, auto_adjust=False, repair=True))
    except Exception:
        return pd.DataFrame()


def yf_history_download(symbol, period):
    try:
        df = yf.download(symbol, period=period, auto_adjust=False, progress=False, threads=False, timeout=12)
        return standardize_history(df)
    except Exception:
        return pd.DataFrame()


def stooq_history(symbol, period):
    # Last-resort daily-history fallback for many US-listed equities. Never used to fabricate quote fields.
    try:
        if symbol.startswith("^") or "=" in symbol or symbol.endswith("-USD"):
            return pd.DataFrame()
        s = symbol.lower()
        if not s.endswith(".us"):
            s += ".us"
        url = f"https://stooq.com/q/d/l/?s={quote_plus(s)}&i=d"
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if "Date" not in df.columns:
            return pd.DataFrame()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=period_days(period))
        return standardize_history(df[df.index >= cutoff])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def coingecko_market_chart(coin_id, days=365):
    if not coin_id:
        return pd.DataFrame(), {}
    try:
        days = int(max(2, min(days, 365)))
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        r = requests.get(
            url,
            params={"vs_currency": "usd", "days": days, "interval": "daily"},
            timeout=12,
            headers={"User-Agent": "StockScopeAI/3.0"},
        )
        r.raise_for_status()
        data = r.json()
        prices = data.get("prices") or []
        vols = {int(t): v for t, v in (data.get("total_volumes") or [])}
        mcaps = {int(t): v for t, v in (data.get("market_caps") or [])}
        rows = []
        for t, p in prices:
            ts = pd.to_datetime(int(t), unit="ms", utc=True).tz_localize(None)
            nearest_vol = vols.get(int(t), np.nan)
            rows.append((ts, p, nearest_vol))
        if not rows:
            return pd.DataFrame(), {}
        df = pd.DataFrame(rows, columns=["Date", "Close", "Volume"]).set_index("Date").sort_index()
        df["Open"] = df["Close"].shift(1).fillna(df["Close"])
        df["High"] = df[["Open", "Close"]].max(axis=1)
        df["Low"] = df[["Open", "Close"]].min(axis=1)
        # Market-chart fallback does not provide true OHLC; high/low are conservative placeholders for display only.
        df = standardize_history(df)
        meta = {
            "market_cap_latest": safe_float(list(mcaps.values())[-1]) if mcaps else np.nan,
            "volume_latest": safe_float(df["Volume"].iloc[-1]) if not df.empty else np.nan,
        }
        return df, meta
    except Exception:
        return pd.DataFrame(), {}


@st.cache_data(ttl=300, show_spinner=False)
def coingecko_snapshot(coin_id):
    if not coin_id:
        return {}
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        r = requests.get(
            url,
            params={
                "ids": coin_id,
                "vs_currencies": "usd",
                "include_market_cap": "true",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
                "include_last_updated_at": "true",
            },
            timeout=10,
            headers={"User-Agent": "StockScopeAI/3.0"},
        )
        r.raise_for_status()
        return (r.json() or {}).get(coin_id, {}) or {}
    except Exception:
        return {}


def load_history_with_fallbacks(symbol, period, asset_type, coin_id=None):
    sources = []
    h = yf_history_ticker(symbol, period)
    if not h.empty:
        return h, "Yahoo Ticker.history", sources
    sources.append("Yahoo Ticker.history failed")
    h = yf_history_download(symbol, period)
    if not h.empty:
        return h, "Yahoo download fallback", sources
    sources.append("Yahoo download failed")
    if asset_type == "crypto" and coin_id:
        h, _ = coingecko_market_chart(coin_id, period_days(period))
        if not h.empty:
            return h, "CoinGecko market-chart fallback", sources
        sources.append("CoinGecko history failed")
    if asset_type != "crypto":
        h = stooq_history(symbol, period)
        if not h.empty:
            return h, "Stooq daily-history fallback", sources
        sources.append("Stooq history failed")
    return pd.DataFrame(), "Unavailable", sources


def load_model_history_with_fallbacks(symbol, asset_type, coin_id=None):
    h = yf_history_ticker(symbol, "5y")
    if not h.empty and len(h) >= 160:
        return h, "Yahoo 5y history"
    h2 = yf_history_download(symbol, "5y")
    if not h2.empty and len(h2) >= 160:
        return h2, "Yahoo download 5y fallback"
    if asset_type == "crypto" and coin_id:
        h3, _ = coingecko_market_chart(coin_id, 365)
        if not h3.empty and len(h3) >= 160:
            return h3, "CoinGecko 365d model fallback"
    if asset_type != "crypto":
        h4 = stooq_history(symbol, "5y")
        if not h4.empty and len(h4) >= 160:
            return h4, "Stooq 5y model fallback"
    # Best partial history if no source reached the ideal amount.
    choices = [x for x in [h, h2] if isinstance(x, pd.DataFrame) and not x.empty]
    if choices:
        return max(choices, key=len), "Partial Yahoo history"
    return pd.DataFrame(), "Unavailable"


def safe_fast_info(ticker):
    try:
        return dict(ticker.fast_info)
    except Exception:
        return {}


def safe_info(ticker):
    try:
        return ticker.info or {}
    except Exception:
        try:
            return ticker.get_info() or {}
        except Exception:
            return {}


def safe_financials(ticker):
    out = {}
    for name, attr in [
        ("income", "income_stmt"), ("quarterly_income", "quarterly_income_stmt"),
        ("balance", "balance_sheet"), ("quarterly_balance", "quarterly_balance_sheet"),
        ("cashflow", "cashflow"), ("quarterly_cashflow", "quarterly_cashflow"),
    ]:
        try:
            val = getattr(ticker, attr)
            out[name] = val if isinstance(val, pd.DataFrame) else pd.DataFrame()
        except Exception:
            out[name] = pd.DataFrame()
    return out


def statement_row(df, names):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)
    idx_map = {str(i).lower().replace(" ", ""): i for i in df.index}
    for name in names:
        key = name.lower().replace(" ", "")
        if key in idx_map:
            row = pd.to_numeric(df.loc[idx_map[key]], errors="coerce").dropna()
            return row
    return pd.Series(dtype=float)


def derive_fundamentals(info, financials):
    result = {
        "revenueGrowth": safe_float(info.get("revenueGrowth")),
        "earningsGrowth": safe_float(info.get("earningsGrowth")),
        "profitMargins": safe_float(info.get("profitMargins")),
        "returnOnEquity": safe_float(info.get("returnOnEquity")),
        "debtToEquity": safe_float(info.get("debtToEquity")),
        "freeCashflow": safe_float(info.get("freeCashflow")),
        "currentRatio": safe_float(info.get("currentRatio")),
        "forwardPE": safe_float(info.get("forwardPE")),
        "pegRatio": safe_float(info.get("pegRatio")),
    }
    sources = {k: "Yahoo quote" if not np.isnan(v) else None for k, v in result.items()}

    income = financials.get("income", pd.DataFrame())
    balance = financials.get("balance", pd.DataFrame())
    cashflow = financials.get("cashflow", pd.DataFrame())
    rev = statement_row(income, ["Total Revenue", "Operating Revenue"])
    ni = statement_row(income, ["Net Income", "Net Income Common Stockholders"])
    equity = statement_row(balance, ["Stockholders Equity", "Total Equity Gross Minority Interest"])
    debt = statement_row(balance, ["Total Debt"])
    current_assets = statement_row(balance, ["Current Assets", "Total Current Assets"])
    current_liab = statement_row(balance, ["Current Liabilities", "Total Current Liabilities"])
    fcf = statement_row(cashflow, ["Free Cash Flow"])
    ocf = statement_row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    capex = statement_row(cashflow, ["Capital Expenditure", "Capital Expenditures"])

    if np.isnan(result["revenueGrowth"]) and len(rev) >= 2 and rev.iloc[1] != 0:
        result["revenueGrowth"] = safe_float(rev.iloc[0] / rev.iloc[1] - 1)
        sources["revenueGrowth"] = "Calculated from income statement"
    if np.isnan(result["earningsGrowth"]) and len(ni) >= 2 and ni.iloc[1] != 0:
        result["earningsGrowth"] = safe_float(ni.iloc[0] / abs(ni.iloc[1]) - 1)
        sources["earningsGrowth"] = "Calculated from income statement"
    if np.isnan(result["profitMargins"]) and len(ni) and len(rev) and rev.iloc[0] != 0:
        result["profitMargins"] = safe_float(ni.iloc[0] / rev.iloc[0])
        sources["profitMargins"] = "Calculated from statements"
    if np.isnan(result["returnOnEquity"]) and len(ni) and len(equity) and equity.iloc[0] != 0:
        result["returnOnEquity"] = safe_float(ni.iloc[0] / equity.iloc[0])
        sources["returnOnEquity"] = "Calculated from statements"
    if np.isnan(result["debtToEquity"]) and len(debt) and len(equity) and equity.iloc[0] != 0:
        result["debtToEquity"] = safe_float(100 * debt.iloc[0] / equity.iloc[0])
        sources["debtToEquity"] = "Calculated from balance sheet"
    if np.isnan(result["freeCashflow"]):
        if len(fcf):
            result["freeCashflow"] = safe_float(fcf.iloc[0])
            sources["freeCashflow"] = "Cash-flow statement"
        elif len(ocf) and len(capex):
            result["freeCashflow"] = safe_float(ocf.iloc[0] + capex.iloc[0] if capex.iloc[0] < 0 else ocf.iloc[0] - capex.iloc[0])
            sources["freeCashflow"] = "Calculated OCF - capex"
    if np.isnan(result["currentRatio"]) and len(current_assets) and len(current_liab) and current_liab.iloc[0] != 0:
        result["currentRatio"] = safe_float(current_assets.iloc[0] / current_liab.iloc[0])
        sources["currentRatio"] = "Calculated from balance sheet"
    return result, sources


def fast_get(fast, *keys):
    if not isinstance(fast, dict):
        return np.nan
    for key in keys:
        if key in fast:
            v = safe_float(fast.get(key))
            if not np.isnan(v):
                return v
    return np.nan


def resolve_snapshot(info, fast, hist, model_hist, asset_type, cg):
    info = info or {}
    fast = fast or {}
    cg = cg or {}
    quote_candidates = []
    for label, val in [
        ("Yahoo fast", fast_get(fast, "last_price", "lastPrice")),
        ("Yahoo info", first_valid(info.get("currentPrice"), info.get("regularMarketPrice"))),
        ("CoinGecko", cg.get("usd") if asset_type == "crypto" else np.nan),
        ("Latest daily close", safe_float(hist["Close"].iloc[-1]) if not hist.empty else np.nan),
    ]:
        v = safe_float(val)
        if not np.isnan(v) and v > 0:
            quote_candidates.append((label, v))
    live = [x for x in quote_candidates if x[0] != "Latest daily close"]
    spot = np.median([v for _, v in live]) if live else (quote_candidates[0][1] if quote_candidates else np.nan)
    spot_source = "Median of live sources" if len(live) >= 2 else (live[0][0] if live else (quote_candidates[0][0] if quote_candidates else "Unavailable"))

    price_spread = np.nan
    if len(live) >= 2:
        vals = [v for _, v in live]
        price_spread = (max(vals) - min(vals)) / np.mean(vals)

    shares = first_valid(info.get("sharesOutstanding"), info.get("impliedSharesOutstanding"), fast_get(fast, "shares", "shares_outstanding"))
    market_cap = first_valid(
        info.get("marketCap"), fast_get(fast, "market_cap", "marketCap"),
        cg.get("usd_market_cap") if asset_type == "crypto" else np.nan,
        shares * spot if not np.isnan(shares) and not np.isnan(spot) else np.nan,
    )

    range_hist = model_hist if isinstance(model_hist, pd.DataFrame) and not model_hist.empty else hist
    year = range_hist.tail(365 if asset_type == "crypto" else 252) if not range_hist.empty else pd.DataFrame()
    year_high = first_valid(info.get("fiftyTwoWeekHigh"), fast_get(fast, "year_high", "yearHigh"), safe_float(year["High"].max()) if not year.empty else np.nan)
    year_low = first_valid(info.get("fiftyTwoWeekLow"), fast_get(fast, "year_low", "yearLow"), safe_float(year["Low"].min()) if not year.empty else np.nan)

    pe = np.nan
    pe_label = "P/E"
    pe_note = "Not applicable to crypto" if asset_type == "crypto" else "Unavailable after fallbacks"
    if asset_type != "crypto":
        pe = safe_float(info.get("forwardPE"))
        pe_label = "Forward P/E"
        pe_note = "Yahoo direct"
        if np.isnan(pe):
            eps = safe_float(info.get("forwardEps"))
            if not np.isnan(eps) and eps > 0 and spot > 0:
                pe = spot / eps
                pe_note = "Calculated from forward EPS"
            else:
                pe = safe_float(info.get("trailingPE"))
                teps = safe_float(info.get("trailingEps"))
                if np.isnan(pe) and not np.isnan(teps) and teps > 0 and spot > 0:
                    pe = spot / teps
                if not np.isnan(pe):
                    pe_label = "P/E (TTM fallback)"
                    pe_note = "Forward P/E unavailable"

    return {
        "spot": spot, "spot_source": spot_source, "price_sources": quote_candidates,
        "price_spread": price_spread, "market_cap": market_cap, "year_high": year_high,
        "year_low": year_low, "pe": pe, "pe_label": pe_label, "pe_note": pe_note,
    }


def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = -delta.clip(upper=0).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_features(df):
    x = df.copy()
    x["Return_1d"] = x["Close"].pct_change()
    for n in [5, 10, 20, 50, 100, 200]:
        x[f"MA{n}"] = x["Close"].rolling(n).mean()
    x["Volatility20"] = x["Return_1d"].rolling(20).std()
    x["RSI14"] = rsi(x["Close"], 14)
    x["Momentum5"] = x["Close"].pct_change(5)
    x["Momentum20"] = x["Close"].pct_change(20)
    x["Momentum60"] = x["Close"].pct_change(60)
    x["VolumeChange"] = x["Volume"].replace(0, np.nan).pct_change()
    x["VolumeRatio20"] = x["Volume"].replace(0, np.nan) / x["Volume"].replace(0, np.nan).rolling(20).mean()
    ema12 = x["Close"].ewm(span=12, adjust=False).mean()
    ema26 = x["Close"].ewm(span=26, adjust=False).mean()
    x["MACD"] = ema12 - ema26
    x["MACDSignal"] = x["MACD"].ewm(span=9, adjust=False).mean()
    return x


def sector_etf(sector):
    return {
        "Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV",
        "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP", "Industrials": "XLI",
        "Energy": "XLE", "Basic Materials": "XLB", "Real Estate": "XLRE", "Utilities": "XLU",
        "Communication Services": "XLC",
    }.get(sector, "SPY")


@st.cache_data(ttl=900, show_spinner=False)
def load_asset(symbol, chart_period, guessed_type, coin_id):
    t = yf.Ticker(symbol)
    info = safe_info(t)
    fast = safe_fast_info(t)
    quote_type = str(info.get("quoteType") or "").upper()
    asset_type = "crypto" if guessed_type == "crypto" or quote_type == "CRYPTOCURRENCY" else "equity"
    if asset_type == "crypto" and not coin_id:
        coin_id = CRYPTO_IDS.get(symbol.replace("-USD", ""))

    hist, hist_source, hist_failures = load_history_with_fallbacks(symbol, chart_period, asset_type, coin_id)
    model_hist, model_source = load_model_history_with_fallbacks(symbol, asset_type, coin_id)
    financials = safe_financials(t) if asset_type != "crypto" else {}

    try:
        news = t.news or []
    except Exception:
        news = []
    try:
        recs = t.recommendations_summary
    except Exception:
        recs = pd.DataFrame()
    try:
        raw_recs = t.recommendations
    except Exception:
        raw_recs = pd.DataFrame()
    try:
        targets = t.analyst_price_targets or {}
    except Exception:
        targets = {}
    try:
        earnings = t.get_earnings_dates(limit=8)
    except Exception:
        earnings = pd.DataFrame()
    try:
        calendar = t.calendar or {}
    except Exception:
        calendar = {}

    cg = coingecko_snapshot(coin_id) if asset_type == "crypto" and coin_id else {}
    return {
        "hist": hist, "model_hist": model_hist, "info": info, "fast": fast, "financials": financials,
        "news": news, "recs": recs, "raw_recs": raw_recs, "targets": targets, "earnings": earnings,
        "calendar": calendar, "cg": cg, "asset_type": asset_type, "coin_id": coin_id,
        "hist_source": hist_source, "hist_failures": hist_failures, "model_source": model_source,
    }


@st.cache_data(ttl=900, show_spinner=False)
def load_context(sector, asset_type):
    symbols = ["SPY", "QQQ", "^VIX", "^TNX", "DX-Y.NYB", "GC=F"]
    if asset_type == "crypto":
        symbols += ["BTC-USD", "ETH-USD"]
    else:
        symbols += [sector_etf(sector)]
    out = {}
    source = {}
    for s in dict.fromkeys(symbols):
        h = yf_history_ticker(s, "6mo")
        if not h.empty:
            out[s] = h
            source[s] = "Yahoo Ticker.history"
            continue
        h = yf_history_download(s, "6mo")
        if not h.empty:
            out[s] = h
            source[s] = "Yahoo download fallback"
    return out, source


@st.cache_data(ttl=900, show_spinner=False)
def load_options_snapshot(symbol, spot, asset_type):
    result = {"atm_iv": np.nan, "put_call_oi": np.nan, "expiration": None, "source": "Not applicable" if asset_type == "crypto" else "Unavailable"}
    if asset_type == "crypto":
        return result
    try:
        t = yf.Ticker(symbol)
        exps = t.options
        if not exps:
            return result
        exp = exps[0]
        chain = t.option_chain(exp)
        calls, puts = chain.calls.copy(), chain.puts.copy()
        if calls.empty or puts.empty:
            return result
        calls["dist"] = (calls["strike"] - spot).abs()
        puts["dist"] = (puts["strike"] - spot).abs()
        c = calls.nsmallest(3, "dist")["impliedVolatility"].replace([np.inf, -np.inf], np.nan).mean()
        p = puts.nsmallest(3, "dist")["impliedVolatility"].replace([np.inf, -np.inf], np.nan).mean()
        result["atm_iv"] = np.nanmean([c, p])
        call_oi = calls["openInterest"].fillna(0).sum()
        put_oi = puts["openInterest"].fillna(0).sum()
        result["put_call_oi"] = (put_oi / call_oi) if call_oi > 0 else np.nan
        result["expiration"] = exp
        result["source"] = "Yahoo option chain"
    except Exception:
        pass
    return result


def keyword_sentiment(text):
    words = set(re.findall(r"[a-z]+", (text or "").lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    if pos + neg == 0:
        return 0.0
    return float(np.clip((pos - neg) / max(2, pos + neg), -1, 1))


@st.cache_data(ttl=900, show_spinner=False)
def google_news_fallback(query_text):
    rows = []
    try:
        url = f"https://news.google.com/rss/search?q={quote_plus(query_text)}&hl=en-US&gl=US&ceid=US:en"
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for item in root.findall(".//item")[:20]:
            title = (item.findtext("title") or "").strip()
            source = (item.findtext("source") or "Google News").strip()
            pub = (item.findtext("pubDate") or "").strip()
            if title:
                rows.append({"title": title, "provider": source, "published": pub, "origin": "Google News RSS fallback"})
    except Exception:
        pass
    return rows


def extract_news_rows(news, company, symbol, asset_type):
    rows = []
    analyzer = SentimentIntensityAnalyzer() if VADER_OK else None
    now = datetime.now(timezone.utc)
    seen = set()

    for item in (news or [])[:25]:
        if not isinstance(item, dict):
            continue
        content = item.get("content", item)
        title = content.get("title") or item.get("title")
        if not title or title.strip().lower() in seen:
            continue
        seen.add(title.strip().lower())
        provider = ""
        p = content.get("provider")
        if isinstance(p, dict):
            provider = p.get("displayName", "")
        elif isinstance(p, str):
            provider = p
        ts = content.get("pubDate") or content.get("providerPublishTime") or item.get("providerPublishTime")
        rows.append({"title": title, "provider": provider or "Yahoo News", "published": ts, "origin": "Yahoo Finance news"})

    search_query = f'"{company}" {"crypto" if asset_type == "crypto" else "stock"}'
    if len(rows) < 5:
        for item in google_news_fallback(search_query):
            title = item["title"]
            if title.strip().lower() not in seen:
                seen.add(title.strip().lower())
                rows.append(item)

    out = []
    for row in rows[:25]:
        title = row["title"]
        ts = row.get("published")
        age_hours = np.nan
        try:
            if isinstance(ts, (int, float)):
                dt = datetime.fromtimestamp(ts, timezone.utc)
            elif isinstance(ts, str) and ts:
                try:
                    dt = pd.to_datetime(ts, utc=True).to_pydatetime()
                except Exception:
                    dt = parsedate_to_datetime(ts).astimezone(timezone.utc)
            else:
                dt = None
            if dt:
                age_hours = max(0, (now - dt).total_seconds() / 3600)
        except Exception:
            pass
        sentiment = analyzer.polarity_scores(title)["compound"] if analyzer else keyword_sentiment(title)
        out.append({"Title": title, "Source": row.get("provider") or "News", "AgeHours": age_hours, "Sentiment": sentiment, "Data source": row.get("origin", "News")})
    return pd.DataFrame(out)


def technical_score(df, asset_type):
    feat = add_features(df)
    if feat.empty:
        return np.nan, None, False
    f = feat.iloc[-1]
    p = safe_float(f["Close"])
    score = 50
    for n, w in [(20, 7), (50, 8), (100, 5), (200, 8)]:
        ma = safe_float(f.get(f"MA{n}"))
        if not np.isnan(ma):
            score += w if p > ma else -w
    rv = safe_float(f["RSI14"])
    if not np.isnan(rv):
        if 50 <= rv <= 68:
            score += 7
        elif 32 <= rv < 50:
            score -= 3
        elif rv > 78:
            score -= 8
        elif rv < 25:
            score += 5
    for key, mult, lim in [("Momentum20", 100, 10), ("Momentum60", 50, 8)]:
        m = safe_float(f[key])
        if not np.isnan(m):
            score += np.clip(m * mult, -lim, lim)
    macd, sig = safe_float(f["MACD"]), safe_float(f["MACDSignal"])
    if not np.isnan(macd) and not np.isnan(sig):
        score += 6 if macd > sig else -6
    vr = safe_float(f["VolumeRatio20"])
    m20 = safe_float(f["Momentum20"])
    if not np.isnan(vr) and vr > 1.3 and not np.isnan(m20):
        score += 4 if m20 > 0 else -4
    # Crypto gets a modest volatility penalty because very high realized volatility makes trends less dependable.
    vol20 = safe_float(f["Volatility20"])
    if asset_type == "crypto" and not np.isnan(vol20) and vol20 > 0.07:
        score -= min(8, (vol20 - 0.07) * 100)
    return clamp(score), f, True


def fundamental_score(fund):
    score = 50
    used = 0
    def add(v):
        nonlocal score, used
        score += v
        used += 1
    rg, eg = safe_float(fund.get("revenueGrowth")), safe_float(fund.get("earningsGrowth"))
    pm, roe = safe_float(fund.get("profitMargins")), safe_float(fund.get("returnOnEquity"))
    de, fcf = safe_float(fund.get("debtToEquity")), safe_float(fund.get("freeCashflow"))
    cr, pe, peg = safe_float(fund.get("currentRatio")), safe_float(fund.get("forwardPE")), safe_float(fund.get("pegRatio"))
    if not np.isnan(rg): add(np.clip(rg * 35, -12, 14))
    if not np.isnan(eg): add(np.clip(eg * 25, -12, 14))
    if not np.isnan(pm): add(np.clip((pm - .08) * 35, -8, 10))
    if not np.isnan(roe): add(np.clip((roe - .12) * 24, -8, 10))
    if not np.isnan(de): add(7 if de < 60 else 2 if de < 120 else -7 if de > 250 else -2)
    if not np.isnan(fcf): add(5 if fcf > 0 else -8)
    if not np.isnan(cr): add(4 if cr > 1.5 else -4 if cr < 0.8 else 0)
    if not np.isnan(pe): add(4 if 8 < pe < 30 else -4 if pe > 60 or pe < 0 else 0)
    if not np.isnan(peg): add(5 if 0 < peg < 1.5 else -4 if peg > 3 else 0)
    return (clamp(score), used, used >= 2)


def crypto_market_score(model_hist, snapshot, cg):
    score = 50
    used = 0
    cap = safe_float(snapshot.get("market_cap"))
    vol = first_valid(cg.get("usd_24h_vol"), safe_float(model_hist["Volume"].tail(7).median()) if not model_hist.empty else np.nan)
    if not np.isnan(cap):
        used += 1
        score += 8 if cap >= 50e9 else 5 if cap >= 10e9 else 1 if cap >= 1e9 else -4
    if not np.isnan(cap) and not np.isnan(vol) and cap > 0:
        used += 1
        ratio = vol / cap
        score += 6 if .02 <= ratio <= .20 else 1 if ratio > .005 else -5
    if len(model_hist) >= 100:
        close = model_hist["Close"]
        r30 = safe_float(close.iloc[-1] / close.iloc[-31] - 1)
        r90 = safe_float(close.iloc[-1] / close.iloc[-91] - 1)
        vol30 = safe_float(close.pct_change().tail(30).std())
        if not np.isnan(r30):
            used += 1; score += np.clip(r30 * 35, -10, 10)
        if not np.isnan(r90):
            used += 1; score += np.clip(r90 * 20, -8, 8)
        if not np.isnan(vol30):
            used += 1; score += 4 if vol30 < .035 else -6 if vol30 > .09 else 0
    return clamp(score), used, used >= 2


def analyst_score(recs, raw_recs, targets, info, spot):
    score = 50
    coverage = 0
    if isinstance(recs, pd.DataFrame) and not recs.empty:
        row = recs.iloc[0]
        sb, b = safe_float(row.get("strongBuy"), 0), safe_float(row.get("buy"), 0)
        h, s, ss = safe_float(row.get("hold"), 0), safe_float(row.get("sell"), 0), safe_float(row.get("strongSell"), 0)
        total = sb + b + h + s + ss
        if total > 0:
            score += (((sb * 2 + b) - (s + ss * 2)) / (2 * total)) * 32
            coverage += 1
    elif isinstance(raw_recs, pd.DataFrame) and not raw_recs.empty:
        # Fallback: count recent textual grades if the summary endpoint is missing.
        recent = raw_recs.tail(50).astype(str).apply(lambda col: col.str.lower())
        text = " ".join(recent.to_numpy().ravel().tolist())
        pos = sum(text.count(w) for w in ["buy", "outperform", "overweight", "upgrade"])
        neg = sum(text.count(w) for w in ["sell", "underperform", "underweight", "downgrade"])
        if pos + neg > 0:
            score += np.clip((pos - neg) / (pos + neg) * 20, -20, 20)
            coverage += 1
    mean_t = np.nan
    if isinstance(targets, dict):
        mean_t = safe_float(targets.get("mean"))
    mean_t = first_valid(mean_t, info.get("targetMeanPrice"))
    if not np.isnan(mean_t) and spot > 0:
        score += np.clip((mean_t / spot - 1) * 80, -20, 20)
        coverage += 1
    return clamp(score), coverage, mean_t, coverage > 0


def news_score(news_df):
    if news_df.empty:
        return np.nan, 0, np.nan, False
    age = news_df["AgeHours"].fillna(72)
    weights = np.exp(-age / 72).clip(lower=.08)
    weighted = float((news_df["Sentiment"] * weights).sum() / weights.sum())
    return clamp(50 + weighted * 42), len(news_df), weighted, True


def context_score(ctx, sector, asset_type, symbol):
    score = 50
    used = 0
    parts = {}
    def ret(sym, n=20):
        h = ctx.get(sym)
        if h is None or len(h) < n + 1:
            return np.nan
        return safe_float(h["Close"].iloc[-1] / h["Close"].iloc[-n - 1] - 1)

    if asset_type == "crypto":
        for sym, w, key in [("BTC-USD", 65, "BTC20"), ("ETH-USD", 45, "ETH20"), ("QQQ", 25, "QQQ20")]:
            if sym == symbol:
                continue
            r = ret(sym)
            parts[key] = r
            if not np.isnan(r):
                score += np.clip(r * w, -8, 8); used += 1
    else:
        sec = sector_etf(sector)
        for sym, w, key in [("SPY", 80, "SPY20"), ("QQQ", 40, "QQQ20"), (sec, 70, "Sector20")]:
            r = ret(sym)
            parts[key] = r
            if not np.isnan(r):
                score += np.clip(r * w, -8, 8); used += 1

    vix = ctx.get("^VIX")
    if vix is not None and not vix.empty:
        vv = safe_float(vix["Close"].iloc[-1])
        parts["VIX"] = vv
        if not np.isnan(vv):
            score += 6 if vv < 17 else -12 if vv > 35 else -6 if vv > 25 else 0
            used += 1
    tnx20 = ret("^TNX")
    dxy20 = ret("DX-Y.NYB")
    parts["TNX20"] = tnx20
    parts["DXY20"] = dxy20
    if not np.isnan(tnx20):
        score += np.clip(-tnx20 * 25, -5, 5); used += 1
    if not np.isnan(dxy20):
        score += np.clip(-dxy20 * (35 if asset_type == "crypto" else 18), -5, 5); used += 1
    return clamp(score), parts, used, used >= 2


def next_earnings_days(earnings, calendar):
    now = pd.Timestamp.now(tz="UTC")
    candidates = []
    try:
        if isinstance(earnings, pd.DataFrame) and not earnings.empty:
            idx = pd.to_datetime(earnings.index, utc=True, errors="coerce").dropna()
            candidates.extend([x for x in idx if x > now])
    except Exception:
        pass
    try:
        if isinstance(calendar, dict):
            ed = calendar.get("Earnings Date") or calendar.get("EarningsDate")
            vals = ed if isinstance(ed, (list, tuple)) else [ed]
            for v in vals:
                dt = pd.to_datetime(v, utc=True, errors="coerce")
                if not pd.isna(dt) and dt > now:
                    candidates.append(dt)
    except Exception:
        pass
    if not candidates:
        return np.nan
    return min((x - now).total_seconds() / 86400 for x in candidates)


def risk_score(info, options, earnings, calendar, model_hist, asset_type):
    score = 55
    used = 0
    daily_vol = safe_float(model_hist["Close"].pct_change().tail(60).std()) if not model_hist.empty else np.nan
    realized_annual = daily_vol * math.sqrt(365 if asset_type == "crypto" else 252) if not np.isnan(daily_vol) else np.nan
    if not np.isnan(realized_annual):
        used += 1
        if asset_type == "crypto":
            score += 5 if realized_annual < .55 else -8 if realized_annual > 1.25 else -3 if realized_annual > .85 else 0
        else:
            score += 5 if realized_annual < .28 else -8 if realized_annual > .75 else -3 if realized_annual > .50 else 0

    beta = safe_float(info.get("beta"))
    short = safe_float(info.get("shortPercentOfFloat"))
    if asset_type != "crypto" and not np.isnan(beta):
        score += 5 if beta < 1 else -5 if beta > 1.8 else 0; used += 1
    if asset_type != "crypto" and not np.isnan(short):
        score += 4 if short < .03 else -8 if short > .15 else -3 if short > .08 else 0; used += 1

    iv = safe_float(options.get("atm_iv"))
    pc = safe_float(options.get("put_call_oi"))
    if not np.isnan(iv):
        score += 5 if iv < .30 else -7 if iv > .70 else -3 if iv > .50 else 0; used += 1
    if not np.isnan(pc):
        score += 4 if .6 <= pc <= 1.1 else -5 if pc > 1.5 else 0; used += 1

    days_to_earn = np.nan if asset_type == "crypto" else next_earnings_days(earnings, calendar)
    if not np.isnan(days_to_earn):
        used += 1
        if days_to_earn <= 7:
            score -= 8
        elif days_to_earn <= 14:
            score -= 4
    return clamp(score), days_to_earn, realized_annual, used > 0


def weighted_overall(scores, base_weights):
    active = {k: v for k, v in scores.items() if v.get("available") and not np.isnan(safe_float(v.get("score")))}
    if not active:
        return 50.0, {}
    total_w = sum(base_weights.get(k, 0) for k in active)
    if total_w <= 0:
        total_w = len(active)
        effective = {k: 1 / len(active) for k in active}
    else:
        effective = {k: base_weights.get(k, 0) / total_w for k in active}
    overall = sum(active[k]["score"] * effective[k] for k in active)
    return clamp(overall), effective


@st.cache_data(ttl=3600, show_spinner=False)
def make_forecast_cached(model_hist, days, asset_type):
    feat = add_features(model_hist)
    cols = ["Close", "MA5", "MA10", "MA20", "MA50", "Volatility20", "RSI14", "Momentum5", "Momentum20", "Momentum60", "VolumeRatio20", "MACD", "MACDSignal"]
    feat["TargetReturn"] = feat["Close"].shift(-1) / feat["Close"] - 1
    train = feat.dropna(subset=cols + ["TargetReturn"]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(train) < 120:
        raise ValueError("Not enough clean historical data for a useful forecast model.")
    X, y = train[cols], train["TargetReturn"]

    def new_models():
        return [
            RandomForestRegressor(n_estimators=220, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1),
            GradientBoostingRegressor(n_estimators=140, max_depth=3, learning_rate=.035, loss="huber", random_state=42),
            make_pipeline(StandardScaler(), Ridge(alpha=12.0)),
        ]

    tscv = TimeSeriesSplit(n_splits=4)
    maes, dir_hits = [], []
    for tr, te in tscv.split(X):
        fold_preds = []
        for model in new_models():
            model.fit(X.iloc[tr], y.iloc[tr])
            fold_preds.append(model.predict(X.iloc[te]))
        p = np.mean(fold_preds, axis=0)
        actual = y.iloc[te].to_numpy()
        maes.append(mean_absolute_error(actual, p))
        dir_hits.append(np.mean(np.sign(actual) == np.sign(p)))

    models = new_models()
    for model in models:
        model.fit(X, y)

    work = model_hist.copy()
    preds = []
    model_disagreements = []
    last = work.index[-1]
    typical_vol = safe_float(work["Volume"].replace(0, np.nan).tail(30).median(), 0.0)
    max_daily = .20 if asset_type == "crypto" else .12
    for _ in range(days):
        tmp = add_features(work)
        row = tmp.iloc[-1]
        vals = row[cols].to_numpy(dtype=float).reshape(1, -1)
        indiv = np.array([float(m.predict(vals)[0]) for m in models])
        pred_ret = float(np.clip(np.median(indiv), -max_daily, max_daily))
        model_disagreements.append(float(np.std(indiv)))
        last_close = float(work["Close"].iloc[-1])
        pred = last_close * (1 + pred_ret)
        nxt = last + (pd.Timedelta(days=1) if asset_type == "crypto" else pd.tseries.offsets.BDay(1))
        synthetic = pd.DataFrame({"Open": [pred], "High": [pred], "Low": [pred], "Close": [pred], "Volume": [typical_vol]}, index=[nxt])
        work = pd.concat([work, synthetic])
        preds.append((nxt, pred))
        last = nxt
    pred_df = pd.DataFrame(preds, columns=["Date", "Predicted Close"]).set_index("Date")
    return pred_df, float(np.mean(maes)), float(np.mean(dir_hits)), float(np.mean(model_disagreements))


def data_quality_score(asset, snapshot, news_df, fund_used, analyst_available, context_available, options, model_hist, asset_type):
    quality = 100.0
    notes = []
    if "fallback" in asset["hist_source"].lower() or "stooq" in asset["hist_source"].lower():
        quality -= 8; notes.append(f"Price history is using {asset['hist_source']}.")
    if "fallback" in asset["model_source"].lower() or "stooq" in asset["model_source"].lower() or "partial" in asset["model_source"].lower():
        quality -= 8; notes.append(f"Model history is using {asset['model_source']}.")
    spread = safe_float(snapshot.get("price_spread"))
    if not np.isnan(spread):
        if spread > .05:
            quality -= 25; notes.append(f"Live price sources disagree by about {spread*100:.1f}%.")
        elif spread > .02:
            quality -= 12; notes.append(f"Live price sources differ by about {spread*100:.1f}%.")
    elif len(snapshot.get("price_sources", [])) < 2:
        quality -= 6; notes.append("Only one usable live-price source was available for cross-checking.")

    if model_hist.empty or len(model_hist) < 250:
        quality -= 15; notes.append("The forecast has limited historical training data.")
    try:
        last_date = pd.Timestamp(model_hist.index[-1]).tz_localize(None)
        age_days = (pd.Timestamp.utcnow().tz_localize(None) - last_date).total_seconds() / 86400
        stale_limit = 2.5 if asset_type == "crypto" else 6.0
        if age_days > stale_limit:
            quality -= 18; notes.append(f"Latest daily history appears {age_days:.0f} days old.")
    except Exception:
        pass

    if news_df.empty:
        quality -= 6; notes.append("No recent news feed was available; news is excluded from the score.")
    if asset_type != "crypto" and fund_used < 2:
        quality -= 8; notes.append("Fundamental coverage is thin; missing metrics are excluded.")
    if asset_type != "crypto" and not analyst_available:
        quality -= 5; notes.append("Analyst data is unavailable and excluded.")
    if not context_available:
        quality -= 8; notes.append("Market/macro context is incomplete.")
    if asset_type != "crypto" and np.isnan(safe_float(options.get("atm_iv"))):
        quality -= 3; notes.append("Options IV was unavailable; realized volatility is used as the risk fallback.")
    return clamp(quality), notes


if not run:
    st.info("Enter a stock ticker or crypto symbol above and tap **Analyze Everything**.")
    st.stop()
if not ticker_input:
    st.error("Enter a stock or crypto symbol.")
    st.stop()

symbol, guessed_type, coin_id, base_symbol = normalize_input(ticker_input)
with st.spinner(f"Cross-checking market data, news, fundamentals and macro conditions for {symbol}..."):
    try:
        asset = load_asset(symbol, period, guessed_type, coin_id)
        hist, model_hist = asset["hist"], asset["model_hist"]
        if hist.empty or model_hist.empty:
            raise ValueError("No usable price history was returned by the primary or backup sources.")
        info = asset["info"]
        asset_type = asset["asset_type"]
        company = info.get("longName") or info.get("shortName") or (base_symbol if asset_type == "crypto" else symbol)
        sector = info.get("sector") or ("Cryptocurrency" if asset_type == "crypto" else "Unknown")
        snapshot = resolve_snapshot(info, asset["fast"], hist, model_hist, asset_type, asset["cg"])
        spot = safe_float(snapshot["spot"])
        if np.isnan(spot) or spot <= 0:
            raise ValueError("No reliable current price could be resolved after fallback checks.")
        ctx, ctx_sources = load_context(sector, asset_type)
        options = load_options_snapshot(symbol, spot, asset_type)
        news_df = extract_news_rows(asset["news"], company, symbol, asset_type)
        fund, fund_sources = derive_fundamentals(info, asset["financials"]) if asset_type != "crypto" else ({}, {})
    except Exception as e:
        st.error(f"Could not analyze {symbol}: {e}")
        st.stop()

prev_close = first_valid(fast_get(asset["fast"], "previous_close", "previousClose"), info.get("previousClose"), safe_float(hist["Close"].iloc[-2]) if len(hist) > 1 else np.nan)
change_pct = (spot / prev_close - 1) * 100 if not np.isnan(prev_close) and prev_close else np.nan

asset_badge = "₿ CRYPTO" if asset_type == "crypto" else "📊 STOCK / EQUITY"
st.subheader(f"{company} ({symbol})")
st.caption(f"{asset_badge} · Price source: {snapshot['spot_source']} · History: {asset['hist_source']}")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Price", fmt_money(spot), f"{change_pct:+.2f}%" if not np.isnan(change_pct) else None)
m2.metric("Market Cap", fmt_big(snapshot["market_cap"]), help="Direct quote → fast quote → CoinGecko (crypto) → shares × price where possible.")
m3.metric(snapshot["pe_label"], f"{snapshot['pe']:.2f}" if not np.isnan(snapshot["pe"]) else ("N/A for crypto" if asset_type == "crypto" else "Data unavailable"), help=snapshot["pe_note"])
m4.metric("52W High", fmt_money(snapshot["year_high"]), help="Direct field → fast quote → calculated from recent history.")
m5.metric("52W Low", fmt_money(snapshot["year_low"]), help="Direct field → fast quote → calculated from recent history.")

if not np.isnan(safe_float(snapshot["price_spread"])) and snapshot["price_spread"] > .02:
    st.warning(f"Price cross-check warning: available live sources differ by about **{snapshot['price_spread']*100:.1f}%**. Forecast reliability is reduced automatically.")

st.markdown("### Price chart")
chart_df = add_features(hist)
fig = go.Figure()
fig.add_trace(go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"], name="Price"))
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MA20"], mode="lines", name="20-day MA"))
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MA50"], mode="lines", name="50-day MA"))
fig.update_layout(height=455, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.03, x=0))
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

tech, feat, tech_ok = technical_score(model_hist, asset_type)
news_s, news_count, news_sent, news_ok = news_score(news_df)
market, market_parts, market_used, market_ok = context_score(ctx, sector, asset_type, symbol)
risk, days_to_earn, realized_annual, risk_ok = risk_score(info, options, asset["earnings"], asset["calendar"], model_hist, asset_type)

if asset_type == "crypto":
    market_health, health_used, health_ok = crypto_market_score(model_hist, snapshot, asset["cg"])
    scores = {
        "Technical": {"score": tech, "available": tech_ok},
        "Crypto market / liquidity": {"score": market_health, "available": health_ok},
        "News": {"score": news_s, "available": news_ok},
        "Market / macro": {"score": market, "available": market_ok},
        "Risk / volatility": {"score": risk, "available": risk_ok},
    }
    base_weights = {"Technical": .34, "Crypto market / liquidity": .22, "News": .14, "Market / macro": .18, "Risk / volatility": .12}
    fund_used = 0
    analyst_available = False
    mean_target = np.nan
else:
    fund_score, fund_used, fund_ok = fundamental_score(fund)
    analyst, analyst_used, mean_target, analyst_available = analyst_score(asset["recs"], asset["raw_recs"], asset["targets"], info, spot)
    scores = {
        "Technical": {"score": tech, "available": tech_ok},
        "Fundamentals": {"score": fund_score, "available": fund_ok},
        "Analysts": {"score": analyst, "available": analyst_available},
        "News": {"score": news_s, "available": news_ok},
        "Market / macro": {"score": market, "available": market_ok},
        "Risk / positioning": {"score": risk, "available": risk_ok},
    }
    base_weights = {"Technical": .27, "Fundamentals": .22, "Analysts": .16, "News": .13, "Market / macro": .14, "Risk / positioning": .08}

overall, effective_weights = weighted_overall(scores, base_weights)
quality, quality_notes = data_quality_score(asset, snapshot, news_df, fund_used, analyst_available, market_ok, options, model_hist, asset_type)

st.markdown("## 🧠 Multi-factor score")
label = "Strongly Bullish" if overall >= 72 else "Bullish" if overall >= 60 else "Neutral" if overall >= 45 else "Bearish" if overall >= 32 else "Strongly Bearish"
a, b, c = st.columns(3)
a.metric("Overall score", f"{overall:.0f}/100", label)
b.metric("Data quality", f"{quality:.0f}/100", "Cross-check + coverage")
if asset_type == "crypto":
    c.metric("24h crypto volume", fmt_big(asset["cg"].get("usd_24h_vol")), "CoinGecko fallback" if asset["cg"] else "Yahoo/history fallback")
else:
    c.metric("Analyst mean target", fmt_money(mean_target), f"{((mean_target / spot - 1) * 100):+.1f}%" if not np.isnan(mean_target) and spot else "Excluded if unavailable")

score_rows = []
for name, data in scores.items():
    score_rows.append({
        "Factor": name,
        "Score": round(data["score"], 1) if data["available"] else "Excluded",
        "Effective weight": f"{effective_weights.get(name, 0) * 100:.0f}%" if data["available"] else "0%",
        "Status": "Included" if data["available"] else "Unavailable — not guessed",
    })
st.dataframe(pd.DataFrame(score_rows), hide_index=True, use_container_width=True)

left, right = st.columns(2)
if asset_type != "crypto":
    with left:
        st.markdown("### Company fundamentals")
        rows = {
            "Sector": sector, "Industry": info.get("industry", "Data unavailable"),
            "Revenue growth": fmt_pct(fund.get("revenueGrowth")), "Earnings growth": fmt_pct(fund.get("earningsGrowth")),
            "Profit margin": fmt_pct(fund.get("profitMargins"), signed=False), "ROE": fmt_pct(fund.get("returnOnEquity"), signed=False),
            "Debt / equity": f"{safe_float(fund.get('debtToEquity')):.1f}" if not np.isnan(safe_float(fund.get("debtToEquity"))) else "Data unavailable",
            "Free cash flow": fmt_big(fund.get("freeCashflow")), "Short % of float": fmt_pct(info.get("shortPercentOfFloat"), signed=False),
            "Institutional ownership": fmt_pct(info.get("heldPercentInstitutions"), signed=False), "Insider ownership": fmt_pct(info.get("heldPercentInsiders"), signed=False),
        }
        st.dataframe(pd.DataFrame(rows.items(), columns=["Metric", "Value"]), hide_index=True, use_container_width=True)
else:
    with left:
        st.markdown("### Crypto market health")
        close = model_hist["Close"]
        rows = {
            "Market cap": fmt_big(snapshot["market_cap"]),
            "24h volume": fmt_big(asset["cg"].get("usd_24h_vol")),
            "24h change": fmt_pct(safe_float(asset["cg"].get("usd_24h_change")) / 100 if not np.isnan(safe_float(asset["cg"].get("usd_24h_change"))) else np.nan),
            "30-day change": fmt_pct(close.iloc[-1] / close.iloc[-31] - 1) if len(close) > 31 else "Data unavailable",
            "90-day change": fmt_pct(close.iloc[-1] / close.iloc[-91] - 1) if len(close) > 91 else "Data unavailable",
            "Annualized realized volatility": fmt_pct(realized_annual, signed=False),
        }
        st.dataframe(pd.DataFrame(rows.items(), columns=["Metric", "Value"]), hide_index=True, use_container_width=True)

with right:
    st.markdown("### Market, macro & risk")
    vix_val = safe_float(market_parts.get("VIX"))
    rows = {
        "VIX": f"{vix_val:.1f}" if not np.isnan(vix_val) else "Data unavailable",
        "10Y yield 20-day move": fmt_pct(market_parts.get("TNX20")),
        "Dollar index 20-day move": fmt_pct(market_parts.get("DXY20")),
        "Annualized realized volatility": fmt_pct(realized_annual, signed=False),
    }
    if asset_type == "crypto":
        rows["Bitcoin 20-day"] = fmt_pct(market_parts.get("BTC20"))
        rows["Ethereum 20-day"] = fmt_pct(market_parts.get("ETH20"))
        rows["Nasdaq 20-day"] = fmt_pct(market_parts.get("QQQ20"))
    else:
        rows["S&P 500 20-day"] = fmt_pct(market_parts.get("SPY20"))
        rows["Nasdaq 20-day"] = fmt_pct(market_parts.get("QQQ20"))
        rows[f"{sector_etf(sector)} sector 20-day"] = fmt_pct(market_parts.get("Sector20"))
        rows["ATM implied volatility"] = fmt_pct(options.get("atm_iv"), signed=False) if not np.isnan(safe_float(options.get("atm_iv"))) else f"Realized-vol fallback: {fmt_pct(realized_annual, signed=False)}"
        rows["Put / call OI"] = f"{safe_float(options.get('put_call_oi')):.2f}" if not np.isnan(safe_float(options.get("put_call_oi"))) else "Data unavailable — excluded"
        rows["Days to next earnings"] = f"{days_to_earn:.0f}" if not np.isnan(days_to_earn) else "Data unavailable"
    st.dataframe(pd.DataFrame(rows.items(), columns=["Factor", "Value"]), hide_index=True, use_container_width=True)

st.markdown("### 📰 Recent news sentiment")
if news_df.empty:
    st.info("No recent headlines were returned by Yahoo Finance or the Google News RSS fallback. News is excluded rather than assumed neutral.")
else:
    show = news_df.head(12).copy()
    show["Tone"] = show["Sentiment"].apply(lambda x: "Positive" if x > .15 else "Negative" if x < -.15 else "Neutral")
    show["Age"] = show["AgeHours"].apply(lambda x: f"{x:.0f}h ago" if not pd.isna(x) else "Unknown")
    st.dataframe(show[["Title", "Source", "Tone", "Age", "Data source"]], hide_index=True, use_container_width=True)

with st.expander("🛡️ Fallback & verification status"):
    source_rows = [
        ["Chart history", asset["hist_source"], "Yahoo history → Yahoo download → CoinGecko (crypto) / Stooq (equity)"],
        ["Model history", asset["model_source"], "Yahoo history → Yahoo download → CoinGecko / Stooq"],
        ["Current price", snapshot["spot_source"], "Fast quote + info quote + CoinGecko for crypto + latest close"],
        ["52-week range", "Resolved" if not np.isnan(snapshot["year_high"]) else "Unavailable", "Direct field → fast quote → calculated history"],
        ["Fundamentals", "Included" if asset_type != "crypto" and fund_used >= 2 else "N/A or limited", "Quote fields → financial-statement calculations → exclude missing metrics"],
        ["News", "Available" if not news_df.empty else "Unavailable", "Yahoo Finance news → Google News RSS → exclude"],
        ["Sentiment", "VADER" if VADER_OK else "Keyword fallback", "VADER → built-in keyword sentiment"],
        ["Analysts", "Included" if analyst_available else "Unavailable / N/A", "Recommendation summary → raw grades + targetMeanPrice → exclude"],
        ["Options risk", options.get("source", "Unavailable"), "Option chain → realized-volatility risk proxy → exclude put/call if missing"],
        ["Macro context", f"{len(ctx)} feeds loaded", "Yahoo Ticker.history → Yahoo download; missing proxies excluded"],
    ]
    st.dataframe(pd.DataFrame(source_rows, columns=["Data area", "Current status", "Fallback plan"]), hide_index=True, use_container_width=True)
    if quality_notes:
        st.write("**Current quality notes:**")
        for note in quality_notes:
            st.write("• " + note)
    else:
        st.success("No major data-quality warnings detected in the current scan.")

st.divider()
st.markdown("## 🔮 Ensemble forecast")
horizon_label = "calendar days" if asset_type == "crypto" else "trading days"
with st.spinner("Backtesting three models and blending the live multi-factor signals..."):
    try:
        pred_df, return_mae, dir_acc, model_disagreement = make_forecast_cached(model_hist, forecast_days, asset_type)
        pure_end = float(pred_df["Predicted Close"].iloc[-1])
        pure_ret = pure_end / spot - 1
        factor_tilt = (overall - 50) / 50
        overlay_scale = .014 if asset_type == "crypto" else .012
        overlay_cap = .12 if asset_type == "crypto" else .08
        overlay = np.clip(factor_tilt * (overlay_scale * math.sqrt(max(forecast_days, 1))), -overlay_cap, overlay_cap)
        blended_ret = np.clip(pure_ret + overlay, -.60 if asset_type == "crypto" else -.45, .60 if asset_type == "crypto" else .45)
        ending = spot * (1 + blended_ret)

        daily_vol = safe_float(model_hist["Close"].pct_change().tail(60).std())
        base_uncertainty = spot * daily_vol * math.sqrt(forecast_days) if not np.isnan(daily_vol) else spot * (.14 if asset_type == "crypto" else .08)
        residual_uncertainty = spot * return_mae * math.sqrt(max(1, forecast_days))
        uncertainty = max(base_uncertainty, residual_uncertainty)
        low = max(0, ending - 1.35 * uncertainty)
        high = ending + 1.35 * uncertainty

        model_quality = clamp(100 - (return_mae * 100) * (8 if asset_type == "crypto" else 12))
        included_scores = [d["score"] for d in scores.values() if d["available"]]
        agreement = 100 - min(100, np.std(included_scores) * 3) if len(included_scores) > 1 else 55
        disagreement_penalty = min(22, model_disagreement * 100 * 65)
        vol_penalty = min(32, (daily_vol * 100) * (5 if asset_type == "crypto" else 10)) if not np.isnan(daily_vol) else 18
        horizon_penalty = max(0, (forecast_days - 10) * (.28 if asset_type == "crypto" else .35))
        confidence = .34 * model_quality + .26 * (dir_acc * 100) + .16 * agreement + .24 * quality - disagreement_penalty - vol_penalty - horizon_penalty
        # Deliberately cap this: it is a reliability score, never a probability or guarantee.
        confidence = clamp(confidence, 5, 85 if asset_type == "crypto" else 90)

        p1, p2, p3 = st.columns(3)
        p1.metric(f"Best guess in {forecast_days} {horizon_label}", fmt_money(ending), f"{blended_ret * 100:+.2f}%")
        p2.metric("Estimated uncertainty range", f"{fmt_money(low)} – {fmt_money(high)}")
        p3.metric("Forecast reliability", f"{confidence:.0f}/100", f"{dir_acc * 100:.0f}% walk-forward direction")

        ffig = go.Figure()
        recent = model_hist.tail(140)
        ffig.add_trace(go.Scatter(x=recent.index, y=recent["Close"], mode="lines", name="Historical close"))
        path = pred_df["Predicted Close"].copy()
        if len(path) and path.iloc[-1] != 0:
            path = path * (ending / path.iloc[-1])
        ffig.add_trace(go.Scatter(x=path.index, y=path, mode="lines+markers", name="Ensemble forecast"))
        ffig.update_layout(height=390, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.03, x=0))
        st.plotly_chart(ffig, use_container_width=True, config={"displayModeBar": False})

        stance = "strongly bullish" if blended_ret > .10 else "bullish" if blended_ret > .03 else "strongly bearish" if blended_ret < -.10 else "bearish" if blended_ret < -.03 else "neutral / sideways"
        st.success(f"Current combined outlook: **{stance.upper()}** · Multi-factor score **{overall:.0f}/100** · Data quality **{quality:.0f}/100**.")

        with st.expander("How this prediction is built"):
            st.write(
                "The historical forecast is an ensemble of Random Forest, Gradient Boosting and regularized linear regression. "
                "It is evaluated with walk-forward time-series backtests, then blended with whichever live factors are actually available. "
                "For stocks, those can include technicals, financial-statement fundamentals, analysts, news, broad market/sector trends, rates, dollar, VIX, short interest, earnings timing and options. "
                "For crypto, the app substitutes crypto-specific market-cap/liquidity, Bitcoin/Ethereum regime, 24/7 volatility and crypto news. Missing factors receive zero weight rather than a fake neutral score."
            )
        st.warning(
            "No prediction can be 100%—or 110%—correct. This reliability score is NOT the probability that the target price will occur. "
            "Unexpected news, regulation, earnings, hacks, liquidations, wars, Fed decisions, market crashes and data-provider errors can invalidate any forecast. "
            "Use this as research, not as a guaranteed trading signal or financial advice."
        )
    except Exception as e:
        st.error(f"Forecast unavailable: {e}")

with st.expander("About this asset"):
    if asset_type == "crypto":
        st.write(f"{base_symbol} is being analyzed as a cryptocurrency. Equity-only fields such as P/E, analyst targets and corporate fundamentals are intentionally marked N/A rather than fabricated.")
    else:
        st.write(info.get("longBusinessSummary") or "No company description was returned by the available data source.")
