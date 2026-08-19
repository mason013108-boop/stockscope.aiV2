import math
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_OK = True
except Exception:
    VADER_OK = False

st.set_page_config(
    page_title="StockScope AI Pro",
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

st.title("📈 StockScope AI Pro")
st.caption("Multi-factor stock analysis: price action, fundamentals, news, analysts, market regime, macro, volatility and more.")

with st.form("stock_lookup", border=False):
    q1, q2, q3 = st.columns([1.35, 1, 1])
    with q1:
        ticker_input = st.text_input("Ticker symbol", value="AAPL", placeholder="AAPL, NVDA, TSLA...").upper().strip()
    with q2:
        period = st.selectbox("Chart history", ["3mo", "6mo", "1y", "2y", "5y", "10y"], index=2)
    with q3:
        forecast_days = st.select_slider("Forecast days", options=[5,10,15,20,30,45,60], value=20)
    run = st.form_submit_button("Analyze Everything", type="primary", use_container_width=True)

st.caption("Examples: AAPL · MSFT · NVDA · TSLA · AMZN · GOOGL · META")


def safe_float(x, default=np.nan):
    try:
        if x is None or pd.isna(x): return default
        return float(x)
    except Exception:
        return default


def fmt_money(x):
    x = safe_float(x)
    return "—" if np.isnan(x) else f"${x:,.2f}"


def fmt_pct(x, scale=100):
    x = safe_float(x)
    return "—" if np.isnan(x) else f"{x*scale:+.2f}%"


def fmt_big(x):
    x = safe_float(x)
    if np.isnan(x): return "—"
    for unit, scale in [("T",1e12),("B",1e9),("M",1e6),("K",1e3)]:
        if abs(x) >= scale: return f"{x/scale:.2f}{unit}"
    return f"{x:,.0f}"


def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, float(x)))


def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = -delta.clip(upper=0).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100/(1+rs))


def add_features(df):
    x = df.copy()
    x["Return_1d"] = x["Close"].pct_change()
    for n in [5,10,20,50,100,200]: x[f"MA{n}"] = x["Close"].rolling(n).mean()
    x["Volatility20"] = x["Return_1d"].rolling(20).std()
    x["RSI14"] = rsi(x["Close"],14)
    x["Momentum5"] = x["Close"].pct_change(5)
    x["Momentum20"] = x["Close"].pct_change(20)
    x["Momentum60"] = x["Close"].pct_change(60)
    x["VolumeChange"] = x["Volume"].pct_change()
    x["VolumeRatio20"] = x["Volume"] / x["Volume"].rolling(20).mean()
    # MACD
    ema12 = x["Close"].ewm(span=12, adjust=False).mean()
    ema26 = x["Close"].ewm(span=26, adjust=False).mean()
    x["MACD"] = ema12 - ema26
    x["MACDSignal"] = x["MACD"].ewm(span=9, adjust=False).mean()
    return x


def sector_etf(sector):
    return {
        "Technology":"XLK","Financial Services":"XLF","Healthcare":"XLV","Consumer Cyclical":"XLY",
        "Consumer Defensive":"XLP","Industrials":"XLI","Energy":"XLE","Basic Materials":"XLB",
        "Real Estate":"XLRE","Utilities":"XLU","Communication Services":"XLC"
    }.get(sector, "SPY")


@st.cache_data(ttl=900, show_spinner=False)
def load_stock(ticker_symbol, chart_period):
    t = yf.Ticker(ticker_symbol)
    hist = t.history(period=chart_period, auto_adjust=False)
    model_hist = t.history(period="5y", auto_adjust=False)
    try: info = t.info or {}
    except Exception: info = {}
    try: fast = dict(t.fast_info)
    except Exception: fast = {}
    try: news = t.news or []
    except Exception: news = []
    try: recs = t.recommendations_summary
    except Exception: recs = pd.DataFrame()
    try: targets = t.analyst_price_targets or {}
    except Exception: targets = {}
    try: earnings = t.get_earnings_dates(limit=8)
    except Exception: earnings = pd.DataFrame()
    return hist, model_hist, info, fast, news, recs, targets, earnings


@st.cache_data(ttl=900, show_spinner=False)
def load_context(sector):
    symbols = ["SPY","QQQ",sector_etf(sector),"^VIX","^TNX","DX-Y.NYB","CL=F","GC=F"]
    out = {}
    for s in dict.fromkeys(symbols):
        try:
            h = yf.Ticker(s).history(period="6mo", auto_adjust=True)
            if not h.empty: out[s] = h
        except Exception:
            pass
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def load_options_snapshot(ticker_symbol, spot):
    t = yf.Ticker(ticker_symbol)
    result = {"atm_iv": np.nan, "put_call_oi": np.nan, "expiration": None}
    try:
        exps = t.options
        if not exps: return result
        exp = exps[0]
        chain = t.option_chain(exp)
        calls, puts = chain.calls.copy(), chain.puts.copy()
        if calls.empty or puts.empty: return result
        calls["dist"] = (calls["strike"]-spot).abs()
        puts["dist"] = (puts["strike"]-spot).abs()
        c = calls.nsmallest(3,"dist")["impliedVolatility"].replace([np.inf,-np.inf],np.nan).mean()
        p = puts.nsmallest(3,"dist")["impliedVolatility"].replace([np.inf,-np.inf],np.nan).mean()
        result["atm_iv"] = np.nanmean([c,p])
        call_oi = calls["openInterest"].fillna(0).sum()
        put_oi = puts["openInterest"].fillna(0).sum()
        result["put_call_oi"] = (put_oi/call_oi) if call_oi > 0 else np.nan
        result["expiration"] = exp
    except Exception:
        pass
    return result


def extract_news_rows(news):
    rows=[]
    analyzer = SentimentIntensityAnalyzer() if VADER_OK else None
    now = datetime.now(timezone.utc)
    for item in news[:20]:
        content = item.get("content", item) if isinstance(item,dict) else {}
        title = content.get("title") or item.get("title") if isinstance(item,dict) else None
        if not title: continue
        provider = ""
        p = content.get("provider")
        if isinstance(p,dict): provider = p.get("displayName","")
        elif isinstance(p,str): provider = p
        ts = content.get("pubDate") or content.get("providerPublishTime") or item.get("providerPublishTime")
        age_hours = np.nan
        try:
            if isinstance(ts,(int,float)):
                dt = datetime.fromtimestamp(ts, timezone.utc)
            else:
                dt = pd.to_datetime(ts, utc=True).to_pydatetime()
            age_hours = max(0,(now-dt).total_seconds()/3600)
        except Exception: pass
        sentiment = analyzer.polarity_scores(title)["compound"] if analyzer else 0.0
        rows.append({"Title":title,"Source":provider or "News","AgeHours":age_hours,"Sentiment":sentiment})
    return pd.DataFrame(rows)


def technical_score(df):
    f = add_features(df).iloc[-1]
    p = safe_float(f["Close"]); score=50; reasons=[]
    for n,w in [(20,7),(50,8),(100,5),(200,8)]:
        ma=safe_float(f.get(f"MA{n}"))
        if not np.isnan(ma):
            score += w if p>ma else -w
    r=safe_float(f["RSI14"])
    if not np.isnan(r):
        if 50<=r<=68: score += 7
        elif 32<=r<50: score -= 3
        elif r>75: score -= 8
        elif r<28: score += 5
    m20=safe_float(f["Momentum20"]); m60=safe_float(f["Momentum60"])
    if not np.isnan(m20): score += np.clip(m20*100, -10, 10)
    if not np.isnan(m60): score += np.clip(m60*50, -8, 8)
    macd=safe_float(f["MACD"]); sig=safe_float(f["MACDSignal"])
    if not np.isnan(macd) and not np.isnan(sig): score += 6 if macd>sig else -6
    vr=safe_float(f["VolumeRatio20"])
    if not np.isnan(vr) and vr>1.3 and not np.isnan(m20): score += 4 if m20>0 else -4
    return clamp(score), f


def fundamental_score(info):
    score=50; used=0
    def add(val):
        nonlocal score, used
        score += val; used += 1
    rg=safe_float(info.get("revenueGrowth")); eg=safe_float(info.get("earningsGrowth"));
    pm=safe_float(info.get("profitMargins")); roe=safe_float(info.get("returnOnEquity"));
    de=safe_float(info.get("debtToEquity")); fcf=safe_float(info.get("freeCashflow"));
    cr=safe_float(info.get("currentRatio")); pe=safe_float(info.get("forwardPE"));
    peg=safe_float(info.get("pegRatio"));
    if not np.isnan(rg): add(np.clip(rg*35,-12,14))
    if not np.isnan(eg): add(np.clip(eg*25,-12,14))
    if not np.isnan(pm): add(np.clip((pm-.08)*35,-8,10))
    if not np.isnan(roe): add(np.clip((roe-.12)*24,-8,10))
    if not np.isnan(de): add(7 if de<60 else 2 if de<120 else -7 if de>250 else -2)
    if not np.isnan(fcf): add(5 if fcf>0 else -8)
    if not np.isnan(cr): add(4 if cr>1.5 else -4 if cr<0.8 else 0)
    if not np.isnan(pe): add(4 if 8<pe<30 else -4 if pe>60 or pe<0 else 0)
    if not np.isnan(peg): add(5 if 0<peg<1.5 else -4 if peg>3 else 0)
    return clamp(score), used


def analyst_score(recs, targets, spot):
    score=50; coverage=0; notes=[]
    if isinstance(recs,pd.DataFrame) and not recs.empty:
        row=recs.iloc[0]
        strong_buy=safe_float(row.get("strongBuy"),0); buy=safe_float(row.get("buy"),0)
        hold=safe_float(row.get("hold"),0); sell=safe_float(row.get("sell"),0); strong_sell=safe_float(row.get("strongSell"),0)
        total=strong_buy+buy+hold+sell+strong_sell
        if total>0:
            net=((strong_buy*2+buy)-(sell+strong_sell*2))/(2*total)
            score += net*32; coverage += 1
    mean_t=safe_float(targets.get("mean")) if isinstance(targets,dict) else np.nan
    if not np.isnan(mean_t) and spot>0:
        upside=mean_t/spot-1
        score += np.clip(upside*80,-20,20); coverage += 1
        notes.append(upside)
    return clamp(score), coverage, mean_t


def news_score(news_df):
    if news_df.empty: return 50,0,np.nan
    df=news_df.copy()
    age=df["AgeHours"].fillna(72)
    weights=np.exp(-age/72).clip(lower=.08)
    weighted=(df["Sentiment"]*weights).sum()/weights.sum()
    score=clamp(50+weighted*42)
    return score,len(df),weighted


def context_score(ctx, sector):
    score=50; pieces=[]
    def ret(sym,n):
        h=ctx.get(sym)
        if h is None or len(h)<n+1:return np.nan
        return safe_float(h["Close"].iloc[-1]/h["Close"].iloc[-n-1]-1)
    spy20=ret("SPY",20); qqq20=ret("QQQ",20); sec=sector_etf(sector); sec20=ret(sec,20)
    for r,w in [(spy20,80),(qqq20,40),(sec20,70)]:
        if not np.isnan(r): score += np.clip(r*w,-8,8); pieces.append(r)
    vix=ctx.get("^VIX")
    if vix is not None and not vix.empty:
        vv=safe_float(vix["Close"].iloc[-1]); score += 6 if vv<17 else -6 if vv>25 else -12 if vv>35 else 0
    tnx20=ret("^TNX",20)
    if not np.isnan(tnx20): score += np.clip(-tnx20*25,-5,5)
    return clamp(score), {"SPY20":spy20,"QQQ20":qqq20,"Sector20":sec20}


def risk_score(info, options, earnings):
    # Higher = safer / more favorable risk backdrop
    score=55
    beta=safe_float(info.get("beta")); short=safe_float(info.get("shortPercentOfFloat"));
    if not np.isnan(beta): score += 5 if beta<1 else -5 if beta>1.8 else 0
    if not np.isnan(short): score += 4 if short<.03 else -8 if short>.15 else -3 if short>.08 else 0
    iv=safe_float(options.get("atm_iv")); pc=safe_float(options.get("put_call_oi"))
    if not np.isnan(iv): score += 5 if iv<.3 else -7 if iv>.7 else -3 if iv>.5 else 0
    if not np.isnan(pc): score += 4 if .6<=pc<=1.1 else -5 if pc>1.5 else 0
    # earnings very close adds event risk
    days_to_earn=np.nan
    try:
        if isinstance(earnings,pd.DataFrame) and not earnings.empty:
            idx=pd.to_datetime(earnings.index, utc=True)
            future=idx[idx>pd.Timestamp.now(tz="UTC")]
            if len(future):
                days_to_earn=(future.min()-pd.Timestamp.now(tz="UTC")).total_seconds()/86400
                if days_to_earn<=7: score -= 8
                elif days_to_earn<=14: score -= 4
    except Exception: pass
    return clamp(score), days_to_earn


@st.cache_data(ttl=3600, show_spinner=False)
def make_forecast_cached(model_hist, days):
    feat=add_features(model_hist)
    cols=["Close","MA5","MA10","MA20","MA50","Volatility20","RSI14","Momentum5","Momentum20","Momentum60","VolumeChange","VolumeRatio20","MACD","MACDSignal"]
    feat["TargetReturn"]=feat["Close"].shift(-1)/feat["Close"]-1
    train=feat.dropna(subset=cols+["TargetReturn"]).copy()
    if len(train)<120: raise ValueError("Not enough historical data for a useful model.")
    X=train[cols]; y=train["TargetReturn"]
    tscv=TimeSeriesSplit(n_splits=4); maes=[]; dir_hits=[]
    for tr,te in tscv.split(X):
        m=RandomForestRegressor(n_estimators=180,max_depth=8,min_samples_leaf=3,random_state=42,n_jobs=-1)
        m.fit(X.iloc[tr],y.iloc[tr]); p=m.predict(X.iloc[te])
        actual=y.iloc[te].to_numpy(); maes.append(mean_absolute_error(actual,p)); dir_hits.append(np.mean(np.sign(actual)==np.sign(p)))
    model=RandomForestRegressor(n_estimators=320,max_depth=9,min_samples_leaf=3,random_state=42,n_jobs=-1)
    model.fit(X,y)
    work=model_hist.copy(); preds=[]; last=work.index[-1]; vol=float(work["Volume"].tail(20).median())
    for _ in range(days):
        tmp=add_features(work); row=tmp.iloc[-1]; vals=row[cols].to_numpy(dtype=float).reshape(1,-1)
        pred_ret=float(np.clip(model.predict(vals)[0],-.12,.12)); last_close=float(work["Close"].iloc[-1]); pred=last_close*(1+pred_ret)
        nxt=last+pd.tseries.offsets.BDay(1)
        synthetic=pd.DataFrame({"Open":[pred],"High":[pred],"Low":[pred],"Close":[pred],"Volume":[vol]},index=[nxt])
        work=pd.concat([work,synthetic]); preds.append((nxt,pred)); last=nxt
    pred_df=pd.DataFrame(preds,columns=["Date","Predicted Close"]).set_index("Date")
    return pred_df,float(np.mean(maes)),float(np.mean(dir_hits))


if not run:
    st.info("Enter a ticker above and tap **Analyze Everything**.")
    st.stop()
if not ticker_input:
    st.error("Enter a ticker symbol."); st.stop()

with st.spinner(f"Scanning price, fundamentals, news, analysts and market conditions for {ticker_input}..."):
    try:
        hist,model_hist,info,fast,news,recs,targets,earnings=load_stock(ticker_input,period)
        if hist.empty or model_hist.empty: raise ValueError("No market data found.")
        company=info.get("longName") or info.get("shortName") or ticker_input
        sector=info.get("sector") or "Unknown"
        ctx=load_context(sector)
        spot=float(hist["Close"].iloc[-1])
        options=load_options_snapshot(ticker_input,spot)
        news_df=extract_news_rows(news)
    except Exception as e:
        st.error(f"Could not analyze {ticker_input}: {e}"); st.stop()

prev=float(hist["Close"].iloc[-2]) if len(hist)>1 else spot
change_pct=(spot/prev-1)*100 if prev else 0
st.subheader(f"{company} ({ticker_input})")

m1,m2,m3,m4,m5=st.columns(5)
m1.metric("Price",fmt_money(spot),f"{change_pct:+.2f}%")
m2.metric("Market Cap",fmt_big(info.get("marketCap")))
m3.metric("Forward P/E",f"{safe_float(info.get('forwardPE')):.2f}" if not np.isnan(safe_float(info.get('forwardPE'))) else "—")
m4.metric("52W High",fmt_money(info.get("fiftyTwoWeekHigh")))
m5.metric("52W Low",fmt_money(info.get("fiftyTwoWeekLow")))

st.markdown("### Price chart")
chart_df=add_features(hist)
fig=go.Figure()
fig.add_trace(go.Candlestick(x=hist.index,open=hist["Open"],high=hist["High"],low=hist["Low"],close=hist["Close"],name="Price"))
fig.add_trace(go.Scatter(x=chart_df.index,y=chart_df["MA20"],mode="lines",name="20-day MA"))
fig.add_trace(go.Scatter(x=chart_df.index,y=chart_df["MA50"],mode="lines",name="50-day MA"))
fig.update_layout(height=455,xaxis_rangeslider_visible=False,margin=dict(l=0,r=0,t=10,b=0),legend=dict(orientation="h",y=1.03,x=0))
st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

# Scores
tech,feat=technical_score(model_hist)
fund,fund_used=fundamental_score(info)
analyst,analyst_used,mean_target=analyst_score(recs,targets,spot)
news_s,news_count,news_sent=news_score(news_df)
market,market_parts=context_score(ctx,sector)
risk,days_to_earn=risk_score(info,options,earnings)

# Weighted composite. Technical + fundamentals are largest; news is intentionally capped.
weights={"Technical":.27,"Fundamentals":.22,"Analysts":.16,"News":.13,"Market / macro":.14,"Risk / positioning":.08}
scores={"Technical":tech,"Fundamentals":fund,"Analysts":analyst,"News":news_s,"Market / macro":market,"Risk / positioning":risk}
overall=sum(scores[k]*weights[k] for k in scores)

st.markdown("## 🧠 Everything score")
a,b,c=st.columns([1,1,1])
label="Strongly Bullish" if overall>=72 else "Bullish" if overall>=60 else "Neutral" if overall>=45 else "Bearish" if overall>=32 else "Strongly Bearish"
a.metric("Overall score",f"{overall:.0f}/100",label)
b.metric("News sentiment",f"{news_s:.0f}/100",f"{news_count} recent stories")
c.metric("Analyst mean target",fmt_money(mean_target),f"{((mean_target/spot-1)*100):+.1f}%" if not np.isnan(mean_target) and spot else "—")

score_df=pd.DataFrame({"Factor":list(scores.keys()),"Score":[round(scores[k],1) for k in scores],"Weight":[f"{weights[k]*100:.0f}%" for k in scores]})
st.dataframe(score_df,hide_index=True,use_container_width=True)

left,right=st.columns(2)
with left:
    st.markdown("### Company fundamentals")
    rows={
        "Sector":sector,"Industry":info.get("industry","—"),"Revenue growth":fmt_pct(info.get("revenueGrowth")),
        "Earnings growth":fmt_pct(info.get("earningsGrowth")),"Profit margin":fmt_pct(info.get("profitMargins")),
        "ROE":fmt_pct(info.get("returnOnEquity")),"Debt / equity":f"{safe_float(info.get('debtToEquity')):.1f}" if not np.isnan(safe_float(info.get('debtToEquity'))) else "—",
        "Free cash flow":fmt_big(info.get("freeCashflow")),"Short % of float":fmt_pct(info.get("shortPercentOfFloat")),
        "Institutional ownership":fmt_pct(info.get("heldPercentInstitutions")),"Insider ownership":fmt_pct(info.get("heldPercentInsiders")),
    }
    st.dataframe(pd.DataFrame(rows.items(),columns=["Metric","Value"]),hide_index=True,use_container_width=True)
with right:
    st.markdown("### Market & positioning")
    vix=ctx.get("^VIX"); vix_val=safe_float(vix["Close"].iloc[-1]) if vix is not None and not vix.empty else np.nan
    tnx=ctx.get("^TNX"); tnx_val=safe_float(tnx["Close"].iloc[-1]) if tnx is not None and not tnx.empty else np.nan
    rows={
        "S&P 500 20-day":fmt_pct(market_parts.get("SPY20")),"Nasdaq 20-day":fmt_pct(market_parts.get("QQQ20")),
        f"{sector_etf(sector)} sector 20-day":fmt_pct(market_parts.get("Sector20")),"VIX":f"{vix_val:.1f}" if not np.isnan(vix_val) else "—",
        "10Y yield proxy":f"{tnx_val:.2f}" if not np.isnan(tnx_val) else "—","ATM implied volatility":fmt_pct(options.get("atm_iv")),
        "Put / call open interest":f"{safe_float(options.get('put_call_oi')):.2f}" if not np.isnan(safe_float(options.get('put_call_oi'))) else "—",
        "Days to next earnings":f"{days_to_earn:.0f}" if not np.isnan(days_to_earn) else "—",
    }
    st.dataframe(pd.DataFrame(rows.items(),columns=["Factor","Value"]),hide_index=True,use_container_width=True)

st.markdown("### 📰 Recent news sentiment")
if news_df.empty:
    st.info("No recent Yahoo Finance news items were available for this ticker.")
else:
    show=news_df.head(10).copy()
    show["Tone"]=show["Sentiment"].apply(lambda x:"Positive" if x>.15 else "Negative" if x<-.15 else "Neutral")
    show["Age"]=show["AgeHours"].apply(lambda x:f"{x:.0f}h ago" if not pd.isna(x) else "—")
    st.dataframe(show[["Title","Source","Tone","Age"]],hide_index=True,use_container_width=True)

st.divider(); st.markdown("## 🔮 Multi-factor forecast")
with st.spinner("Building machine-learning forecast and blending outside factors..."):
    try:
        pred_df,return_mae,dir_acc=make_forecast_cached(model_hist,forecast_days)
        pure_end=float(pred_df["Predicted Close"].iloc[-1]); pure_ret=pure_end/spot-1

        # The external-factor overlay is deliberately bounded so headlines/analysts cannot overwhelm price history.
        factor_tilt=(overall-50)/50
        overlay=np.clip(factor_tilt * (0.012*math.sqrt(max(forecast_days,1))),-.08,.08)
        blended_ret=np.clip(pure_ret+overlay,-.45,.45)
        ending=spot*(1+blended_ret)

        daily_vol=safe_float(model_hist["Close"].pct_change().tail(60).std())
        uncertainty=spot*daily_vol*math.sqrt(forecast_days) if not np.isnan(daily_vol) else spot*.08
        low=max(0,ending-1.28*uncertainty); high=ending+1.28*uncertainty

        # Confidence considers validation, agreement, data coverage, volatility, and horizon.
        model_quality=clamp(100-(return_mae*100)*12)
        agreement=100-min(100,np.std(list(scores.values()))*3)
        coverage=np.mean([1, fund_used>=5, analyst_used>=1, news_count>=3, len(ctx)>=4, not np.isnan(safe_float(options.get('atm_iv')))])*100
        vol_penalty=min(35,(daily_vol*100)*10) if not np.isnan(daily_vol) else 15
        horizon_penalty=max(0,(forecast_days-10)*.35)
        confidence=clamp(.40*model_quality+.22*(dir_acc*100)+.18*agreement+.20*coverage-vol_penalty-horizon_penalty)

        p1,p2,p3=st.columns(3)
        p1.metric(f"Best-guess price in {forecast_days} trading days",fmt_money(ending),f"{blended_ret*100:+.2f}%")
        p2.metric("Estimated range",f"{fmt_money(low)} – {fmt_money(high)}")
        p3.metric("Model confidence",f"{confidence:.0f}/100",f"{dir_acc*100:.0f}% backtest direction")

        ffig=go.Figure(); recent=model_hist.tail(120)
        ffig.add_trace(go.Scatter(x=recent.index,y=recent["Close"],mode="lines",name="Historical close"))
        # scale pure model path to finish at blended end, preserving shape
        path=pred_df["Predicted Close"].copy()
        if len(path) and path.iloc[-1] != 0: path=path*(ending/path.iloc[-1])
        ffig.add_trace(go.Scatter(x=path.index,y=path,mode="lines+markers",name="Multi-factor forecast"))
        ffig.update_layout(height=390,margin=dict(l=0,r=0,t=10,b=0),legend=dict(orientation="h",y=1.03,x=0))
        st.plotly_chart(ffig,use_container_width=True,config={"displayModeBar":False})

        stance="strongly bullish" if blended_ret>.10 else "bullish" if blended_ret>.03 else "strongly bearish" if blended_ret<-.10 else "bearish" if blended_ret<-.03 else "neutral / sideways"
        st.success(f"Current combined outlook: **{stance.upper()}**. The historical ML model is being adjusted by the live multi-factor score of **{overall:.0f}/100**.")

        with st.expander("What is included in this forecast?", expanded=False):
            st.write(
                "The app combines: historical price/volume behavior, moving averages, RSI, MACD, momentum, volatility, company growth/profitability/cash flow/debt/valuation, "
                "recent news-title sentiment, analyst recommendations and price targets, S&P 500/Nasdaq/sector trends, VIX, Treasury-yield proxy, dollar/oil/gold context, "
                "short interest, institutional/insider ownership, options implied volatility and put/call positioning when available, plus proximity to earnings. "
                "Some fields may be unavailable for certain tickers, and the app reduces confidence when coverage is weaker."
            )
        st.warning(
            "This is still an estimate, not a guarantee or financial advice. No model can see future headlines, surprise earnings, Fed decisions, lawsuits, wars, product failures, fraud, market crashes, or other events before they happen. "
            "News sentiment is based mainly on available headline text, and Yahoo Finance data availability varies by ticker."
        )
    except Exception as e:
        st.error(f"Forecast unavailable: {e}")

with st.expander("About the company"):
    st.write(info.get("longBusinessSummary") or "No company description available.")
