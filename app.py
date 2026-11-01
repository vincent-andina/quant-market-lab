
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

st.set_page_config(
    page_title="Quant Market Lab",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Quant Market Lab")
st.caption("US Equity Research | Market Analysis | Factor Modelling")

# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.header("Research Settings")

stocks = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA"]

selected_stock = st.sidebar.selectbox(
    "Select Stock",
    stocks
)

start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.Timestamp("2020-01-01")
)

end_date = st.sidebar.date_input(
    "End Date",
    value=pd.Timestamp.today().date()
)

window = st.sidebar.slider(
    "Rolling Window (Trading Days)",
    min_value=20,
    max_value=252,
    value=60
)

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# -------------------------------
# DATA COLLECTION
# -------------------------------

@st.cache_data(ttl=3600)
def load_data(start, end):
    tickers = [
        "NVDA", "AAPL", "MSFT", "AMZN",
        "GOOGL", "META", "TSLA",
        "SPY", "QQQ", "SOXX"
    ]

    raw = yf.download(
        tickers,
        start=start,
        end=end + pd.Timedelta(days=1),
        auto_adjust=True,
        progress=False,
        threads=False
    )

    prices = raw["Close"].copy()
    return prices.dropna(how="all").ffill(limit=3)


with st.spinner("Downloading market data..."):
    try:
        prices = load_data(
            pd.Timestamp(start_date),
            pd.Timestamp(end_date)
        )
    except Exception as e:
        st.error(f"Unable to download market data: {e}")
        st.stop()

required = [selected_stock, "SPY", "QQQ", "SOXX"]

if prices.empty or any(c not in prices.columns for c in required):
    st.error("Market data unavailable for the selected period.")
    st.stop()

returns = prices.pct_change(fill_method=None)

stock_returns = returns[selected_stock].dropna()

if len(stock_returns) < max(window, 30):
    st.warning("Select a longer period for reliable analysis.")
    st.stop()

# -------------------------------
# NAVIGATION
# -------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Market Overview",
    "Correlation Analysis",
    "Factor Regression",
    "Rolling Risk Analysis"
])

# -------------------------------
# TAB 1: MARKET OVERVIEW
# -------------------------------

with tab1:
    st.header("Market Overview")

    stock_prices = prices[selected_stock].dropna()

    latest_price = stock_prices.iloc[-1]
    latest_return = stock_returns.iloc[-1] * 100
    annual_vol = stock_returns.std() * np.sqrt(252) * 100
    total_return = (
        stock_prices.iloc[-1] / stock_prices.iloc[0] - 1
    ) * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Latest Price", f"${latest_price:,.2f}")
    c2.metric("Latest Daily Return", f"{latest_return:+.2f}%")
    c3.metric("Annualised Volatility", f"{annual_vol:.2f}%")
    c4.metric("Period Return", f"{total_return:+.2f}%")

    st.subheader("Historical Adjusted Price")

    fig = px.line(
        stock_prices,
        title=f"{selected_stock} Adjusted Closing Price"
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily Return Distribution")

    fig = px.histogram(
        stock_returns * 100,
        nbins=60,
        title="Distribution of Daily Returns"
    )
    fig.update_layout(
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Normalised Price Comparison")

    comparison = prices[required].dropna()

    if not comparison.empty:
        normalised = comparison.div(comparison.iloc[0]) * 100

        fig = px.line(
            normalised,
            title="Growth of $100 (Adjusted Prices)"
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Indexed Value"
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 2: CORRELATION
# -------------------------------

with tab2:
    st.header("Correlation Analysis")

    correlation = returns[required].dropna().corr()

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        title="Daily Return Correlation Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rolling Correlation with S&P 500")

    rolling_corr = (
        returns[selected_stock]
        .rolling(window)
        .corr(returns["SPY"])
    )

    fig = px.line(
        rolling_corr.dropna(),
        title=f"{window}-Day Rolling Correlation"
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Correlation",
        yaxis_range=[-1, 1],
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Correlation measures statistical co-movement, "
        "not causation or future predictive power."
    )

# -------------------------------
# TAB 3: FACTOR REGRESSION
# -------------------------------

with tab3:
    st.header("Factor Regression")

    st.markdown(
        "Estimate how stock returns relate to broad-market "
        "and sector movements using ordinary least squares."
    )

    # SOXX and SPY overlap, so we use sector excess return.
    factor_data = pd.DataFrame({
        "Stock": returns[selected_stock],
        "Market": returns["SPY"],
        "Sector_Excess": returns["SOXX"] - returns["SPY"]
    }).dropna()

    if len(factor_data) < 30:
        st.warning("Insufficient observations for regression.")
    else:
        X = sm.add_constant(
            factor_data[["Market", "Sector_Excess"]]
        )
        y = factor_data["Stock"]

        model = sm.OLS(y, X).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 5}
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("R²", f"{model.rsquared:.3f}")
        c2.metric(
            "Market Beta",
            f"{model.params['Market']:.2f}"
        )
        c3.metric(
            "Sector Excess Beta",
            f"{model.params['Sector_Excess']:.2f}"
        )

        st.subheader("Regression Coefficients")

        coefficients = pd.DataFrame({
            "Coefficient": model.params,
            "P-Value": model.pvalues,
            "T-Statistic": model.tvalues
        })

        st.dataframe(
            coefficients,
            use_container_width=True
        )

        st.subheader("Actual vs Model-Fitted Returns")

        fitted = pd.DataFrame({
            "Actual": y * 100,
            "Fitted": model.fittedvalues * 100
        })

        fig = px.line(
            fitted,
            title="Actual and Fitted Daily Returns"
        )
        fig.update_layout(
            yaxis_title="Daily Return (%)",
            xaxis_title="Date"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Unexplained Return (Residual)")

        residual = model.resid * 100

        fig = px.line(
            residual,
            title="Daily Regression Residual"
        )
        fig.update_layout(
            yaxis_title="Residual (%)",
            xaxis_title="Date",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "This is an in-sample explanatory model, "
            "not a validated forecasting or causal model. "
            "HAC standard errors account for some "
            "serial correlation and heteroskedasticity."
        )

        with st.expander("Full Regression Summary"):
            st.text(model.summary().as_text())

# -------------------------------
# TAB 4: ROLLING RISK
# -------------------------------

with tab4:
    st.header("Rolling Risk Analysis")

    stock_r = returns[selected_stock]
    market_r = returns["SPY"]

    rolling_cov = stock_r.rolling(window).cov(market_r)
    rolling_var = market_r.rolling(window).var()

    rolling_beta = rolling_cov / rolling_var.replace(0, np.nan)

    st.subheader("Rolling Market Beta")

    fig = px.line(
        rolling_beta.dropna(),
        title=f"{window}-Day Rolling Market Beta"
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Beta",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    rolling_vol = (
        stock_r.rolling(window).std()
        * np.sqrt(252) * 100
    )

    st.subheader("Rolling Annualised Volatility")

    fig = px.line(
        rolling_vol.dropna(),
        title=f"{window}-Day Rolling Volatility"
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Rolling beta and volatility are historical estimates "
        "and can change significantly over time."
    )

st.divider()
st.caption(
    "Quant Market Lab v1 | Educational research only | "
    "Data from Yahoo Finance via yfinance"
)
