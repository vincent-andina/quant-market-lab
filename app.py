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

# -------------------------------
# MANUAL DATA REFRESH
# -------------------------------

if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()


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


@st.cache_data(ttl=3600)
def load_fundamentals(ticker):
    """Latest available Yahoo Finance valuation data (not historical as-of-date)."""
    info = yf.Ticker(ticker).info
    return {
        "trailing_pe": info.get("trailingPE"),
        "trailing_eps": info.get("trailingEps"),
    }


def format_fundamental(value, suffix=""):
    """Show N/A for missing, non-numeric, or non-finite values."""
    try:
        number = float(value)
        return f"{number:,.2f}{suffix}" if np.isfinite(number) else "N/A"
    except (TypeError, ValueError):
        return "N/A"


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
    daily_vol = stock_returns.std() * 100
    annual_vol = stock_returns.std() * np.sqrt(252) * 100
    total_return = (
        stock_prices.iloc[-1] / stock_prices.iloc[0] - 1
    ) * 100

    # Keep the overview compact: explanations open only when requested.
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Latest Price", f"${latest_price:,.2f}")
    c2.metric("Latest Daily Return", f"{latest_return:+.2f}%")

    c3.metric("Annualised Volatility", f"{annual_vol:.2f}%")

    with c3.popover("ⓘ How is this calculated?"):
        st.markdown(r"""
### Daily Volatility

**Daily volatility** is the sample standard deviation of daily returns.
It measures how much daily returns fluctuate around their average,
regardless of whether returns are positive or negative.

The sample standard deviation is:

$$
s_{\text{daily}} =
\sqrt{
\frac{
\sum_{i=1}^{n}(R_i-\bar{R})^2
}{
n-1
}
}
$$

Where:

- $R_i$ = return on day $i$
- $\bar{R}$ = average daily return
- $n$ = number of daily return observations

The expression inside the square root is the **sample variance**.

We divide by $n-1$ because we are estimating variance from a sample.
This is also the default behaviour of `pandas.std()`.

---

### Why Multiply by the Square Root of 252?

**Step 1: Understand the addition property of variances**

For two random variables $A$ and $B$:

$$
\operatorname{Var}(A+B)
=
\operatorname{Var}(A)
+
\operatorname{Var}(B)
+
2\operatorname{Cov}(A,B)
$$

If $A$ and $B$ are independent, their covariance is zero.

Therefore:

$$
\boxed{
\operatorname{Var}(A+B)
=
\operatorname{Var}(A)
+
\operatorname{Var}(B)
}
$$

**Key concept:** The variance of the sum of independent
random variables equals the sum of their individual variances.

---

**Step 2: Extend this property to multiple trading days**

Suppose every day's return has the same variance:

$$
\operatorname{Var}(R_i)=\sigma_{\text{daily}}^2
$$

For 2 independent trading days:

$$
\operatorname{Var}(R_1+R_2)
=
2\sigma_{\text{daily}}^2
$$

For 3 independent trading days:

$$
\operatorname{Var}(R_1+R_2+R_3)
=
3\sigma_{\text{daily}}^2
$$

For $T$ independent trading days:

$$
\boxed{
\operatorname{Var}(R_1+\cdots+R_T)
=
T\sigma_{\text{daily}}^2
}
$$

Therefore, **variance increases linearly with the number
of trading days**.

---

**Step 3: Convert variance into standard deviation**

Standard deviation is the square root of variance:

$$
\sigma_T
=
\sqrt{\operatorname{Var}(R_1+\cdots+R_T)}
$$

Substitute the variance addition formula:

$$
\sigma_T
=
\sqrt{T\sigma_{\text{daily}}^2}
$$

Using the square root property:

$$
\sqrt{ab}=\sqrt{a}\sqrt{b}
$$

We obtain:

$$
\sigma_T
=
\sqrt{T}\sqrt{\sigma_{\text{daily}}^2}
$$

Since volatility is non-negative:

$$
\boxed{
\sigma_T=\sqrt{T}\sigma_{\text{daily}}
}
$$

**Key concept:** Variance scales linearly with time,
but standard deviation scales with the square root of time.

---

**Step 4: Annualise using 252 trading days**

The US stock market has approximately 252 trading days per year.

Setting $T=252$:

$$
\boxed{
\sigma_{\text{annual}}
=
\sigma_{\text{daily}}\sqrt{252}
}
$$

Since:

$$
\sqrt{252}\approx15.87
$$

Annualised volatility is approximately 15.87 times daily volatility.

---

### Numerical Example

Suppose daily volatility is 2%.

Daily variance:

$$
\sigma_{\text{daily}}^2
=
0.02^2
=
0.0004
$$

Annual variance:

$$
\sigma_{\text{annual}}^2
=
252\times0.0004
=
0.1008
$$

Taking the square root:

$$
\sigma_{\text{annual}}
=
\sqrt{0.1008}
\approx31.75\%
$$

Therefore:

$$
\boxed{
\sigma_{\text{annual}}\approx31.75\%
}
$$

This is an estimate of the standard deviation of annual returns,
not an expected annual return or maximum possible loss.
""")

        st.info(
            f"For {selected_stock}: {daily_vol:.2f}% daily volatility × √252 "
            f"= **{annual_vol:.2f}% annualised volatility**."
        )

        st.caption(
            "Assumes independent daily returns with constant variance. "
            "The variance derivation applies to sums of returns; "
            "simple percentage returns compound rather than add exactly. "
            "Square-root scaling may be inaccurate with serial correlation "
            "or changing volatility."
        )

    c4.metric("Period Return", f"{total_return:+.2f}%")

    # Fundamentals are current Yahoo Finance figures,
    # not historical as-of-date.
    try:
        fundamentals = load_fundamentals(selected_stock)
    except Exception:
        fundamentals = {
            "trailing_pe": None,
            "trailing_eps": None
        }

    pe_value = format_fundamental(
        fundamentals["trailing_pe"], "×"
    )
    eps_value = format_fundamental(
        fundamentals["trailing_eps"]
    )
    eps_display = (
        f"${eps_value}" if eps_value != "N/A" else "N/A"
    )

    v1, v2 = st.columns(2)

    v1.metric("Trailing P/E Ratio", pe_value)

    with v1.popover("ⓘ What is P/E?"):
        st.markdown(
            "**Price-to-Earnings (P/E)** compares the current share price "
            "with earnings per share over the trailing twelve months."
        )

        st.latex(
            r"P/E=\frac{\text{Current share price}}"
            r"{\text{Trailing EPS}}"
        )

        st.markdown(
            "A P/E of 20× means investors pay $20 per share for each "
            "$1 of annual earnings per share. It is not, by itself, "
            "a measure of whether a stock is cheap or expensive."
        )

        st.caption(
            "Latest available Yahoo Finance trailing P/E; may be unavailable "
            "for negative earnings. Not tied to the selected historical period."
        )

    v2.metric("Trailing EPS", eps_display)

    with v2.popover("ⓘ What is EPS?"):
        st.markdown(
            "**Earnings Per Share (EPS)** measures earnings attributable "
            "to each weighted-average common share outstanding. "
            "Trailing EPS generally covers the latest twelve months."
        )

        st.latex(
            r"EPS=\frac{\text{Earnings attributable to common shareholders}}"
            r"{\text{Weighted-average common shares}}"
        )

        st.caption(
            "Latest available Yahoo Finance trailing EPS, which may be "
            "missing or revised. Not tied to the selected historical period."
        )

    # -------------------------------
    # HISTORICAL ADJUSTED PRICE
    # -------------------------------

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

    # -------------------------------
    # DAILY RETURN DISTRIBUTION
    # -------------------------------

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

    # -------------------------------
    # NORMALISED PRICE COMPARISON
    # -------------------------------

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
