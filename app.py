import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
#HELOO 

st.set_page_config(
    page_title="Online Retail Intelligence Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .stMetric { background-color: #1c1f26; padding: 15px; border-radius: 10px; }
    div[data-testid="metric-container"] { background-color: #1c1f26; border-radius: 10px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────
@st.cache_data
def load_data(path: str = "online_retail.csv") -> pd.DataFrame:
    """Load and clean the Online Retail dataset."""
    df = pd.read_csv(path)

    # Drop rows with missing CustomerID or InvoiceNo
    df.dropna(subset=["CustomerID", "InvoiceNo"], inplace=True)

    # Remove cancelled orders (InvoiceNo starts with 'C') and invalid values
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # Feature engineering
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    return df


df_full = load_data()

# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
st.sidebar.header("🔍 Filters")

countries = ["All"] + sorted(df_full["Country"].unique().tolist())
country_filter = st.sidebar.selectbox("Market (Country)", countries)

df = df_full if country_filter == "All" else df_full[df_full["Country"] == country_filter]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Records after filter:** `{len(df):,}`")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🛒 Online Retail Intelligence Dashboard")
st.markdown("##### Advanced Sales Analytics · Outlier Detection · CLV Modelling · Hypothesis Testing")
st.markdown("---")

# ─────────────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────────────
st.subheader("📊 Key Performance Indicators")

total_orders     = df["InvoiceNo"].nunique()
total_revenue    = df["Revenue"].sum()
avg_order_value  = total_revenue / total_orders if total_orders else 0
unique_customers = df["CustomerID"].nunique()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Orders",       f"{total_orders:,}")
col2.metric("Total Revenue",      f"${total_revenue:,.0f}")
col3.metric("Avg Order Value",    f"${avg_order_value:,.2f}")
col4.metric("Unique Customers",   f"{unique_customers:,}")

st.markdown("---")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def center_chart(fig):
    """Render a matplotlib figure centred in a 3-column layout."""
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.pyplot(fig)
    plt.close(fig)


def styled_fig(figsize=(7, 4)):
    """Return a figure with a dark background consistent with the app theme."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    return fig, ax

# ─────────────────────────────────────────────
# 1. ORDER VALUE DISTRIBUTION
# ─────────────────────────────────────────────
st.subheader("📈 Order Value Distribution")
st.markdown("Distribution of revenue per invoice (95th-percentile cap applied to reduce skew).")

order_values = df.groupby("InvoiceNo")["Revenue"].sum()

if len(order_values) > 5:
    cap = order_values.quantile(0.95)
    plot_data = order_values[order_values <= cap]

    fig1, ax1 = styled_fig()
    sns.histplot(plot_data, kde=True, bins=40, color="#2ec4b6", ax=ax1)
    ax1.set_xlabel("Order Value ($)", color="white")
    ax1.set_ylabel("Frequency", color="white")
    ax1.set_title("Revenue per Order (capped at 95th pct)", color="white")
    center_chart(fig1)
else:
    st.info("Not enough data to plot distribution.")

st.markdown("---")

# ─────────────────────────────────────────────
# 2. MONTHLY REVENUE TREND
# ─────────────────────────────────────────────
st.subheader("📅 Monthly Revenue Trend")
st.markdown("Aggregated revenue by calendar month.")

if "Month" in df.columns and df["Month"].notna().any():
    monthly = df.groupby("Month")["Revenue"].sum().reset_index().sort_values("Month")

    fig2, ax2 = styled_fig(figsize=(9, 4))
    ax2.plot(monthly["Month"], monthly["Revenue"], marker="o", color="#ff6b6b", linewidth=2)
    ax2.fill_between(range(len(monthly)), monthly["Revenue"], alpha=0.15, color="#ff6b6b")
    ax2.set_xticks(range(len(monthly)))
    ax2.set_xticklabels(monthly["Month"], rotation=45, ha="right", color="white", fontsize=8)
    ax2.set_xlabel("Month", color="white")
    ax2.set_ylabel("Revenue ($)", color="white")
    ax2.set_title("Monthly Revenue", color="white")
    center_chart(fig2)

st.markdown("---")

# ─────────────────────────────────────────────
# 3. OUTLIER DETECTION (LOG-SCALE BOX PLOTS)
# ─────────────────────────────────────────────
st.subheader("📦 Outlier Detection — Log-Transformed Box Plots")
st.markdown("Log₁ transformation compresses extreme values, making outliers visible across all three numeric dimensions simultaneously.")

log_df = np.log1p(df[["Quantity", "UnitPrice", "Revenue"]])

fig3, ax3 = styled_fig()
sns.boxplot(data=log_df, palette="Set2", ax=ax3)
ax3.set_ylabel("log(1 + value)", color="white")
ax3.set_title("Log-Scaled Box Plots", color="white")
ax3.tick_params(axis="x", colors="white")
center_chart(fig3)

st.markdown("---")

# ─────────────────────────────────────────────
# 4. CORRELATION HEATMAP
# ─────────────────────────────────────────────
st.subheader("🔥 Multivariate Correlation Heatmap")
st.markdown("Pearson correlation between the three core numeric features.")

corr_matrix = df[["Quantity", "UnitPrice", "Revenue"]].corr()

fig4, ax4 = styled_fig()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f",
            linewidths=0.5, ax=ax4,
            annot_kws={"color": "white"})
ax4.set_title("Pearson Correlation Matrix", color="white")
ax4.tick_params(colors="white")
center_chart(fig4)

st.markdown("---")

# ─────────────────────────────────────────────
# 5. TOP PRODUCTS BY REVENUE
# ─────────────────────────────────────────────
st.subheader("🏆 Top 10 Products by Revenue")

top_products = (
    df.groupby("Description")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig5, ax5 = styled_fig(figsize=(8, 5))
bars = ax5.barh(top_products["Description"], top_products["Revenue"], color="#f7b731")
ax5.invert_yaxis()
ax5.set_xlabel("Total Revenue ($)", color="white")
ax5.set_title("Top 10 Products by Revenue", color="white")
ax5.tick_params(axis="y", labelsize=8, colors="white")
center_chart(fig5)

st.markdown("---")

# ─────────────────────────────────────────────
# 6. HYPOTHESIS TESTING (WELCH'S T-TEST)
# ─────────────────────────────────────────────
st.subheader("🧪 Hypothesis Testing — Welch's T-Test")
st.markdown("""
**H₀:** No significant difference in average order value between UK and International markets.  
**H₁:** A significant difference exists (|t| > 1.96 at α = 0.05).
""")

uk_orders   = df_full[df_full["Country"] == "United Kingdom"].groupby("InvoiceNo")["Revenue"].sum().values
intl_orders = df_full[df_full["Country"] != "United Kingdom"].groupby("InvoiceNo")["Revenue"].sum().values

if len(uk_orders) > 1 and len(intl_orders) > 1:
    mean1, mean2 = np.mean(uk_orders), np.mean(intl_orders)
    var1,  var2  = np.var(uk_orders, ddof=1), np.var(intl_orders, ddof=1)
    n1,    n2    = len(uk_orders), len(intl_orders)

    t_stat = (mean1 - mean2) / np.sqrt((var1 / n1) + (var2 / n2))

    c1, c2, c3 = st.columns(3)
    c1.metric("UK Avg Order Value",       f"${mean1:.2f}")
    c2.metric("Int'l Avg Order Value",    f"${mean2:.2f}")
    c3.metric("T-Statistic",              f"{t_stat:.4f}")

    if abs(t_stat) > 1.96:
        st.success("✅ Reject H₀ — Statistically significant difference in order values detected.")
    else:
        st.warning("⚠️ Fail to Reject H₀ — No statistically significant difference found.")
else:
    st.info("Select **All** countries in the sidebar to run this test.")

st.markdown("---")

# ─────────────────────────────────────────────
# 7. CLV PROXY — OLS REGRESSION
# ─────────────────────────────────────────────
st.subheader("📉 Customer Lifetime Value (CLV) Proxy — OLS Regression")
st.markdown("Regresses **Total Spend** on **Purchase Frequency** to approximate CLV for each customer.")

customer_df = df.groupby("CustomerID").agg(
    Frequency=("InvoiceNo", "nunique"),
    Total_Spend=("Revenue", "sum"),
).dropna()

if len(customer_df) > 1:
    # Cap outliers
    q95_spend = customer_df["Total_Spend"].quantile(0.95)
    q95_freq  = customer_df["Frequency"].quantile(0.95)
    customer_df = customer_df[
        (customer_df["Total_Spend"] <= q95_spend) &
        (customer_df["Frequency"]   <= q95_freq)
    ]

    X = customer_df["Frequency"].values
    Y = customer_df["Total_Spend"].values

    m, c_intercept = np.polyfit(X, Y, 1)
    Y_pred = m * X + c_intercept

    ss_total    = np.sum((Y - np.mean(Y)) ** 2)
    ss_residual = np.sum((Y - Y_pred) ** 2)
    r2          = 1 - (ss_residual / ss_total) if ss_total != 0 else 0

    col_eq, col_r2 = st.columns(2)
    col_eq.info(f"**Equation:** Total Spend = ({m:.2f} × Frequency) + {c_intercept:.2f}")
    col_r2.info(f"**R² Score:** {r2:.4f}")

    fig6, ax6 = styled_fig()
    ax6.scatter(X, Y, alpha=0.4, color="#a29bfe", s=15, label="Customers")
    sort_idx = np.argsort(X)
    ax6.plot(X[sort_idx], Y_pred[sort_idx], color="#fd79a8", linewidth=2, label="OLS Fit")
    ax6.set_xlabel("Purchase Frequency (# Orders)", color="white")
    ax6.set_ylabel("Total Spend ($)", color="white")
    ax6.set_title("Frequency vs Total Spend", color="white")
    ax6.legend(facecolor="#1c1f26", labelcolor="white")
    center_chart(fig6)

    # ─── Interactive Predictor ────────────────────
    st.subheader("🤖 CLV Predictor Tool")
    st.markdown("Drag the slider to estimate how much a customer with a given purchase frequency will spend in total.")

    max_freq  = int(customer_df["Frequency"].max())
    freq_input = st.slider("Target Purchase Frequency", min_value=1, max_value=max(20, max_freq), value=5)
    pred_spend = m * freq_input + c_intercept

    st.success(f"💰 Estimated Total Spend (CLV Proxy): **${pred_spend:,.2f}**")
else:
    st.info("Not enough customer data for regression analysis.")

st.markdown("---")
st.caption("Data: UCI Online Retail Dataset · Dashboard built with Streamlit · © 2024")
