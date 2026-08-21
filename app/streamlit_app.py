"""
Customer Transaction Analytics — Interactive Dashboard
=======================================================

Streamlit dashboard for the Customer Transaction Analytics project.

Run locally:
    streamlit run app/streamlit_app.py

Expects a cleaned, customer-level CSV (see README for schema). If no file
is uploaded, the app falls back to a synthetic demo dataset so the
dashboard is always explorable (useful for a live portfolio demo link).

Expected columns (missing ones are engineered on the fly where possible):
    customer_id, transaction_id, transaction_date, transaction_amount,
    item_price, quantity, product_category, product_name, payment_method,
    customer_age, customer_tenure_days, total_spend_per_customer,
    days_since_last_login, price_tier, loyalty_program_member,
    device_used, shipping_address_state, transaction_status
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Transaction Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

REQUIRED_NUMERIC_DEFAULTS = {
    "customer_age": 35,
    "customer_tenure_days": 365,
    "total_spend_per_customer": 0.0,
    "days_since_last_login": 30,
}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_demo_data(n_customers: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Synthetic fallback dataset so the dashboard works with zero setup."""
    rng = np.random.default_rng(seed)

    categories = ["Electronics", "Apparel", "Home & Kitchen", "Beauty", "Sports", "Books"]
    payment_methods = ["Credit Card", "PayPal", "Bank Transfer", "Debit Card"]
    devices = ["Mobile", "Desktop", "Tablet"]
    states = ["Maharashtra", "Karnataka", "Delhi", "Tamil Nadu", "Gujarat", "West Bengal", "Uttar Pradesh"]
    statuses = ["Completed", "Refunded", "Completed", "Completed", "Cancelled"]

    customer_id = np.arange(1, n_customers + 1)
    tenure_days = rng.integers(10, 1500, n_customers)
    # Recency skewed so a meaningful minority look "at risk"
    recency_days = rng.exponential(scale=45, size=n_customers).astype(int)
    recency_days = np.clip(recency_days, 0, 400)
    loyalty = rng.choice([True, False], n_customers, p=[0.35, 0.65])

    # Loyal / recently-active customers spend more, on average
    base_spend = rng.gamma(shape=2.2, scale=90, size=n_customers)
    spend_multiplier = np.where(loyalty, 1.35, 1.0) * np.where(recency_days < 30, 1.2, 0.85)
    total_spend = np.round(base_spend * spend_multiplier, 2)

    age = rng.integers(18, 65, n_customers)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "customer_age": age,
            "customer_tenure_days": tenure_days,
            "days_since_last_login": recency_days,
            "total_spend_per_customer": total_spend,
            "loyalty_program_member": loyalty,
            "product_category": rng.choice(categories, n_customers),
            "payment_method": rng.choice(payment_methods, n_customers),
            "device_used": rng.choice(devices, n_customers),
            "shipping_address_state": rng.choice(states, n_customers),
            "transaction_status": rng.choice(statuses, n_customers),
            "quantity": rng.integers(1, 6, n_customers),
        }
    )
    df["transaction_amount"] = np.round(df["total_spend_per_customer"] / rng.integers(1, 5, n_customers), 2)
    df["price_tier"] = pd.cut(
        df["transaction_amount"],
        bins=[-1, 25, 75, 200, np.inf],
        labels=["Low", "Mid", "High", "Premium"],
    )
    dates = pd.Timestamp.today().normalize() - pd.to_timedelta(
        rng.integers(0, 365, n_customers), unit="D"
    )
    df["transaction_date"] = dates
    return df


@st.cache_data(show_spinner=False)
def load_uploaded_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    return df


def engineer_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill in engineered columns the dashboard relies on if they're absent."""
    df = df.copy()

    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

    if "total_spend_per_customer" not in df.columns and "transaction_amount" in df.columns and "customer_id" in df.columns:
        spend = df.groupby("customer_id")["transaction_amount"].transform("sum")
        df["total_spend_per_customer"] = spend

    if "price_tier" not in df.columns and "transaction_amount" in df.columns:
        df["price_tier"] = pd.cut(
            df["transaction_amount"],
            bins=[-1, 25, 75, 200, np.inf],
            labels=["Low", "Mid", "High", "Premium"],
        )

    for col, default in REQUIRED_NUMERIC_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    if "loyalty_program_member" in df.columns:
        df["loyalty_program_member"] = df["loyalty_program_member"].astype(str).str.lower().isin(
            ["true", "1", "yes", "y"]
        )
    else:
        df["loyalty_program_member"] = False

    return df


# --------------------------------------------------------------------------
# Churn model
# --------------------------------------------------------------------------
CHURN_FEATURES = ["days_since_last_login", "customer_tenure_days", "total_spend_per_customer"]


@st.cache_data(show_spinner=False)
def customer_level_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per customer_id, keeping the churn features."""
    if "customer_id" not in df.columns:
        df = df.copy()
        df["customer_id"] = np.arange(len(df))

    agg = {f: "first" for f in CHURN_FEATURES if f in df.columns}
    if "loyalty_program_member" in df.columns:
        agg["loyalty_program_member"] = "first"
    cust = df.groupby("customer_id").agg(agg).reset_index()
    return cust


def label_churn(cust: pd.DataFrame, recency_threshold_days: int) -> pd.Series:
    return (cust["days_since_last_login"] >= recency_threshold_days).astype(int)


@st.cache_data(show_spinner=False)
def train_churn_model(cust: pd.DataFrame, recency_threshold_days: int, model_name: str):
    cust = cust.copy()
    cust["churned"] = label_churn(cust, recency_threshold_days)

    X = cust[CHURN_FEATURES].fillna(0)
    y = cust["churned"]

    if y.nunique() < 2:
        return None  # can't train with a single class

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_s, y_train)
        importances = pd.Series(np.abs(model.coef_[0]), index=CHURN_FEATURES)
    else:
        model = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
        model.fit(X_train_s, y_train)
        importances = pd.Series(model.feature_importances_, index=CHURN_FEATURES)

    proba_test = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, proba_test) if y_test.nunique() > 1 else np.nan
    fpr, tpr, _ = roc_curve(y_test, proba_test) if y_test.nunique() > 1 else ([0, 1], [0, 1], None)
    cm = confusion_matrix(y_test, (proba_test >= 0.5).astype(int))

    # Score the full customer base
    full_proba = model.predict_proba(scaler.transform(X))[:, 1]
    cust["churn_probability"] = full_proba

    return {
        "cust_scored": cust,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "confusion_matrix": cm,
        "importances": importances.sort_values(ascending=True),
    }


# --------------------------------------------------------------------------
# Sidebar — data source
# --------------------------------------------------------------------------
st.sidebar.title("🛒 Data Source")
uploaded = st.sidebar.file_uploader("Upload cleaned transactions CSV", type=["csv"])

if uploaded is not None:
    raw_df = load_uploaded_csv(uploaded)
    st.sidebar.success(f"Loaded {len(raw_df):,} rows from your file.")
else:
    raw_df = generate_demo_data()
    st.sidebar.info("No file uploaded — showing synthetic demo data.")

df = engineer_missing_features(raw_df)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Expected columns: `customer_id`, `transaction_amount`, `days_since_last_login`, "
    "`customer_tenure_days`, `total_spend_per_customer`, `loyalty_program_member`, "
    "`product_category`, `payment_method`, `device_used`, `shipping_address_state`, "
    "`transaction_date`. Missing engineered columns are filled in automatically."
)

# --------------------------------------------------------------------------
# Header + KPIs
# --------------------------------------------------------------------------
st.title("🛒 Customer Transaction Analytics")
st.caption("Interactive dashboard — EDA, business insights, and churn risk scoring")

cust_df = customer_level_frame(df)

kpi_cols = st.columns(5)
total_customers = cust_df["customer_id"].nunique()
total_revenue = df["transaction_amount"].sum() if "transaction_amount" in df.columns else np.nan
avg_order_value = df["transaction_amount"].mean() if "transaction_amount" in df.columns else np.nan
loyalty_pct = df["loyalty_program_member"].mean() * 100 if "loyalty_program_member" in df.columns else np.nan
at_risk_pct = (cust_df["days_since_last_login"] >= cust_df["days_since_last_login"].quantile(0.75)).mean() * 100

kpi_cols[0].metric("Total Customers", f"{total_customers:,}")
kpi_cols[1].metric("Total Revenue", f"₹{total_revenue:,.0f}" if not np.isnan(total_revenue) else "—")
kpi_cols[2].metric("Avg Order Value", f"₹{avg_order_value:,.2f}" if not np.isnan(avg_order_value) else "—")
kpi_cols[3].metric("Loyalty Members", f"{loyalty_pct:,.1f}%")
kpi_cols[4].metric("Churn-Risk Customers", f"{at_risk_pct:,.1f}%", help="Top quartile of days since last login")

st.markdown("---")

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_overview, tab_eda, tab_top, tab_churn, tab_data = st.tabs(
    ["📊 Overview", "🔍 EDA", "👑 Top Customers", "⚠️ Churn Risk", "📥 Data Explorer"]
)

# ---- Overview -------------------------------------------------------------
with tab_overview:
    c1, c2 = st.columns(2)

    with c1:
        if "product_category" in df.columns and "transaction_amount" in df.columns:
            cat_rev = (
                df.groupby("product_category")["transaction_amount"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            fig = px.bar(
                cat_rev, x="product_category", y="transaction_amount",
                title="Revenue by Product Category", labels={"transaction_amount": "Revenue"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "loyalty_program_member" in df.columns and "transaction_amount" in df.columns:
            loyalty_spend = df.groupby("loyalty_program_member")["transaction_amount"].mean().reset_index()
            loyalty_spend["loyalty_program_member"] = loyalty_spend["loyalty_program_member"].map(
                {True: "Member", False: "Non-Member"}
            )
            fig = px.bar(
                loyalty_spend, x="loyalty_program_member", y="transaction_amount",
                title="Avg Transaction Amount: Loyalty vs Non-Loyalty",
                labels={"transaction_amount": "Avg Amount", "loyalty_program_member": ""},
            )
            st.plotly_chart(fig, use_container_width=True)

    if "transaction_status" in df.columns and "product_category" in df.columns:
        refund_rate = (
            df.assign(is_refunded=df["transaction_status"].eq("Refunded"))
            .groupby("product_category")["is_refunded"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
            .reset_index()
        )
        fig = px.bar(
            refund_rate, x="product_category", y="is_refunded",
            title="Refund Rate (%) by Category", labels={"is_refunded": "Refund Rate (%)"},
        )
        st.plotly_chart(fig, use_container_width=True)

# ---- EDA --------------------------------------------------------------
with tab_eda:
    c1, c2 = st.columns(2)

    with c1:
        if "transaction_amount" in df.columns:
            fig = px.histogram(
                df, x="transaction_amount", nbins=40, title="Transaction Amount Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "payment_method" in df.columns:
            pm = df["payment_method"].value_counts().reset_index()
            pm.columns = ["payment_method", "count"]
            fig = px.pie(pm, names="payment_method", values="count", title="Payment Method Mix")
            st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        if "transaction_date" in df.columns and df["transaction_date"].notna().any():
            monthly = (
                df.set_index("transaction_date")["transaction_amount"]
                .resample("MS").sum().reset_index()
            )
            fig = px.line(monthly, x="transaction_date", y="transaction_amount", title="Revenue by Month")
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        if "device_used" in df.columns and "transaction_amount" in df.columns:
            dev = df.groupby("device_used")["transaction_amount"].sum().reset_index()
            fig = px.bar(dev, x="device_used", y="transaction_amount", title="Spend by Device")
            st.plotly_chart(fig, use_container_width=True)

    if "shipping_address_state" in df.columns:
        state_counts = df["shipping_address_state"].value_counts().reset_index()
        state_counts.columns = ["state", "transactions"]
        fig = px.bar(
            state_counts.head(15), x="state", y="transactions",
            title="Transactions by State (Top 15)",
        )
        st.plotly_chart(fig, use_container_width=True)

# ---- Top Customers ------------------------------------------------------
with tab_top:
    top_n_pct = st.slider("Top % of customers by spend", 1, 50, 10)
    threshold = cust_df["total_spend_per_customer"].quantile(1 - top_n_pct / 100)
    top_customers = cust_df[cust_df["total_spend_per_customer"] >= threshold].sort_values(
        "total_spend_per_customer", ascending=False
    )

    revenue_share = (
        top_customers["total_spend_per_customer"].sum() / cust_df["total_spend_per_customer"].sum() * 100
        if cust_df["total_spend_per_customer"].sum() > 0 else 0
    )
    st.metric(f"Revenue share from top {top_n_pct}% of customers", f"{revenue_share:,.1f}%")

    st.dataframe(
        top_customers[["customer_id"] + [c for c in CHURN_FEATURES if c in top_customers.columns]],
        use_container_width=True,
        height=350,
    )

    if "product_category" in df.columns and "customer_id" in df.columns:
        top_ids = set(top_customers["customer_id"])
        top_cat = df[df["customer_id"].isin(top_ids)]["product_category"].value_counts().reset_index()
        top_cat.columns = ["product_category", "count"]
        fig = px.treemap(top_cat, path=["product_category"], values="count", title="What Top Customers Buy")
        st.plotly_chart(fig, use_container_width=True)

# ---- Churn Risk -----------------------------------------------------------
with tab_churn:
    st.subheader("Churn Prediction")
    st.caption(
        "No historical churn label exists in the raw data, so churn is defined here as "
        "'inactive beyond a recency threshold.' Adjust the threshold, pick a model, "
        "and the classifier learns the relationship between recency, tenure, and spend."
    )

    c1, c2 = st.columns(2)
    with c1:
        recency_threshold = st.slider(
            "Days since last login = churned",
            min_value=15, max_value=365,
            value=int(cust_df["days_since_last_login"].quantile(0.75)) or 60,
            step=5,
        )
    with c2:
        model_choice = st.selectbox("Model", ["Logistic Regression", "Random Forest"])

    result = train_churn_model(cust_df, recency_threshold, model_choice)

    if result is None:
        st.warning(
            "Every customer falls on one side of that threshold — move the slider "
            "so both churned and active customers are represented."
        )
    else:
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Test-set ROC-AUC", f"{result['auc']:.3f}" if not np.isnan(result["auc"]) else "—")
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=result["fpr"], y=result["tpr"], mode="lines", name="Model"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
            fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, use_container_width=True)

        with m2:
            imp_fig = px.bar(
                result["importances"], orientation="h",
                title=f"Feature Importance ({model_choice})",
                labels={"value": "Importance", "index": "Feature"},
            )
            st.plotly_chart(imp_fig, use_container_width=True)

        st.markdown("#### Customers scored as highest churn risk")
        scored = result["cust_scored"].sort_values("churn_probability", ascending=False)
        st.dataframe(
            scored[["customer_id", "churn_probability"] + CHURN_FEATURES],
            use_container_width=True,
            height=350,
        )

        st.download_button(
            "⬇️ Download churn scores (CSV)",
            data=scored.to_csv(index=False).encode("utf-8"),
            file_name="churn_scores.csv",
            mime="text/csv",
        )

# ---- Data Explorer --------------------------------------------------------
with tab_data:
    st.subheader("Filter & Explore")

    filter_cols = st.columns(3)
    filtered = df.copy()

    if "product_category" in df.columns:
        cats = filter_cols[0].multiselect("Category", sorted(df["product_category"].dropna().unique()))
        if cats:
            filtered = filtered[filtered["product_category"].isin(cats)]

    if "payment_method" in df.columns:
        pms = filter_cols[1].multiselect("Payment Method", sorted(df["payment_method"].dropna().unique()))
        if pms:
            filtered = filtered[filtered["payment_method"].isin(pms)]

    if "loyalty_program_member" in df.columns:
        loyalty_filter = filter_cols[2].selectbox("Loyalty", ["All", "Members only", "Non-members only"])
        if loyalty_filter == "Members only":
            filtered = filtered[filtered["loyalty_program_member"]]
        elif loyalty_filter == "Non-members only":
            filtered = filtered[~filtered["loyalty_program_member"]]

    st.dataframe(filtered, use_container_width=True, height=450)
    st.caption(f"{len(filtered):,} rows shown")

    st.download_button(
        "⬇️ Download filtered data (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_transactions.csv",
        mime="text/csv",
    )
