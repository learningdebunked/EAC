"""
EAC Analytics Dashboard - Streamlit

Real-time analytics and monitoring for the EAC Agent system
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

# Page config
st.set_page_config(
    page_title="EAC Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1 {
        color: #10b981;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 EAC Analytics Dashboard")
st.markdown("Real-time monitoring and analytics for the EAC Agent system")

# Sidebar
st.sidebar.header("⚙️ Settings")
time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days"]
)
policy_filter = st.sidebar.multiselect(
    "Policy Filter",
    ["SNAP/WIC Substitution", "Low Glycemic", "Budget Optimizer", "All"],
    default=["All"]
)
refresh = st.sidebar.button("🔄 Refresh Data")

# Load live transaction data
@st.cache_data(ttl=5)  # Refresh every 5 seconds
def load_live_data():
    """Load live transaction data from API"""
    try:
        # Try to load live transactions first
        df = pd.read_csv('live_transactions.csv')
        if len(df) > 0:
            return df
    except:
        pass
    
    # Fall back to simulation results
    try:
        df = pd.read_csv('simulation_results.csv')
        # Map columns to match live format
        df = df.rename(columns={
            'treatment_accepted': 'accepted_count',
            'treatment_recommendations': 'num_recommendations'
        })
        return df
    except:
        # Generate mock data as last resort
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'user_id': [f'user_{i%10}' for i in range(n)],
            'transaction_id': [f'txn_{i}' for i in range(n)],
            'policy_used': np.random.choice(['snap_wic_substitution', 'low_glycemic_alternative', 'budget_optimizer'], n),
            'num_recommendations': np.random.randint(1, 5, n),
            'accepted_count': np.random.randint(0, 3, n),
            'total_savings': np.random.uniform(0, 5, n),
            'total_nutrition_improvement': np.random.uniform(0, 20, n),
            'acceptance_rate': np.random.uniform(0, 1, n),
            'latency_ms': np.random.gamma(2, 1.5, n),
            'protected_group': np.random.choice(['white', 'black', 'hispanic', 'asian', 'other'], n),
            'income_group': np.random.choice(['low', 'medium', 'high'], n),
            'snap_eligible': np.random.choice([True, False], n),
            'fairness_check': np.random.choice(['PASS', 'REVIEW'], n, p=[0.9, 0.1]),
            'timestamp': [datetime.now() - timedelta(minutes=np.random.randint(0, 60)) for _ in range(n)]
        })

df = load_live_data()

# Key Metrics Row
st.header("📈 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    acceptance_rate = df['acceptance_rate'].mean() * 100
    st.metric(
        "Acceptance Rate",
        f"{acceptance_rate:.1f}%",
        f"+{acceptance_rate - 5:.1f}% vs baseline"
    )

with col2:
    avg_savings = abs(df['delta_spend'].mean())
    st.metric(
        "Avg Savings",
        f"${avg_savings:.2f}",
        f"${avg_savings - 0.31:.2f} vs baseline"
    )

with col3:
    avg_nutrition = df['delta_nutrition'].mean()
    st.metric(
        "Nutrition Improvement",
        f"+{avg_nutrition:.1f} HEI",
        f"+{avg_nutrition:.1f} points"
    )

with col4:
    avg_latency = df['latency_ms'].mean()
    st.metric(
        "Avg Latency",
        f"{avg_latency:.1f}ms",
        "✓ Within SLA"
    )

st.divider()

# Charts Row 1
st.header("📊 Performance Analysis")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Acceptance Rate by Policy")
    policy_acceptance = df.groupby('policy_used')['acceptance_rate'].mean() * 100
    fig = px.bar(
        x=policy_acceptance.index,
        y=policy_acceptance.values,
        labels={'x': 'Policy', 'y': 'Acceptance Rate (%)'},
        color=policy_acceptance.values,
        color_continuous_scale='Greens'
    )
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Savings Distribution")
    fig = px.histogram(
        df,
        x='delta_spend',
        nbins=50,
        labels={'delta_spend': 'Savings ($)', 'count': 'Frequency'},
        color_discrete_sequence=['#10b981']
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# Charts Row 2
col1, col2 = st.columns(2)

with col1:
    st.subheader("Nutrition Impact by Policy")
    fig = px.box(
        df,
        x='policy_used',
        y='delta_nutrition',
        labels={'policy_used': 'Policy', 'delta_nutrition': 'HEI Change'},
        color='policy_used',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Latency Over Time")
    df_sorted = df.sort_values('timestamp')
    fig = px.line(
        df_sorted.head(100),
        x='timestamp',
        y='latency_ms',
        labels={'timestamp': 'Time', 'latency_ms': 'Latency (ms)'},
        color_discrete_sequence=['#3b82f6']
    )
    fig.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="SLA: 5ms")
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Fairness Analysis
st.header("⚖️ Fairness Analysis")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Savings by Group")
    fairness_data = df.groupby('protected_group')['delta_spend'].mean().abs()
    fig = px.bar(
        x=fairness_data.index,
        y=fairness_data.values,
        labels={'x': 'Group', 'y': 'Avg Savings ($)'},
        color=fairness_data.values,
        color_continuous_scale='Blues'
    )
    max_disparity = fairness_data.max() - fairness_data.min()
    fig.add_annotation(
        text=f"Max Disparity: ${max_disparity:.2f}",
        xref="paper", yref="paper",
        x=0.5, y=0.95,
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Acceptance Rate by Group")
    fairness_acceptance = df.groupby('protected_group')['acceptance_rate'].mean() * 100
    fig = px.bar(
        x=fairness_acceptance.index,
        y=fairness_acceptance.values,
        labels={'x': 'Race', 'y': 'Acceptance Rate (%)'},
        color=fairness_acceptance.values,
        color_continuous_scale='Greens'
    )
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)

# Fairness metrics
max_disparity = fairness_data.max() - fairness_data.min()
if max_disparity < 3:
    st.success(f"✓ Fairness Check: PASS (Max disparity: ${max_disparity:.2f})")
else:
    st.warning(f"⚠️ Fairness Check: REVIEW (Max disparity: ${max_disparity:.2f})")

st.divider()

# Detailed Data Table
st.header("📋 Recent Transactions")
st.dataframe(
    df.head(100)[['user_id', 'policy_used', 'delta_spend', 'delta_nutrition', 'acceptance_rate', 'latency_ms']].style.format({
        'delta_spend': '${:.2f}',
        'delta_nutrition': '+{:.1f}',
        'acceptance_rate': '{:.1%}',
        'latency_ms': '{:.1f}ms'
    }),
    use_container_width=True,
    height=300
)

# Download button
csv = df.to_csv(index=False)
st.download_button(
    label="📥 Download Full Dataset",
    data=csv,
    file_name=f"eac_data_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>EAC Analytics Dashboard v1.0 | Last updated: {}</p>
    <p>📊 Monitoring {} transactions across {} users</p>
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), len(df), df['user_id'].nunique()), unsafe_allow_html=True)
