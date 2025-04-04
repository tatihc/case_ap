# import warnings
# import logging
import streamlit as st
import pandas as pd
# import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import datetime, timedelta

# Configure logging to suppress specific warnings
# logging.getLogger('streamlit').setLevel(logging.ERROR)

# # Filter warnings more broadly
# warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
# warnings.filterwarnings("ignore", message=".*No runtime found.*")
# warnings.filterwarnings("ignore", message=".*Session state.*")


# Page setup
st.set_page_config(
    page_title="Fall 2025 Collection Dashboard",
    page_icon="🍂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data loading function
@st.cache_data
def load_data():
    campaigns = pd.read_csv('data/campaigns.csv')
    ads = pd.read_csv('data/ads.csv')
    performance = pd.read_csv('data/daily_performance.csv')
    products = pd.read_csv('data/products.csv')

    # Converting dates
    performance['date'] = pd.to_datetime(performance['date'])
    campaigns['start_date'] = pd.to_datetime(campaigns['start_date'])
    campaigns['end_date'] = pd.to_datetime(campaigns['end_date'])

    # Gather data for analysis
    perf_with_ads = performance.merge(ads, on='ad_id')
    full_data = perf_with_ads.merge(campaigns, on='campaign_id')
    full_data = full_data.merge(products, on='product_id')

    # Calculate additional metrics
    full_data['ctr'] = full_data['clicks'] / full_data['impressions']
    full_data['cpa'] = full_data.apply(lambda x: x['cost'] / x['conversions'] if x['conversions'] > 0 else 0, axis=1)
    full_data['roas'] = full_data.apply(lambda x: x['revenue'] / x['cost'] if x['cost'] > 0 else 0, axis=1)

    return campaigns, ads, performance, products, full_data

# Load data
campaigns, ads, performance, products, full_data = load_data()

# Sidebar for filters
st.sidebar.title("Filters")

# Data filter
min_date = full_data['date'].min().date()
max_date = full_data['date'].max().date()
date_range = st.sidebar.date_input(
    "Period",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_data = full_data[(full_data['date'].dt.date >= start_date) &
                             (full_data['date'].dt.date <= end_date)]
else:
    filtered_data = full_data

# Channel filter
channel_options = ['All'] + list(filtered_data['channel'].unique())
selected_channel = st.sidebar.selectbox("Channel", channel_options)

if selected_channel != 'All':
    filtered_data = filtered_data[filtered_data['channel'] == selected_channel]

# Goal filter
objective_options = ['All'] + list(filtered_data['objective'].unique())
selected_objective = st.sidebar.selectbox("Goal", objective_options)

if selected_objective != 'All':
    filtered_data = filtered_data[filtered_data['objective'] == selected_objective]

# Product category filter
category_options = ['All'] + list(filtered_data['product_category'].unique())
selected_category = st.sidebar.selectbox("Product Category", category_options)

if selected_category != 'All':
    filtered_data = filtered_data[filtered_data['product_category'] == selected_category]

# Dashboard title
st.title("Launching Dashboard - Fall Collection 2025")
st.markdown("Performance analysis of paid digital channels for the launch of the Fall 2025 collection")

# Main Metrics
st.header("Main Metrics")

# Calculate total metrics
total_impressions = filtered_data['impressions'].sum()
total_clicks = filtered_data['clicks'].sum()
total_cost = filtered_data['cost'].sum()
total_conversions = filtered_data['conversions'].sum()
total_revenue = filtered_data['revenue'].sum()
avg_ctr = total_clicks / total_impressions if total_impressions > 0 else 0
avg_cpa = total_cost / total_conversions if total_conversions > 0 else 0
avg_roas = total_revenue / total_cost if total_cost > 0 else 0

# Show metrics in columns
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Impressions", f"{total_impressions:,.0f}")
    st.metric("Clicks", f"{total_clicks:,.0f}")
with col2:
    st.metric("CTR", f"{avg_ctr:.2%}")
    st.metric("Cost", f"${total_cost:,.2f}")
with col3:
    st.metric("Conversions", f"{total_conversions:,.0f}")
    st.metric("CPA", f"${avg_cpa:.2f}")
with col4:
    st.metric("Revenue", f"${total_revenue:,.2f}")
    st.metric("ROAS", f"{avg_roas:.2f}x")

# Trend over time
st.header("Trend over time")

# Gather data by time
daily_trends = filtered_data.groupby('date').agg({
    'impressions': 'sum',
    'clicks': 'sum',
    'cost': 'sum',
    'conversions': 'sum',
    'revenue': 'sum'
}).reset_index()

daily_trends['ctr'] = daily_trends['clicks'] / daily_trends['impressions']
daily_trends['cpa'] = daily_trends.apply(lambda x: x['cost'] / x['conversions'] if x['conversions'] > 0 else 0, axis=1)
daily_trends['roas'] = daily_trends.apply(lambda x: x['revenue'] / x['cost'] if x['cost'] > 0 else 0, axis=1)

# Create trend chart
tab1, tab2, tab3 = st.tabs(["Engagement", "Conversion", "ROI"])

with tab1:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=daily_trends['date'], y=daily_trends['impressions'], name="Impressions"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=daily_trends['date'], y=daily_trends['clicks'], name="Clicks", mode="lines+markers"),
        secondary_y=True,
    )

    fig.update_layout(
        title_text="Impressions and Clicks over time",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(title_text="Impressions", secondary_y=False)
    fig.update_yaxes(title_text="Clicks", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # CTR aover time
    fig_ctr = px.line(daily_trends, x='date', y='ctr', title='CTR Over Time')
    fig_ctr.update_traces(mode='lines+markers')
    fig_ctr.update_layout(yaxis_tickformat='.2%')
    st.plotly_chart(fig_ctr, use_container_width=True)

with tab2:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=daily_trends['date'], y=daily_trends['conversions'], name="Conversions"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=daily_trends['date'], y=daily_trends['cpa'], name="CPA", mode="lines+markers"),
        secondary_y=True,
    )

    fig.update_layout(
        title_text="Conversions and CPA Over Time",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(title_text="Conversions", secondary_y=False)
    fig.update_yaxes(title_text="CPA ($)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=daily_trends['date'], y=daily_trends['revenue'], name="Revenue"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=daily_trends['date'], y=daily_trends['roas'], name="ROAS", mode="lines+markers"),
        secondary_y=True,
    )

    fig.update_layout(
        title_text="Revenue and ROAS Over Time",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
    fig.update_yaxes(title_text="ROAS", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

# Channel comparison
st.header("Channel comparison")

# Group data by channel
channel_comparison = filtered_data.groupby('channel').agg({
    'impressions': 'sum',
    'clicks': 'sum',
    'cost': 'sum',
    'conversions': 'sum',
    'revenue': 'sum'
}).reset_index()

channel_comparison['ctr'] = channel_comparison['clicks'] / channel_comparison['impressions']
channel_comparison['cpa'] = channel_comparison.apply(lambda x: x['cost'] / x['conversions'] if x['conversions'] > 0 else 0, axis=1)
channel_comparison['roas'] = channel_comparison.apply(lambda x: x['revenue'] / x['cost'] if x['cost'] > 0 else 0, axis=1)

# Create charts to compare
col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        channel_comparison,
        x='channel',
        y=['impressions', 'clicks', 'conversions'],
        title='Volume by Channel',
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(
        channel_comparison,
        x='channel',
        y=['ctr', 'cpa', 'roas'],
        title='Channel Efficiency',
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)

# Product Analysis
st.header("Product Analysis")

# Group data by product
product_performance = filtered_data.groupby(['product_id', 'product_name', 'product_category']).agg({
    'impressions': 'sum',
    'clicks': 'sum',
    'cost': 'sum',
    'conversions': 'sum',
    'revenue': 'sum'
}).reset_index()

product_performance['ctr'] = product_performance['clicks'] / product_performance['impressions']
product_performance['cpa'] = product_performance.apply(lambda x: x['cost'] / x['conversions'] if x['conversions'] > 0 else 0, axis=1)
product_performance['roas'] = product_performance.apply(lambda x: x['revenue'] / x['cost'] if x['cost'] > 0 else 0, axis=1)

# Order by conversions
top_products = product_performance.sort_values('conversions', ascending=False).head(10)

# Create top sellers chart
fig = px.bar(
    top_products,
    x='product_name',
    y='conversions',
    color='product_category',
    title='Top 10 Products by Conversions',
    labels={'product_name': 'Product', 'conversions': 'Conversions', 'product_category': 'Category'}
)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

# Category Analysis
category_performance = filtered_data.groupby('product_category').agg({
    'impressions': 'sum',
    'clicks': 'sum',
    'cost': 'sum',
    'conversions': 'sum',
    'revenue': 'sum'
}).reset_index()

category_performance['ctr'] = category_performance['clicks'] / category_performance['impressions']
category_performance['cpa'] = category_performance.apply(lambda x: x['cost'] / x['conversions'] if x['conversions'] > 0 else 0, axis=1)
category_performance['roas'] = category_performance.apply(lambda x: x['revenue'] / x['cost'] if x['cost'] > 0 else 0, axis=1)

# Create performance chart by category
col1, col2 = st.columns(2)

with col1:
    fig = px.pie(
        category_performance,
        values='revenue',
        names='product_category',
        title='Revenue Distribution by Category'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(
        category_performance,
        x='product_category',
        y='roas',
        title='ROAS by Category',
        color='product_category'
    )
    st.plotly_chart(fig, use_container_width=True)

# Funnel Analysis
st.header("Funnel Analysis")

# Group data by goal
funnel_data = filtered_data.groupby('objective').agg({
    'impressions': 'sum',
    'clicks': 'sum',
    'conversions': 'sum',
    'cost': 'sum',
    'revenue': 'sum'
}).reset_index()

# Order by funnel stage
funnel_order = {'Awareness': 0, 'Consideration': 1, 'Conversion': 2}
funnel_data['order'] = funnel_data['objective'].map(funnel_order)
funnel_data = funnel_data.sort_values('order')

# Creat funnel chart
fig = go.Figure()

fig.add_trace(go.Funnel(
    name = 'Marketing Funnel',
    y = funnel_data['objective'],
    x = funnel_data['impressions'],
    textposition = "inside",
    textinfo = "value+percent initial",
    opacity = 0.65,
    marker = {"color": ["#4B878B", "#D01257", "#01295F"]}, 
    connector = {"line": {"color": "royalblue", "width": 1}}
))

fig.update_layout(
    title = "Marketing Funnel - Impressions",
    font_size = 15
)

st.plotly_chart(fig, use_container_width=True)

# Metrics by funnel stage
st.subheader("Metrics by Funnel Stage")
st.dataframe(
    funnel_data[['objective', 'impressions', 'clicks', 'conversions', 'cost', 'revenue']].style.format({
        'impressions': '{:,.0f}',
        'clicks': '{:,.0f}',
        'conversions': '{:,.0f}',
        'cost': '${:,.2f}',
        'revenue': '${:,.2f}'
    }),
    use_container_width=True
)


# Footer
st.markdown("---")
st.markdown("Dashboard developed to analyze the Fall 2025 collection | Updated on: " + datetime.now().strftime("%d/%m/%Y %H:%M"))