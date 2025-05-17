import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_sales_trend_chart(data):
    """
    Create a line chart showing sales trends over time.
    
    Args:
        data (pandas.DataFrame): Processed customer data
        
    Returns:
        plotly.Figure: Sales trend chart
    """
    # Aggregate data by month
    sales_by_month = data.groupby(pd.Grouper(key='purchase_date', freq='M'))['transaction_amount'].sum().reset_index()
    sales_by_month['month'] = sales_by_month['purchase_date'].dt.strftime('%b %Y')
    
    # Create line chart
    fig = px.line(
        sales_by_month, 
        x='month', 
        y='transaction_amount',
        markers=True,
        title='Monthly Sales Trend',
        labels={'transaction_amount': 'Sales Amount ($)', 'month': 'Month'}
    )
    
    # Add moving average
    window_size = min(3, len(sales_by_month))
    if window_size > 1:
        sales_by_month['moving_avg'] = sales_by_month['transaction_amount'].rolling(window=window_size).mean()
        
        fig.add_trace(
            go.Scatter(
                x=sales_by_month['month'],
                y=sales_by_month['moving_avg'],
                mode='lines',
                name=f'{window_size}-Month Moving Average',
                line=dict(color='rgba(255, 0, 0, 0.5)', width=3)
            )
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Sales ($)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_purchase_frequency_chart(data):
    """
    Create a histogram showing distribution of purchase frequency.
    
    Args:
        data (pandas.DataFrame): Processed customer data
        
    Returns:
        plotly.Figure: Purchase frequency histogram
    """
    # Get unique customers
    customer_data = data.drop_duplicates(subset=['customer_id'])
    
    # Create histogram
    fig = px.histogram(
        customer_data,
        x='purchase_frequency',
        nbins=20,
        title='Distribution of Customer Purchase Frequency',
        labels={'purchase_frequency': 'Purchases per Month'},
        color_discrete_sequence=['#3366CC']
    )
    
    # Add vertical line for the mean
    mean_frequency = customer_data['purchase_frequency'].mean()
    
    fig.add_vline(
        x=mean_frequency,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_frequency:.2f}",
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Purchases per Month',
        yaxis_title='Number of Customers',
        bargap=0.1
    )
    
    return fig

def create_customer_lifetime_value_chart(data):
    """
    Create a chart showing distribution of customer lifetime value.
    
    Args:
        data (pandas.DataFrame): Processed customer data
        
    Returns:
        plotly.Figure: CLV distribution chart
    """
    # Get unique customers
    customer_data = data.drop_duplicates(subset=['customer_id'])
    
    # Sort by CLV
    sorted_data = customer_data.sort_values('customer_lifetime_value', ascending=False)
    
    # Create a Pareto chart (80/20 rule visualization)
    sorted_data = sorted_data.reset_index(drop=True)
    sorted_data['cumulative_percentage'] = sorted_data['customer_lifetime_value'].cumsum() / sorted_data['customer_lifetime_value'].sum() * 100
    sorted_data['customer_rank'] = sorted_data.index + 1
    sorted_data['customer_percentile'] = sorted_data['customer_rank'] / len(sorted_data) * 100
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bars for CLV
    fig.add_trace(
        go.Bar(
            x=sorted_data['customer_percentile'],
            y=sorted_data['customer_lifetime_value'],
            name="Customer Lifetime Value",
            marker_color='rgb(55, 83, 109)'
        ),
        secondary_y=False,
    )
    
    # Add line for cumulative percentage
    fig.add_trace(
        go.Scatter(
            x=sorted_data['customer_percentile'],
            y=sorted_data['cumulative_percentage'],
            name="Cumulative % of Revenue",
            marker_color='rgb(26, 118, 255)',
            mode='lines'
        ),
        secondary_y=True,
    )
    
    # Add 80% reference line
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="red",
        line_width=1,
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title_text="Customer Lifetime Value Distribution (Pareto Analysis)",
        xaxis_title="Customer Percentile (%)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_yaxes(title_text="Customer Lifetime Value ($)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative % of Revenue", secondary_y=True)
    
    return fig

def create_product_category_chart(data):
    """
    Create a chart showing product category distribution.
    
    Args:
        data (pandas.DataFrame): Processed customer data
        
    Returns:
        plotly.Figure: Product category chart
    """
    # Aggregate data by product category
    category_data = data.groupby('product_category').agg(
        total_sales=('transaction_amount', 'sum'),
        transaction_count=('transaction_amount', 'count'),
        unique_customers=('customer_id', 'nunique')
    ).reset_index()
    
    # Sort by total sales
    category_data = category_data.sort_values('total_sales', ascending=False)
    
    # Create grouped bar chart
    fig = go.Figure()
    
    # Add trace for sales amount
    fig.add_trace(go.Bar(
        x=category_data['product_category'],
        y=category_data['total_sales'],
        name='Total Sales ($)',
        marker_color='#3366CC'
    ))
    
    # Add trace for transaction count
    fig.add_trace(go.Bar(
        x=category_data['product_category'],
        y=category_data['transaction_count'],
        name='Transaction Count',
        marker_color='#DC3912'
    ))
    
    # Add trace for unique customers
    fig.add_trace(go.Bar(
        x=category_data['product_category'],
        y=category_data['unique_customers'],
        name='Unique Customers',
        marker_color='#FF9900'
    ))
    
    # Update layout
    fig.update_layout(
        title='Product Category Analysis',
        xaxis_title='Product Category',
        yaxis_title='Value',
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_customer_journey_chart(data):
    """
    Create a visualization of the customer journey.
    
    Args:
        data (pandas.DataFrame): Processed customer data
        
    Returns:
        plotly.Figure: Customer journey chart
    """
    # Calculate average time between purchases
    purchase_intervals = []
    
    for customer in data['customer_id'].unique():
        customer_purchases = data[data['customer_id'] == customer].sort_values('purchase_date')
        
        if len(customer_purchases) > 1:
            # Calculate time differences between consecutive purchases
            purchase_dates = customer_purchases['purchase_date'].tolist()
            intervals = [(purchase_dates[i+1] - purchase_dates[i]).days for i in range(len(purchase_dates)-1)]
            purchase_intervals.extend(intervals)
    
    # Create histogram of purchase intervals
    if purchase_intervals:
        # Filter out extremely large values (outliers)
        purchase_intervals = [interval for interval in purchase_intervals if interval <= 100]
        
        fig = px.histogram(
            x=purchase_intervals,
            nbins=20,
            title='Time Between Customer Purchases',
            labels={'x': 'Days Between Purchases'},
            color_discrete_sequence=['#3366CC']
        )
        
        # Add mean line
        mean_interval = np.mean(purchase_intervals)
        median_interval = np.median(purchase_intervals)
        
        fig.add_vline(
            x=mean_interval,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_interval:.1f} days",
            annotation_position="top right"
        )
        
        fig.add_vline(
            x=median_interval,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Median: {median_interval:.1f} days",
            annotation_position="top left"
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Days Between Purchases',
            yaxis_title='Frequency',
            bargap=0.1
        )
        
        return fig
    else:
        # Create empty figure with a message if no data
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text="Not enough purchase history to analyze customer journey",
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            xaxis_visible=False,
            yaxis_visible=False,
            title='Customer Journey Analysis'
        )
        return fig

def create_cohort_analysis(data):
    """
    Create a cohort analysis heatmap.
    
    Args:
        data (pandas.DataFrame): Processed customer data
        
    Returns:
        plotly.Figure: Cohort analysis heatmap
    """
    # Prepare data for cohort analysis
    cohort_data = data.copy()
    
    # Ensure the purchase date is in datetime format
    cohort_data['purchase_date'] = pd.to_datetime(cohort_data['purchase_date'])
    
    # Create cohort groups based on the first purchase month
    cohort_data['cohort_month'] = cohort_data.groupby('customer_id')['purchase_date'].transform('min').dt.to_period('M')
    cohort_data['purchase_month'] = cohort_data['purchase_date'].dt.to_period('M')
    
    # Calculate the month index (how many months from the first purchase)
    cohort_data['month_index'] = ((cohort_data['purchase_month'].dt.year - cohort_data['cohort_month'].dt.year) * 12 + 
                                 (cohort_data['purchase_month'].dt.month - cohort_data['cohort_month'].dt.month))
    
    # Count the number of unique customers in each cohort and month index
    cohort_counts = cohort_data.groupby(['cohort_month', 'month_index']).agg(
        customer_count=('customer_id', 'nunique')
    ).reset_index()
    
    # Get the initial size of each cohort
    initial_cohort_size = cohort_counts[cohort_counts['month_index'] == 0].copy()
    initial_cohort_size.rename(columns={'customer_count': 'initial_size'}, inplace=True)
    initial_cohort_size.drop('month_index', axis=1, inplace=True)
    
    # Merge with the cohort counts
    cohort_retention = cohort_counts.merge(initial_cohort_size, on='cohort_month')
    
    # Calculate the retention rate
    cohort_retention['retention_rate'] = (cohort_retention['customer_count'] / cohort_retention['initial_size'] * 100).round(1)
    
    # Pivot the data for the heatmap
    cohort_pivot = cohort_retention.pivot_table(
        index='cohort_month',
        columns='month_index',
        values='retention_rate'
    )
    
    # Convert the index to string for better display
    cohort_pivot.index = cohort_pivot.index.astype(str)
    
    # Limit to first 12 months for visibility
    months_to_show = min(12, cohort_pivot.shape[1])
    cohort_pivot = cohort_pivot.iloc[:, :months_to_show]
    
    # Limit to top 12 cohorts for visibility
    cohorts_to_show = min(12, cohort_pivot.shape[0])
    cohort_pivot = cohort_pivot.iloc[-cohorts_to_show:, :]
    
    # Create the heatmap
    fig = px.imshow(
        cohort_pivot.values,
        labels=dict(x="Month Index", y="Cohort", color="Retention Rate (%)"),
        x=[f"Month {i}" for i in cohort_pivot.columns],
        y=[str(cohort) for cohort in cohort_pivot.index],
        color_continuous_scale="RdYlGn",
        origin='lower',
        aspect="auto",
        title="Cohort Analysis: Customer Retention Rate (%)"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Months Since First Purchase",
        yaxis_title="Customer Cohort (First Purchase Month)",
        coloraxis_colorbar=dict(title="Retention Rate (%)"),
    )
    
    # Add text annotations with retention rates
    for i, cohort in enumerate(cohort_pivot.index):
        for j, month in enumerate(cohort_pivot.columns):
            value = cohort_pivot.iloc[i, j]
            if not np.isnan(value):
                text_color = "black" if value > 50 else "white"
                fig.add_annotation(
                    x=month,
                    y=cohort,
                    text=f"{value:.1f}%",
                    showarrow=False,
                    font=dict(color=text_color)
                )
    
    return fig
