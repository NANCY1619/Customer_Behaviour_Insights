import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_animated_sales_trend(data):
    """
    Create an animated line chart showing sales trends over time.
    
    Args:
        data (pandas.DataFrame): Processed customer data
        
    Returns:
        plotly.Figure: Animated sales trend chart
    """
    # Aggregate sales by category and month
    monthly_category_sales = data.groupby(
        [pd.Grouper(key='purchase_date', freq='M'), 'product_category']
    )['transaction_amount'].sum().reset_index()
    
    monthly_category_sales['month'] = monthly_category_sales['purchase_date'].dt.strftime('%b %Y')
    
    # Calculate cumulative sales for each category over time
    cumulative_sales = monthly_category_sales.sort_values('purchase_date')
    cumulative_sales['cumulative_sales'] = cumulative_sales.groupby('product_category')['transaction_amount'].cumsum()
    
    # Create the animated line chart
    fig = px.line(
        cumulative_sales,
        x='month',
        y='cumulative_sales',
        color='product_category',
        animation_frame='purchase_date',
        animation_group='product_category',
        range_y=[0, cumulative_sales['cumulative_sales'].max() * 1.1],
        title='Animated Cumulative Sales by Product Category',
        labels={
            'month': 'Month',
            'cumulative_sales': 'Cumulative Sales ($)',
            'product_category': 'Product Category'
        },
        height=600
    )
    
    # Improve animation settings
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 500
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 300
    
    fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': sorted(monthly_category_sales['month'].unique())},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_animated_customer_growth(data):
    """
    Create an animated chart showing customer growth over time.
    
    Args:
        data (pandas.DataFrame): Processed customer data
        
    Returns:
        plotly.Figure: Animated customer growth chart
    """
    # Get the first purchase date for each customer
    first_purchases = data.groupby('customer_id')['purchase_date'].min().reset_index()
    first_purchases['month'] = first_purchases['purchase_date'].dt.to_period('M')
    
    # Count new customers by month
    new_customers = first_purchases.groupby('month').size().reset_index()
    new_customers.columns = ['month', 'new_customers']
    new_customers['month_str'] = new_customers['month'].dt.strftime('%b %Y')
    
    # Calculate cumulative customer count
    new_customers['cumulative_customers'] = new_customers['new_customers'].cumsum()
    
    # Convert period to datetime for animation
    new_customers['month_date'] = new_customers['month'].dt.to_timestamp()
    
    # Create the animation
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart for new customers
    fig.add_trace(
        go.Bar(
            x=new_customers['month_str'],
            y=new_customers['new_customers'],
            name='New Customers',
            marker_color='rgba(58, 71, 80, 0.6)',
            customdata=new_customers['month_date']
        ),
        secondary_y=False
    )
    
    # Add line chart for cumulative customers
    fig.add_trace(
        go.Scatter(
            x=new_customers['month_str'],
            y=new_customers['cumulative_customers'],
            name='Total Customer Base',
            mode='lines+markers',
            line=dict(width=3, color='firebrick'),
            customdata=new_customers['month_date']
        ),
        secondary_y=True
    )
    
    # Create animation frames
    frames = []
    for i in range(1, len(new_customers) + 1):
        frame = go.Frame(
            data=[
                go.Bar(
                    x=new_customers['month_str'][:i],
                    y=new_customers['new_customers'][:i],
                    marker_color='rgba(58, 71, 80, 0.6)'
                ),
                go.Scatter(
                    x=new_customers['month_str'][:i],
                    y=new_customers['cumulative_customers'][:i],
                    mode='lines+markers',
                    line=dict(width=3, color='firebrick')
                )
            ],
            name=new_customers['month_str'][i-1]
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Setup animation
    animation_settings = dict(
        frame=dict(duration=500, redraw=True),
        fromcurrent=True,
        transition=dict(duration=300, easing="cubic-in-out")
    )
    
    # Add play button
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, animation_settings]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=True), mode="immediate")]
                )
            ],
            direction="left",
            pad=dict(r=10, t=10),
            showactive=False,
            x=0.1,
            xanchor="right",
            y=0,
            yanchor="top"
        )]
    )
    
    # Add slider
    sliders = [{
        "active": len(frames) - 1,
        "steps": [
            {
                "args": [
                    [f.name],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }
                ],
                "label": f.name,
                "method": "animate"
            }
            for f in frames
        ],
        "x": 0.1,
        "y": 0,
        "len": 0.9,
        "xanchor": "left",
        "yanchor": "top"
    }]
    
    fig.update_layout(
        sliders=sliders,
        title='Customer Acquisition Growth Over Time',
        xaxis_title='Month',
        yaxis_title='New Customers',
        yaxis2_title='Total Customer Base',
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis2=dict(title='Total Customer Base', titlefont=dict(color='firebrick'), tickfont=dict(color='firebrick')),
        yaxis=dict(title='New Customers', titlefont=dict(color='rgba(58, 71, 80, 0.6)'), tickfont=dict(color='rgba(58, 71, 80, 0.6)'))
    )
    
    return fig


def create_animated_customer_segments(data, segment_col='segment_label'):
    """
    Create an animated pie chart showing how customer segments evolved.
    
    Args:
        data (pandas.DataFrame): Customer data with segment information
        segment_col (str): Column name containing segment labels
        
    Returns:
        plotly.Figure: Animated segment evolution chart
    """
    # Check if data has segment column
    if segment_col not in data.columns:
        # Create a placeholder chart with an error message
        fig = go.Figure()
        fig.add_annotation(
            text="Segment data not available. Please perform segmentation first.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Check if data has purchase_date
    if 'purchase_date' not in data.columns:
        # Create a placeholder chart with an error message
        fig = go.Figure()
        fig.add_annotation(
            text="Purchase date information not available.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Prepare data for animation
    data['month'] = data['purchase_date'].dt.to_period('M')
    data['month_str'] = data['purchase_date'].dt.strftime('%b %Y')
    
    # Count customers by segment and month
    segment_counts = data.groupby(['month', segment_col])['customer_id'].nunique().reset_index()
    segment_counts.columns = ['month', 'segment', 'customer_count']
    
    # Sort by month for animation
    unique_months = sorted(segment_counts['month'].unique())
    month_to_date = {month: month.to_timestamp() for month in unique_months}
    
    # Create initial pie chart
    fig = px.pie(
        segment_counts[segment_counts['month'] == unique_months[0]],
        values='customer_count',
        names='segment',
        title=f'Customer Segment Distribution - {unique_months[0].strftime("%b %Y")}',
        hole=0.4
    )
    
    # Create animation frames
    frames = []
    for month in unique_months:
        month_data = segment_counts[segment_counts['month'] == month]
        frame = go.Frame(
            data=[go.Pie(
                labels=month_data['segment'],
                values=month_data['customer_count'],
                hole=0.4
            )],
            name=month.strftime('%b %Y'),
            layout=go.Layout(
                title_text=f'Customer Segment Distribution - {month.strftime("%b %Y")}'
            )
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Setup animation
    animation_settings = dict(
        frame=dict(duration=500, redraw=True),
        fromcurrent=True,
        transition=dict(duration=300, easing="cubic-in-out")
    )
    
    # Add play button
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, animation_settings]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=True), mode="immediate")]
                )
            ],
            direction="left",
            pad=dict(r=10, t=10),
            showactive=False,
            x=0.1,
            xanchor="right",
            y=0,
            yanchor="top"
        )]
    )
    
    # Add slider
    sliders = [{
        "active": 0,
        "steps": [
            {
                "args": [
                    [f.name],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }
                ],
                "label": f.name,
                "method": "animate"
            }
            for f in frames
        ]
    }]
    
    fig.update_layout(
        sliders=sliders,
        height=600
    )
    
    return fig


def create_geographic_animation(data):
    """
    Create an animated bubble map showing sales by location over time.
    
    Args:
        data (pandas.DataFrame): Processed customer data
        
    Returns:
        plotly.Figure: Animated geographic sales chart
    """
    # Check if data has location information
    if 'latitude' not in data.columns or 'longitude' not in data.columns:
        # Generate synthetic location data for demonstration
        num_locations = 30
        np.random.seed(42)  # For reproducibility
        
        locations = pd.DataFrame({
            'location_id': range(1, num_locations + 1),
            'location_name': [f'Region {i}' for i in range(1, num_locations + 1)],
            'latitude': np.random.uniform(25, 50, num_locations),
            'longitude': np.random.uniform(-125, -70, num_locations)
        })
        
        # Assign locations to customers
        if 'customer_id' in data.columns:
            unique_customers = data['customer_id'].unique()
            customer_locations = pd.DataFrame({
                'customer_id': unique_customers,
                'location_id': np.random.choice(locations['location_id'], len(unique_customers))
            })
            
            # Merge locations with transactions
            data = data.merge(customer_locations, on='customer_id')
            data = data.merge(locations, on='location_id')
    
    # Prepare data for animation
    data['month'] = data['purchase_date'].dt.to_period('M')
    data['month_str'] = data['purchase_date'].dt.strftime('%b %Y')
    
    # Aggregate sales by location and month
    location_sales = data.groupby(['month', 'latitude', 'longitude'])['transaction_amount'].sum().reset_index()
    
    # Create choropleth map animation
    fig = px.scatter_geo(
        location_sales,
        lat='latitude',
        lon='longitude',
        size='transaction_amount',
        animation_frame='month',
        color='transaction_amount',
        color_continuous_scale='Viridis',
        scope='usa',  # Adjust if data is for a different region
        title='Geographic Sales Distribution Over Time',
        projection='albers usa',  # Adjust if data is for a different region
        labels={
            'transaction_amount': 'Sales Amount ($)',
            'month': 'Month'
        },
        height=600
    )
    
    # Improve animation settings
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 800
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 300
    
    return fig