import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def perform_rfm_analysis(data):
    """
    Perform RFM (Recency, Frequency, Monetary) analysis on customer data.
    
    Args:
        data (pandas.DataFrame): Processed customer data
        
    Returns:
        pandas.DataFrame: Customer data with RFM scores and segments
    """
    # Make a copy of the data to avoid modifying the original
    rfm_data = data.copy()
    
    # Calculate RFM metrics
    # Get the maximum date to calculate recency
    max_date = rfm_data['purchase_date'].max()
    
    # Group by customer_id and calculate RFM metrics
    rfm = rfm_data.groupby('customer_id').agg({
        'purchase_date': lambda x: (max_date - x.max()).days,  # Recency
        'order_id': 'nunique',  # Frequency
        'transaction_amount': 'sum'  # Monetary
    }).reset_index()
    
    # Rename columns
    rfm.columns = ['customer_id', 'recency_days', 'frequency', 'monetary']
    
    # Calculate RFM quartiles
    rfm['R_Quartile'] = pd.qcut(rfm['recency_days'], 4, labels=False, duplicates='drop')
    rfm['F_Quartile'] = pd.qcut(rfm['frequency'], 4, labels=False, duplicates='drop')
    rfm['M_Quartile'] = pd.qcut(rfm['monetary'], 4, labels=False, duplicates='drop')
    
    # Adjust recency score (lower is better for recency)
    rfm['R_Score'] = 4 - rfm['R_Quartile']
    rfm['F_Score'] = rfm['F_Quartile']
    rfm['M_Score'] = rfm['M_Quartile']
    
    # Calculate RFM Score
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    
    # Create RFM segments
    rfm['RFM_Segment'] = rfm.apply(assign_rfm_segment, axis=1)
    
    # Additional metrics for analysis
    rfm['days_since_last_purchase'] = rfm['recency_days']
    rfm['avg_purchase_frequency'] = rfm['frequency'] / 12  # assuming 12 months of data
    rfm['total_revenue'] = rfm['monetary']
    
    return rfm


def assign_rfm_segment(row):
    """
    Assign RFM segment based on R, F, and M scores.
    
    Args:
        row (pandas.Series): Row with R_Score, F_Score, and M_Score
        
    Returns:
        str: RFM segment name
    """
    # Convert scores to integers
    r = int(row['R_Score'])
    f = int(row['F_Score'])
    m = int(row['M_Score'])
    
    # Champions: Customers who bought recently, buy often and spend the most
    if r >= 3 and f >= 3 and m >= 3:
        return "Champions"
    
    # Loyal Customers: Customers who buy on a regular basis
    elif r >= 2 and f >= 3 and m >= 2:
        return "Loyal Customers"
    
    # Promising: Recent customers with average frequency
    elif r >= 3 and f >= 1 and m >= 1:
        return "Promising"
    
    # New Customers: Bought most recently, but not often
    elif r >= 3 and f <= 1 and m <= 1:
        return "New Customers"
    
    # At Risk: Above average recency, frequency and monetary values
    # Haven't purchased recently though
    elif r <= 1 and f >= 2 and m >= 2:
        return "At Risk"
    
    # Can't Lose Them: Used to purchase frequently but haven't returned for a long time
    elif r <= 1 and f >= 3 and m >= 3:
        return "Can't Lose Them"
    
    # Hibernating: Last purchase was long back, purchased few times
    elif r <= 1 and f <= 2 and m <= 2:
        return "Hibernating"
    
    # About to Sleep: Below average recency, frequency and monetary values
    elif r <= 2 and f <= 2 and m <= 2:
        return "About to Sleep"
    
    # Need Attention: Haven't purchased for some time, but used to purchase frequently
    elif r <= 2 and f >= 2 and m <= 2:
        return "Need Attention"
    
    # Potential Loyalists: Recent customers with decent spending
    elif r >= 2 and f <= 2 and m >= 2:
        return "Potential Loyalists"
    
    # Others
    else:
        return "Others"


def plot_rfm_analysis(rfm_data):
    """
    Create visualizations for RFM analysis.
    
    Args:
        rfm_data (pandas.DataFrame): Customer data with RFM scores and segments
        
    Returns:
        list: List of plotly figures
    """
    figures = []
    
    # 1. Segment Distribution
    segment_counts = rfm_data['RFM_Segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    
    # Sort by count
    segment_counts = segment_counts.sort_values('Count', ascending=False)
    
    fig1 = px.bar(
        segment_counts,
        x='Segment',
        y='Count',
        color='Segment',
        title='Customer Segment Distribution',
        labels={'Count': 'Number of Customers', 'Segment': 'RFM Segment'},
        height=500
    )
    
    fig1.update_layout(xaxis={'categoryorder': 'total descending'})
    figures.append(fig1)
    
    # 2. RFM Score Distribution (3D Scatter Plot)
    fig2 = px.scatter_3d(
        rfm_data,
        x='recency_days',
        y='frequency',
        z='monetary',
        color='RFM_Segment',
        opacity=0.7,
        title='3D RFM Distribution',
        labels={
            'recency_days': 'Recency (days)',
            'frequency': 'Frequency (orders)',
            'monetary': 'Monetary (total spend)'
        },
        height=700
    )
    
    fig2.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    figures.append(fig2)
    
    # 3. Segment Characteristics Heatmap
    segment_metrics = rfm_data.groupby('RFM_Segment').agg({
        'recency_days': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'R_Score': 'mean',
        'F_Score': 'mean',
        'M_Score': 'mean'
    }).reset_index()
    
    # Create a heatmap of segment metrics
    fig3 = make_subplots(
        rows=1, cols=3, 
        subplot_titles=("Recency (days)", "Frequency (orders)", "Monetary (spend)"),
        shared_yaxes=True
    )
    
    # Sort segments by count
    segment_order = segment_counts['Segment'].tolist()
    segment_metrics['order'] = segment_metrics['RFM_Segment'].map({seg: i for i, seg in enumerate(segment_order)})
    segment_metrics = segment_metrics.sort_values('order')
    
    # Recency heatmap
    fig3.add_trace(
        go.Heatmap(
            z=[segment_metrics['recency_days']],
            x=segment_metrics['RFM_Segment'],
            colorscale='Blues_r',  # Reverse scale as lower is better for recency
            showscale=False
        ),
        row=1, col=1
    )
    
    # Frequency heatmap
    fig3.add_trace(
        go.Heatmap(
            z=[segment_metrics['frequency']],
            x=segment_metrics['RFM_Segment'],
            colorscale='Greens',
            showscale=False
        ),
        row=1, col=2
    )
    
    # Monetary heatmap
    fig3.add_trace(
        go.Heatmap(
            z=[segment_metrics['monetary']],
            x=segment_metrics['RFM_Segment'],
            colorscale='Reds',
            showscale=True
        ),
        row=1, col=3
    )
    
    fig3.update_layout(
        title_text="Segment Characteristics Heatmap",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    # Update x-axis labels with 45-degree rotation
    fig3.update_xaxes(tickangle=45)
    figures.append(fig3)
    
    # 4. RFM Metric Distributions by Segment
    fig4 = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Recency by Segment", "Frequency by Segment", "Monetary by Segment"),
        shared_yaxes=True
    )
    
    metrics = ['recency_days', 'frequency', 'monetary']
    cols = [1, 2, 3]
    colors = ['blue', 'green', 'red']
    
    for metric, col, color in zip(metrics, cols, colors):
        boxplot = go.Box(
            y=rfm_data[metric],
            x=rfm_data['RFM_Segment'],
            name=metric,
            marker_color=color
        )
        fig4.add_trace(boxplot, row=1, col=col)
    
    fig4.update_layout(
        title_text="RFM Metric Distributions by Segment",
        height=500,
        showlegend=False,
        boxmode='group'
    )
    
    # Update x-axis labels with 45-degree rotation
    fig4.update_xaxes(tickangle=45)
    figures.append(fig4)
    
    return figures