import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots

def perform_customer_segmentation(data, features, num_segments=3):
    """
    Perform customer segmentation using KMeans clustering.
    
    Args:
        data (pandas.DataFrame): Processed customer data
        features (list): Features to use for segmentation
        num_segments (int): Number of segments to create
        
    Returns:
        dict: Dictionary with segmentation results
    """
    # Get unique customers with their features
    customer_data = data.drop_duplicates(subset=['customer_id'])
    
    # Select features for clustering
    X = customer_data[features].copy()
    
    # Handle any missing values
    X = X.fillna(X.mean())
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_segments, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to the customer data
    customer_segments = customer_data.copy()
    customer_segments['segment'] = clusters
    
    # Create profile for each segment
    segment_profiles = customer_segments.groupby('segment').agg(
        customer_count=('customer_id', 'nunique'),
        avg_order_value=('avg_order_value', 'mean'),
        avg_purchase_frequency=('purchase_frequency', 'mean'),
        avg_clv=('customer_lifetime_value', 'mean'),
        avg_recency=('recency_days', 'mean')
    ).reset_index()
    
    # Add percentage of total customers
    segment_profiles['customer_percentage'] = (segment_profiles['customer_count'] / 
                                              segment_profiles['customer_count'].sum() * 100).round(1)
    
    # Sort by average CLV to assign meaningful labels
    segment_profiles = segment_profiles.sort_values('avg_clv', ascending=False)
    
    # Create segment labels based on CLV and frequency
    segment_labels = []
    for i, row in segment_profiles.iterrows():
        if row['avg_clv'] > segment_profiles['avg_clv'].median() and row['avg_purchase_frequency'] > segment_profiles['avg_purchase_frequency'].median():
            label = "High Value & Frequent"
        elif row['avg_clv'] > segment_profiles['avg_clv'].median():
            label = "High Value & Infrequent"
        elif row['avg_purchase_frequency'] > segment_profiles['avg_purchase_frequency'].median():
            label = "Low Value & Frequent"
        else:
            label = "Low Value & Infrequent"
        segment_labels.append(label)
    
    segment_profiles['segment_label'] = segment_labels
    
    # Create a mapping from original segments to labeled segments
    segment_mapping = dict(zip(segment_profiles['segment'], segment_profiles['segment_label']))
    
    # Apply the labels to the customer segments
    customer_segments['segment_label'] = customer_segments['segment'].map(segment_mapping)
    
    # Dimensionality reduction for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create dataframe with PCA results
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['segment'] = clusters
    pca_df['segment_label'] = pca_df['segment'].map(segment_mapping)
    pca_df['customer_id'] = customer_segments['customer_id'].values
    
    # Calculate feature importance for each segment
    feature_importance = pd.DataFrame()
    segment_centers = kmeans.cluster_centers_
    
    for i, feature in enumerate(features):
        importance = []
        for j in range(num_segments):
            importance.append(segment_centers[j, i])
        feature_importance[feature] = importance
    
    feature_importance['segment'] = range(num_segments)
    feature_importance['segment_label'] = feature_importance['segment'].map(segment_mapping)
    
    # Return all results
    return {
        'customer_segments': customer_segments,
        'segment_profiles': segment_profiles,
        'pca_results': pca_df,
        'feature_importance': feature_importance,
        'segment_mapping': segment_mapping
    }

def plot_segmentation_results(segmentation_results, features):
    """
    Create visualizations of the segmentation results.
    
    Args:
        segmentation_results (dict): Results from customer segmentation
        features (list): Features used for segmentation
        
    Returns:
        list: List of plotly figures
    """
    figures = []
    
    # 1. Scatter plot of customers using PCA
    pca_df = segmentation_results['pca_results']
    
    fig1 = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='segment_label',
        title='Customer Segments (PCA Visualization)',
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
        hover_data=['customer_id']
    )
    
    fig1.update_layout(
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        legend_title='Segment'
    )
    
    figures.append(fig1)
    
    # 2. Segment profiles
    segment_profiles = segmentation_results['segment_profiles']
    
    # Create radar chart for each segment
    feature_cols = ['avg_order_value', 'avg_purchase_frequency', 'avg_clv', 'avg_recency']
    feature_labels = ['Avg Order Value', 'Purchase Frequency', 'Customer Lifetime Value', 'Recency (days)']
    
    fig2 = go.Figure()
    
    for i, row in segment_profiles.iterrows():
        # Scale the values for better visualization
        values = []
        for col in feature_cols:
            # For recency, lower is better, so invert the scale
            if col == 'avg_recency':
                max_val = segment_profiles[col].max()
                values.append((max_val - row[col]) / max_val * 100)
            else:
                max_val = segment_profiles[col].max()
                values.append(row[col] / max_val * 100)
        
        # Add the first value at the end to close the radar
        values.append(values[0])
        labels = feature_labels + [feature_labels[0]]
        
        fig2.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            name=f"{row['segment_label']} ({row['customer_percentage']}%)"
        ))
    
    fig2.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title="Segment Characteristics (Normalized)",
        showlegend=True
    )
    
    figures.append(fig2)
    
    # 3. Segment size visualization
    fig3 = px.pie(
        segment_profiles,
        values='customer_count',
        names='segment_label',
        title='Customer Segment Distribution',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig3.update_traces(textposition='inside', textinfo='percent+label')
    fig3.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
    
    figures.append(fig3)
    
    # 4. Feature importance for segments
    segment_profiles_dyn = segment_profiles.copy()
    
    # Display only the features we used
    feature_display = [col for col in segment_profiles_dyn.columns 
                      if any(feat in col for feat in features)]
    
    feature_display = [col for col in feature_display 
                      if col not in ['segment', 'customer_count', 'customer_percentage', 'segment_label']]
    
    # Melt the dataframe for barplot
    melted_profiles = segment_profiles_dyn.melt(
        id_vars=['segment_label', 'customer_percentage'],
        value_vars=feature_display,
        var_name='Feature',
        value_name='Value'
    )
    
    # Rename features for display
    feature_map = {
        'avg_purchase_frequency': 'Purchase Frequency',
        'avg_order_value': 'Avg Order Value',
        'avg_clv': 'Customer Lifetime Value',
        'avg_recency': 'Recency (days)'
    }
    
    melted_profiles['Feature'] = melted_profiles['Feature'].map(
        lambda x: feature_map.get(x, x)
    )
    
    fig4 = px.bar(
        melted_profiles,
        x='segment_label',
        y='Value',
        color='segment_label',
        facet_col='Feature',
        title='Segment Characteristics by Feature',
        labels={'segment_label': 'Segment', 'Value': 'Value'},
        height=500
    )
    
    fig4.update_layout(
        showlegend=False,
        xaxis_title="",
        xaxis1_tickangle=45,
        xaxis2_tickangle=45,
        xaxis3_tickangle=45,
        xaxis4_tickangle=45
    )
    
    figures.append(fig4)
    
    return figures
