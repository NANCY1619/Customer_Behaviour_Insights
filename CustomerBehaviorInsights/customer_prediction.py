import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def train_churn_prediction_model(data):
    """
    Train a model to predict customer churn.
    
    Args:
        data (pandas.DataFrame): Processed customer data
        
    Returns:
        tuple: (model, features, accuracy, feature_importance)
    """
    # Prepare data for churn prediction
    # First, identify customers who haven't made a purchase in the last 90 days
    max_date = data['purchase_date'].max()
    customers = data.groupby('customer_id').agg({
        'purchase_date': lambda x: (max_date - x.max()).days,  # Recency
        'order_id': 'nunique',  # Frequency
        'transaction_amount': 'sum',  # Monetary
        'avg_order_value': 'mean',
        'purchase_frequency': 'mean',
        'customer_lifetime_value': 'max',
        'days_active': 'max'
    }).reset_index()
    
    # Define churn as no purchase in the last 90 days
    customers['churned'] = customers['purchase_date'] > 90
    
    # Select features for the model
    features = [
        'purchase_date',  # Recency
        'order_id',  # Frequency
        'transaction_amount',  # Monetary
        'avg_order_value',
        'purchase_frequency',
        'customer_lifetime_value',
        'days_active'
    ]
    
    X = customers[features]
    y = customers['churned']
    
    # Handle any potential NaN values
    X = X.fillna(0)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, features, accuracy, feature_importance


def predict_customer_churn(model, features, customer_data):
    """
    Predict churn for each customer.
    
    Args:
        model: Trained model
        features (list): Features used for prediction
        customer_data (pandas.DataFrame): Customer data
        
    Returns:
        pandas.DataFrame: Customer data with churn predictions
    """
    # Prepare customer data for prediction
    X = customer_data[features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Get predictions and probabilities
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]  # Probability of churn
    
    # Add predictions to customer data
    result = customer_data.copy()
    result['churn_prediction'] = y_pred
    result['churn_probability'] = y_proba
    
    # Define risk categories
    result['churn_risk'] = 'Low'
    result.loc[result['churn_probability'] > 0.3, 'churn_risk'] = 'Medium'
    result.loc[result['churn_probability'] > 0.7, 'churn_risk'] = 'High'
    
    return result


def plot_churn_analysis(churn_data, feature_importance):
    """
    Create visualizations for churn analysis.
    
    Args:
        churn_data (pandas.DataFrame): Customer data with churn predictions
        feature_importance (pandas.DataFrame): Feature importance from the model
        
    Returns:
        list: List of plotly figures
    """
    figures = []
    
    # 1. Churn Risk Distribution
    risk_counts = churn_data['churn_risk'].value_counts().reset_index()
    risk_counts.columns = ['Risk Level', 'Count']
    
    # Define color mapping
    color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    
    fig1 = px.pie(
        risk_counts,
        values='Count',
        names='Risk Level',
        title='Customer Churn Risk Distribution',
        color='Risk Level',
        color_discrete_map=color_map,
        hole=0.4
    )
    
    fig1.update_traces(textposition='inside', textinfo='percent+label')
    figures.append(fig1)
    
    # 2. Feature Importance
    fig2 = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Factors Influencing Churn',
        labels={'Importance': 'Feature Importance', 'Feature': 'Customer Attribute'},
        color='Importance',
        color_continuous_scale='Viridis',
        height=500
    )
    
    fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
    figures.append(fig2)
    
    # 3. Churn Probability Distribution
    fig3 = px.histogram(
        churn_data,
        x='churn_probability',
        title='Distribution of Churn Probability',
        labels={'churn_probability': 'Churn Probability'},
        nbins=50,
        color='churn_risk',
        color_discrete_map=color_map,
        height=400
    )
    
    fig3.update_layout(bargap=0.1)
    figures.append(fig3)
    
    # 4. Recency vs Frequency colored by Churn Risk
    fig4 = px.scatter(
        churn_data,
        x='recency_days',
        y='purchase_frequency',
        color='churn_risk',
        size='customer_lifetime_value',
        color_discrete_map=color_map,
        title='Recency vs Frequency by Churn Risk',
        labels={
            'recency_days': 'Recency (days since last purchase)',
            'purchase_frequency': 'Purchase Frequency (orders/month)',
            'customer_lifetime_value': 'Customer Lifetime Value'
        },
        height=600
    )
    
    fig4.update_layout(legend_title_text='Churn Risk')
    figures.append(fig4)
    
    return figures


def train_clv_prediction_model(data):
    """
    Train a model to predict future customer lifetime value.
    
    Args:
        data (pandas.DataFrame): Processed customer data
        
    Returns:
        tuple: (model, features, r2_score, feature_importance)
    """
    # Prepare data for CLV prediction
    # We'll use past behavior to predict future CLV
    customers = data.groupby('customer_id').agg({
        'purchase_date': lambda x: (data['purchase_date'].max() - x.max()).days,  # Recency
        'order_id': 'nunique',  # Frequency
        'transaction_amount': 'sum',  # Monetary
        'avg_order_value': 'mean',
        'purchase_frequency': 'mean',
        'customer_lifetime_value': 'max',
        'days_active': 'max'
    }).reset_index()
    
    # For the target, we'll use current CLV
    # In a real scenario, you would train on past data and predict future CLV
    target = 'customer_lifetime_value'
    
    # Features for the model
    features = [
        'purchase_date',  # Recency
        'order_id',  # Frequency
        'transaction_amount',  # Monetary
        'avg_order_value',
        'purchase_frequency',
        'days_active'
    ]
    
    X = customers[features]
    y = customers[target]
    
    # Handle any NaN values
    X = X.fillna(0)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a Random Forest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, features, r2, feature_importance


def predict_future_clv(model, features, customer_data):
    """
    Predict future CLV for each customer.
    
    Args:
        model: Trained model
        features (list): Features used for prediction
        customer_data (pandas.DataFrame): Customer data
        
    Returns:
        pandas.DataFrame: Customer data with future CLV predictions
    """
    # Prepare customer data for prediction
    X = customer_data[features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Get predictions
    y_pred = model.predict(X_scaled)
    
    # Add predictions to customer data
    result = customer_data.copy()
    result['predicted_future_clv'] = y_pred
    
    # Calculate growth potential (future CLV / current CLV)
    result['clv_growth_potential'] = result['predicted_future_clv'] / result['customer_lifetime_value']
    result['clv_growth_potential'] = result['clv_growth_potential'].fillna(0)
    
    # Define growth potential categories
    result['growth_potential'] = 'Low'
    result.loc[result['clv_growth_potential'] > 1.2, 'growth_potential'] = 'Medium'
    result.loc[result['clv_growth_potential'] > 1.5, 'growth_potential'] = 'High'
    
    return result


def plot_clv_prediction_analysis(clv_data, feature_importance):
    """
    Create visualizations for CLV prediction analysis.
    
    Args:
        clv_data (pandas.DataFrame): Customer data with CLV predictions
        feature_importance (pandas.DataFrame): Feature importance from the model
        
    Returns:
        list: List of plotly figures
    """
    figures = []
    
    # 1. Growth Potential Distribution
    growth_counts = clv_data['growth_potential'].value_counts().reset_index()
    growth_counts.columns = ['Growth Potential', 'Count']
    
    # Define color mapping
    color_map = {'Low': '#FFA07A', 'Medium': '#7B68EE', 'High': '#3CB371'}
    
    fig1 = px.pie(
        growth_counts,
        values='Count',
        names='Growth Potential',
        title='Customer Growth Potential Distribution',
        color='Growth Potential',
        color_discrete_map=color_map,
        hole=0.4
    )
    
    fig1.update_traces(textposition='inside', textinfo='percent+label')
    figures.append(fig1)
    
    # 2. Feature Importance
    fig2 = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Factors Influencing Future CLV',
        labels={'Importance': 'Feature Importance', 'Feature': 'Customer Attribute'},
        color='Importance',
        color_continuous_scale='Viridis',
        height=500
    )
    
    fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
    figures.append(fig2)
    
    # 3. Current vs Predicted CLV
    fig3 = px.scatter(
        clv_data,
        x='customer_lifetime_value',
        y='predicted_future_clv',
        color='growth_potential',
        size='purchase_frequency',
        color_discrete_map=color_map,
        title='Current vs Predicted Future CLV',
        labels={
            'customer_lifetime_value': 'Current CLV ($)',
            'predicted_future_clv': 'Predicted Future CLV ($)',
            'purchase_frequency': 'Purchase Frequency (orders/month)'
        },
        height=600
    )
    
    # Add reference line (y=x)
    fig3.add_trace(
        go.Scatter(
            x=[clv_data['customer_lifetime_value'].min(), clv_data['customer_lifetime_value'].max()],
            y=[clv_data['customer_lifetime_value'].min(), clv_data['customer_lifetime_value'].max()],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='No Growth'
        )
    )
    
    fig3.update_layout(legend_title_text='Growth Potential')
    figures.append(fig3)
    
    # 4. Growth Potential by Customer Segment
    if 'RFM_Segment' in clv_data.columns:
        segment_growth = clv_data.groupby('RFM_Segment')['clv_growth_potential'].mean().reset_index()
        segment_growth = segment_growth.sort_values('clv_growth_potential', ascending=False)
        
        fig4 = px.bar(
            segment_growth,
            x='RFM_Segment',
            y='clv_growth_potential',
            title='Average Growth Potential by Customer Segment',
            labels={
                'RFM_Segment': 'Customer Segment',
                'clv_growth_potential': 'Average Growth Potential'
            },
            color='clv_growth_potential',
            color_continuous_scale='Viridis',
            height=500
        )
        
        fig4.update_layout(xaxis={'categoryorder': 'total descending'})
        figures.append(fig4)
    
    return figures