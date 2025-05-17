import pandas as pd
import numpy as np
from datetime import datetime

def validate_data(data):
    """
    Validate that the uploaded data has the required columns and format.
    
    Args:
        data (pandas.DataFrame): Data to validate
        
    Returns:
        tuple: (is_valid, message)
    """
    required_columns = ['customer_id', 'purchase_date', 'transaction_amount', 'product_category']
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check data types and constraints
    try:
        # Convert purchase_date to datetime
        pd.to_datetime(data['purchase_date'])
        
        # Check if transaction_amount is numeric
        if not pd.api.types.is_numeric_dtype(data['transaction_amount']):
            return False, "transaction_amount must be numeric"
            
    except Exception as e:
        return False, f"Data validation error: {str(e)}"
    
    return True, "Data is valid"

def perform_eda(data):
    """
    Perform Exploratory Data Analysis (EDA) cleaning operations on the raw data.
    
    Args:
        data (pandas.DataFrame): Raw customer data
        
    Returns:
        tuple: (cleaned_data, eda_report)
    """
    # Make a copy to avoid modifying the original
    cleaned_data = data.copy()
    eda_report = {}
    
    # Track initial row count
    initial_row_count = len(cleaned_data)
    eda_report['initial_row_count'] = initial_row_count
    
    # 1. Handle missing values
    missing_values = cleaned_data.isnull().sum()
    eda_report['missing_values'] = missing_values.to_dict()
    
    # Remove rows with missing values in critical columns
    critical_columns = ['customer_id', 'purchase_date', 'transaction_amount', 'product_category']
    pre_missing_count = len(cleaned_data)
    cleaned_data = cleaned_data.dropna(subset=critical_columns)
    post_missing_count = len(cleaned_data)
    eda_report['rows_removed_missing'] = pre_missing_count - post_missing_count
    
    # 2. Remove duplicate rows
    pre_dupes_count = len(cleaned_data)
    cleaned_data = cleaned_data.drop_duplicates()
    post_dupes_count = len(cleaned_data)
    eda_report['duplicates_removed'] = pre_dupes_count - post_dupes_count
    
    # 3. Handle whitespace in string columns
    string_columns = cleaned_data.select_dtypes(include=['object']).columns
    for col in string_columns:
        if col in cleaned_data.columns:  # Check if column exists (it might have been dropped)
            # Strip whitespace
            cleaned_data[col] = cleaned_data[col].astype(str).str.strip()
    
    # 4. Convert data types
    # Try to convert transaction_amount to numeric if it's not already
    if not pd.api.types.is_numeric_dtype(cleaned_data['transaction_amount']):
        # First remove any non-numeric characters (like currency symbols)
        cleaned_data['transaction_amount'] = cleaned_data['transaction_amount'].astype(str)
        cleaned_data['transaction_amount'] = cleaned_data['transaction_amount'].str.replace(r'[^\d.-]', '', regex=True)
        # Convert to numeric
        cleaned_data['transaction_amount'] = pd.to_numeric(cleaned_data['transaction_amount'], errors='coerce')
        # Remove rows with invalid transaction amounts (that got converted to NaN)
        pre_amount_count = len(cleaned_data)
        cleaned_data = cleaned_data.dropna(subset=['transaction_amount'])
        post_amount_count = len(cleaned_data)
        eda_report['rows_removed_invalid_amount'] = pre_amount_count - post_amount_count
    
    # 5. Handle outliers
    # Identify outliers in transaction_amount using IQR method
    Q1 = cleaned_data['transaction_amount'].quantile(0.25)
    Q3 = cleaned_data['transaction_amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    # Track outliers but don't remove them automatically - just log them
    outliers = cleaned_data[(cleaned_data['transaction_amount'] < lower_bound) | 
                            (cleaned_data['transaction_amount'] > upper_bound)]
    eda_report['outliers_detected'] = len(outliers)
    eda_report['outlier_bounds'] = {'lower': lower_bound, 'upper': upper_bound}
    
    # 6. Standardize categories
    # Convert product categories to title case for consistency
    if 'product_category' in cleaned_data.columns:
        cleaned_data['product_category'] = cleaned_data['product_category'].str.title()
    
    # 7. Summary statistics
    eda_report['final_row_count'] = len(cleaned_data)
    eda_report['rows_removed_total'] = initial_row_count - len(cleaned_data)
    eda_report['columns'] = list(cleaned_data.columns)
    
    # 8. Add a source column to track that this is real data
    cleaned_data['data_source'] = 'customer_upload'
    
    return cleaned_data, eda_report

def preprocess_data(data):
    """
    Preprocess the customer data for analysis.
    
    Args:
        data (pandas.DataFrame): Raw customer data
        
    Returns:
        pandas.DataFrame: Processed data with added features
    """
    # First, perform EDA cleaning operations
    cleaned_data, _ = perform_eda(data)
    
    # Use the cleaned data for further processing
    processed_data = cleaned_data.copy()
    
    # Convert date column to datetime
    processed_data['purchase_date'] = pd.to_datetime(processed_data['purchase_date'])
    
    # Sort by customer_id and purchase_date
    processed_data = processed_data.sort_values(['customer_id', 'purchase_date'])
    
    # Extract date components
    processed_data['purchase_year'] = processed_data['purchase_date'].dt.year
    processed_data['purchase_month'] = processed_data['purchase_date'].dt.month
    processed_data['purchase_day'] = processed_data['purchase_date'].dt.day
    processed_data['purchase_weekday'] = processed_data['purchase_date'].dt.dayofweek
    
    # Try to extract hour if format includes time
    try:
        processed_data['purchase_hour'] = processed_data['purchase_date'].dt.hour
    except:
        # If time information is not available, no need to add hour column
        pass
    
    # Calculate days since first purchase for each customer
    first_purchase = processed_data.groupby('customer_id')['purchase_date'].min().reset_index()
    first_purchase.columns = ['customer_id', 'first_purchase_date']
    
    processed_data = pd.merge(processed_data, first_purchase, on='customer_id')
    processed_data['days_since_first_purchase'] = (processed_data['purchase_date'] - processed_data['first_purchase_date']).dt.days
    
    # Calculate recency (days since last purchase, as of the most recent date in the dataset)
    last_date = processed_data['purchase_date'].max()
    last_purchase = processed_data.groupby('customer_id')['purchase_date'].max().reset_index()
    last_purchase.columns = ['customer_id', 'last_purchase_date']
    
    processed_data = pd.merge(processed_data, last_purchase, on='customer_id')
    processed_data['recency_days'] = (last_date - processed_data['last_purchase_date']).dt.days
    
    # Calculate frequency (number of purchases)
    purchase_frequency = processed_data.groupby('customer_id').size().reset_index()
    purchase_frequency.columns = ['customer_id', 'purchase_count']
    
    processed_data = pd.merge(processed_data, purchase_frequency, on='customer_id')
    
    # Calculate total spending by customer
    total_spending = processed_data.groupby('customer_id')['transaction_amount'].sum().reset_index()
    total_spending.columns = ['customer_id', 'total_spent']
    
    processed_data = pd.merge(processed_data, total_spending, on='customer_id')
    
    # Calculate average order value
    processed_data['avg_order_value'] = processed_data['total_spent'] / processed_data['purchase_count']
    
    # Calculate purchase frequency (purchases per month)
    # First, calculate the customer lifetime in months
    processed_data['customer_lifetime_months'] = ((processed_data['last_purchase_date'] - processed_data['first_purchase_date']).dt.days / 30) + 1
    processed_data['purchase_frequency'] = processed_data['purchase_count'] / processed_data['customer_lifetime_months']
    
    # Calculate customer lifetime value (CLV) - simple version
    processed_data['customer_lifetime_value'] = processed_data['total_spent']
    
    return processed_data

def calculate_metrics(processed_data):
    """
    Calculate key business metrics from the processed data.
    
    Args:
        processed_data (pandas.DataFrame): Processed customer data
        
    Returns:
        dict: Dictionary of calculated metrics
    """
    # Get unique data for customer-level metrics
    customer_data = processed_data.drop_duplicates(subset=['customer_id'])
    
    # Basic metrics
    total_customers = len(customer_data)
    total_transactions = len(processed_data)
    total_revenue = processed_data['transaction_amount'].sum()
    
    # Average metrics
    avg_order_value = total_revenue / total_transactions
    avg_customer_value = total_revenue / total_customers
    
    # Calculate revenue for the last month and the month before
    current_month = processed_data['purchase_date'].max().month
    current_year = processed_data['purchase_date'].max().year
    
    last_month_data = processed_data[
        (processed_data['purchase_month'] == current_month) & 
        (processed_data['purchase_year'] == current_year)
    ]
    
    # Calculate previous month (handling year boundary)
    if current_month == 1:
        prev_month = 12
        prev_year = current_year - 1
    else:
        prev_month = current_month - 1
        prev_year = current_year
        
    prev_month_data = processed_data[
        (processed_data['purchase_month'] == prev_month) & 
        (processed_data['purchase_year'] == prev_year)
    ]
    
    last_month_revenue = last_month_data['transaction_amount'].sum() if not last_month_data.empty else 0
    prev_month_revenue = prev_month_data['transaction_amount'].sum() if not prev_month_data.empty else 0
    
    # Calculate growth rate
    if prev_month_revenue > 0:
        revenue_growth = ((last_month_revenue - prev_month_revenue) / prev_month_revenue) * 100
    else:
        revenue_growth = 0
    
    # Customer retention rate
    last_month_customers = set(last_month_data['customer_id'].unique())
    prev_month_customers = set(prev_month_data['customer_id'].unique())
    
    if prev_month_customers:
        retained_customers = len(last_month_customers.intersection(prev_month_customers))
        retention_rate = (retained_customers / len(prev_month_customers)) * 100
    else:
        retention_rate = 0
    
    # Average purchase frequency (purchases per month)
    avg_purchase_frequency = customer_data['purchase_frequency'].mean()
    
    # Return all metrics in a dictionary
    return {
        'total_customers': total_customers,
        'total_transactions': total_transactions,
        'total_revenue': total_revenue,
        'avg_order_value': avg_order_value,
        'avg_customer_value': avg_customer_value,
        'last_month_revenue': last_month_revenue,
        'prev_month_revenue': prev_month_revenue,
        'revenue_growth': revenue_growth,
        'retention_rate': retention_rate,
        'purchase_frequency': avg_purchase_frequency
    }
