import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_customer_data(num_customers=500, months=12):
    """
    Generate realistic dummy customer data for analysis.
    
    Args:
        num_customers (int): Number of unique customers to generate
        months (int): Number of months of historical data to generate
        
    Returns:
        pandas.DataFrame: Generated customer data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create customer IDs and attributes
    customer_ids = [f"C{str(i).zfill(5)}" for i in range(1, num_customers + 1)]
    
    # Customer demographics
    age_groups = {
        '18-24': (18, 24),
        '25-34': (25, 34),
        '35-44': (35, 44),
        '45-54': (45, 54),
        '55-64': (55, 64),
        '65+': (65, 85)
    }
    
    # Age distribution weights (more customers in 25-44 range)
    age_weights = [0.15, 0.25, 0.25, 0.15, 0.1, 0.1]
    
    # Generate customer profiles
    customers = []
    for customer_id in customer_ids:
        # Assign gender
        gender = np.random.choice(['M', 'F'], p=[0.48, 0.52])
        
        # Assign age group and then specific age
        age_group = np.random.choice(list(age_groups.keys()), p=age_weights)
        min_age, max_age = age_groups[age_group]
        age = np.random.randint(min_age, max_age + 1)
        
        # Assign location
        location = np.random.choice([
            'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
            'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'Austin'
        ])
        
        # Customer loyalty - how long they've been a customer (in months)
        loyalty = np.random.randint(1, months + 1)
        
        # Customer type (frequency-based)
        purchase_frequency_category = np.random.choice(
            ['Frequent', 'Regular', 'Occasional', 'Rare'],
            p=[0.2, 0.3, 0.3, 0.2]
        )
        
        # Different customer types have different purchase frequencies
        if purchase_frequency_category == 'Frequent':
            avg_purchases_per_month = np.random.uniform(3, 5)
        elif purchase_frequency_category == 'Regular':
            avg_purchases_per_month = np.random.uniform(1, 3)
        elif purchase_frequency_category == 'Occasional':
            avg_purchases_per_month = np.random.uniform(0.5, 1)
        else:  # Rare
            avg_purchases_per_month = np.random.uniform(0.1, 0.5)
        
        customers.append({
            'customer_id': customer_id,
            'customer_age': age,
            'customer_gender': gender,
            'customer_location': location,
            'loyalty_months': loyalty,
            'purchase_frequency_category': purchase_frequency_category,
            'avg_purchases_per_month': avg_purchases_per_month
        })
    
    # Create customer DataFrame
    customer_df = pd.DataFrame(customers)
    
    # Generate purchase transactions
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30 * months)
    
    # Product categories and their price ranges
    product_categories = {
        'Electronics': (100, 1500),
        'Clothing': (20, 200),
        'Home Goods': (30, 500),
        'Food & Grocery': (10, 150),
        'Beauty & Personal Care': (15, 100),
        'Sports & Outdoors': (25, 300),
        'Books & Media': (10, 50),
        'Toys & Games': (15, 100)
    }
    
    # Weighted probability for product categories
    category_weights = [0.2, 0.25, 0.15, 0.15, 0.1, 0.05, 0.05, 0.05]
    
    # Generate transactions
    transactions = []
    
    for _, customer in customer_df.iterrows():
        # Calculate how many months this customer has been active
        active_months = min(customer['loyalty_months'], months)
        
        # Calculate approximate number of transactions for this customer
        expected_transactions = int(customer['avg_purchases_per_month'] * active_months)
        
        # Add some randomness
        num_transactions = max(1, np.random.poisson(expected_transactions))
        
        # Customer-specific preferences (some categories they buy more often)
        preferred_categories = np.random.choice(
            list(product_categories.keys()),
            size=min(3, len(product_categories)),
            replace=False
        )
        
        # Generate each transaction
        for _ in range(num_transactions):
            # Determine date of purchase (more recent = more likely)
            days_ago = np.random.triangular(0, 0, active_months * 30)
            purchase_date = end_date - timedelta(days=days_ago)
            
            # 70% chance to pick from preferred categories, 30% chance for any category
            if np.random.random() < 0.7:
                category = np.random.choice(preferred_categories)
            else:
                category = np.random.choice(
                    list(product_categories.keys()),
                    p=category_weights
                )
            
            # Determine price range for the category
            min_price, max_price = product_categories[category]
            
            # Generate transaction amount (using gamma distribution for right skew)
            mean_price = (min_price + max_price) / 2
            shape = 2
            scale = mean_price / shape
            amount = np.random.gamma(shape, scale)
            
            # Ensure amount is within reasonable range
            amount = max(min_price, min(max_price, amount))
            
            # Add transaction
            transactions.append({
                'customer_id': customer['customer_id'],
                'purchase_date': purchase_date,
                'transaction_amount': round(amount, 2),
                'product_category': category,
                'customer_age': customer['customer_age'],
                'customer_gender': customer['customer_gender'],
                'customer_location': customer['customer_location']
            })
    
    # Create transaction DataFrame
    transaction_df = pd.DataFrame(transactions)
    
    # Sort by customer_id and purchase_date
    transaction_df = transaction_df.sort_values(['customer_id', 'purchase_date'])
    
    # Convert purchase_date to string format for easier handling
    transaction_df['purchase_date'] = transaction_df['purchase_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return transaction_df
