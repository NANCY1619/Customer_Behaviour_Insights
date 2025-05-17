import pandas as pd
import numpy as np
from itertools import combinations
import random
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def generate_itemsets(data, min_support=0.01):
    """
    Generate frequent itemsets from transaction data.
    This is a simplified implementation of the Apriori algorithm.
    
    Args:
        data (pandas.DataFrame): Transaction data
        min_support (float): Minimum support threshold
        
    Returns:
        tuple: (frequent_itemsets, item_support)
    """
    # Prepare transaction data: each row is a transaction with a list of items
    transactions = data.groupby(['customer_id', 'order_id'])['product_category'].apply(list).reset_index()
    
    # Get total number of transactions
    total_transactions = len(transactions)
    
    # Count support for individual items
    item_counts = {}
    for _, row in transactions.iterrows():
        items = set(row['product_category'])  # Use set to count each item only once per transaction
        for item in items:
            if item in item_counts:
                item_counts[item] += 1
            else:
                item_counts[item] = 1
    
    # Calculate support and filter by min_support
    item_support = {item: count / total_transactions for item, count in item_counts.items()}
    frequent_items = {item: support for item, support in item_support.items() if support >= min_support}
    
    # If no frequent items, return empty results
    if not frequent_items:
        return {}, {}, {}
    
    # Generate frequent pairs
    pair_counts = {}
    for _, row in transactions.iterrows():
        items = set(row['product_category'])
        frequent_items_in_transaction = [item for item in items if item in frequent_items]
        
        # Generate all possible pairs of frequent items in this transaction
        for pair in combinations(frequent_items_in_transaction, 2):
            # Sort the pair to ensure consistent keys
            pair = tuple(sorted(pair))
            if pair in pair_counts:
                pair_counts[pair] += 1
            else:
                pair_counts[pair] = 1
    
    # Calculate support for pairs and filter by min_support
    pair_support = {pair: count / total_transactions for pair, count in pair_counts.items()}
    frequent_pairs = {pair: support for pair, support in pair_support.items() if support >= min_support}
    
    # Combine frequent items and pairs
    frequent_itemsets = {**{(item,): support for item, support in frequent_items.items()}, **frequent_pairs}
    
    return frequent_itemsets, item_support, pair_support


def calculate_association_rules(frequent_itemsets, item_support, pair_support, min_confidence=0.3):
    """
    Calculate association rules from frequent itemsets.
    
    Args:
        frequent_itemsets (dict): Frequent itemsets
        item_support (dict): Support for individual items
        pair_support (dict): Support for item pairs
        min_confidence (float): Minimum confidence threshold
        
    Returns:
        pandas.DataFrame: Association rules
    """
    rules = []
    
    # Generate rules from frequent pairs
    for pair, support in {k: v for k, v in frequent_itemsets.items() if len(k) == 2}.items():
        # For each pair (A, B), generate rules A->B and B->A
        item1, item2 = pair
        
        # Rule: item1 -> item2
        confidence1 = support / item_support[item1]
        lift1 = confidence1 / item_support[item2]
        
        if confidence1 >= min_confidence:
            rules.append({
                'antecedent': item1,
                'consequent': item2,
                'support': support,
                'confidence': confidence1,
                'lift': lift1
            })
        
        # Rule: item2 -> item1
        confidence2 = support / item_support[item2]
        lift2 = confidence2 / item_support[item1]
        
        if confidence2 >= min_confidence:
            rules.append({
                'antecedent': item2,
                'consequent': item1,
                'support': support,
                'confidence': confidence2,
                'lift': lift2
            })
    
    # Convert to DataFrame and sort by confidence
    rules_df = pd.DataFrame(rules)
    if not rules_df.empty:
        rules_df = rules_df.sort_values('confidence', ascending=False)
    
    return rules_df


def perform_market_basket_analysis(data, min_support=0.01, min_confidence=0.3):
    """
    Perform market basket analysis on transaction data.
    
    Args:
        data (pandas.DataFrame): Transaction data
        min_support (float): Minimum support threshold
        min_confidence (float): Minimum confidence threshold
        
    Returns:
        dict: Market basket analysis results
    """
    # Generate frequent itemsets
    frequent_itemsets, item_support, pair_support = generate_itemsets(data, min_support)
    
    # Calculate association rules
    rules = calculate_association_rules(frequent_itemsets, item_support, pair_support, min_confidence)
    
    # Calculate product co-occurrence matrix
    products = list(item_support.keys())
    co_occurrence = pd.DataFrame(0, index=products, columns=products)
    
    for pair, support in pair_support.items():
        if len(pair) == 2:
            item1, item2 = pair
            co_occurrence.loc[item1, item2] = support
            co_occurrence.loc[item2, item1] = support  # Mirror the matrix
    
    # Set diagonal to item support
    for product in products:
        co_occurrence.loc[product, product] = item_support.get(product, 0)
    
    return {
        'rules': rules,
        'frequent_itemsets': frequent_itemsets,
        'item_support': item_support,
        'co_occurrence': co_occurrence
    }


def plot_market_basket_results(mba_results):
    """
    Create visualizations for market basket analysis.
    
    Args:
        mba_results (dict): Results from market basket analysis
        
    Returns:
        list: List of plotly figures
    """
    figures = []
    
    # 1. Product Popularity (Support)
    if mba_results.get('item_support'):
        item_support_df = pd.DataFrame({
            'Product': list(mba_results['item_support'].keys()),
            'Support': list(mba_results['item_support'].values())
        }).sort_values('Support', ascending=False)
        
        fig1 = px.bar(
            item_support_df,
            x='Product',
            y='Support',
            title='Product Popularity (Support)',
            labels={'Support': 'Support (% of Transactions)', 'Product': 'Product Category'},
            color='Support',
            color_continuous_scale='Viridis',
            height=500
        )
        
        fig1.update_layout(xaxis={'categoryorder': 'total descending'})
        figures.append(fig1)
    
    # 2. Association Rules
    if 'rules' in mba_results and not mba_results['rules'].empty:
        top_rules = mba_results['rules'].head(20)
        
        # Create more descriptive labels
        top_rules['rule'] = top_rules.apply(
            lambda x: f"{x['antecedent']} → {x['consequent']}", axis=1
        )
        
        fig2 = px.bar(
            top_rules,
            x='confidence',
            y='rule',
            title='Top Product Association Rules',
            labels={
                'confidence': 'Confidence (Probability of Consequent given Antecedent)',
                'rule': 'Association Rule'
            },
            color='lift',
            color_continuous_scale='Viridis',
            orientation='h',
            height=600
        )
        
        fig2.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_tickformat='.0%'
        )
        figures.append(fig2)
    
    # 3. Product Co-occurrence Heatmap
    if 'co_occurrence' in mba_results and not mba_results['co_occurrence'].empty:
        co_occurrence = mba_results['co_occurrence']
        
        # Get the most popular products for a cleaner visualization
        top_products = list(pd.Series(mba_results['item_support']).nlargest(10).index)
        co_occurrence_subset = co_occurrence.loc[top_products, top_products]
        
        fig3 = px.imshow(
            co_occurrence_subset,
            labels=dict(x='Product Category', y='Product Category', color='Co-occurrence'),
            x=top_products,
            y=top_products,
            color_continuous_scale='Viridis',
            title='Product Co-occurrence Heatmap',
            height=600
        )
        
        fig3.update_layout(
            xaxis={'tickangle': 45},
            yaxis={'autorange': 'reversed'}
        )
        figures.append(fig3)
    
    # 4. Lift Matrix (for top products)
    if 'rules' in mba_results and not mba_results['rules'].empty:
        # Create lift matrix
        top_products = list(pd.Series(mba_results['item_support']).nlargest(10).index)
        
        # Initialize lift matrix with NaN
        lift_matrix = pd.DataFrame(np.nan, index=top_products, columns=top_products)
        
        # Fill in lift values from rules
        for _, rule in mba_results['rules'].iterrows():
            ant = rule['antecedent']
            cons = rule['consequent']
            if ant in top_products and cons in top_products:
                lift_matrix.loc[ant, cons] = rule['lift']
        
        # Replace NaN with 0 for visualization
        lift_matrix = lift_matrix.fillna(0)
        
        fig4 = px.imshow(
            lift_matrix,
            labels=dict(x='Consequent', y='Antecedent', color='Lift'),
            x=top_products,
            y=top_products,
            color_continuous_scale='RdBu_r',
            title='Product Association Lift Matrix',
            height=600
        )
        
        fig4.update_layout(
            xaxis={'tickangle': 45},
            yaxis={'autorange': 'reversed'}
        )
        figures.append(fig4)
    
    return figures


def generate_product_recommendations(customer_id, data, mba_results):
    """
    Generate product recommendations for a specific customer based on market basket analysis.
    
    Args:
        customer_id (str): Customer ID to generate recommendations for
        data (pandas.DataFrame): Transaction data
        mba_results (dict): Results from market basket analysis
        
    Returns:
        list: Recommended products for the customer
    """
    # Get the customer's purchase history
    customer_purchases = data[data['customer_id'] == customer_id]['product_category'].unique().tolist()
    
    # If no purchase history, return empty list
    if not customer_purchases:
        return []
    
    # If no rules, return empty list
    if 'rules' not in mba_results or mba_results['rules'].empty:
        return []
    
    # Filter rules where the antecedent is in the customer's purchase history
    relevant_rules = mba_results['rules'][mba_results['rules']['antecedent'].isin(customer_purchases)]
    
    # Filter out products the customer has already purchased
    new_products = relevant_rules[~relevant_rules['consequent'].isin(customer_purchases)]
    
    # If no new products to recommend, return empty list
    if new_products.empty:
        return []
    
    # Sort by confidence and lift
    new_products = new_products.sort_values(['confidence', 'lift'], ascending=False)
    
    # Deduplicate recommendations (keep the one with highest confidence)
    new_products = new_products.drop_duplicates(subset=['consequent'], keep='first')
    
    # Format recommendations
    recommendations = []
    for _, row in new_products.head(5).iterrows():
        recommendations.append({
            'product': row['consequent'],
            'confidence': row['confidence'],
            'lift': row['lift'],
            'based_on': row['antecedent']
        })
    
    return recommendations