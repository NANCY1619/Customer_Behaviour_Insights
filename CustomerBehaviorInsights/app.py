import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import modules for basic functionality
from data_generator import generate_customer_data
from data_processing import preprocess_data, validate_data, calculate_metrics, perform_eda
from visualization import (
    create_sales_trend_chart, 
    create_purchase_frequency_chart,
    create_customer_lifetime_value_chart,
    create_product_category_chart,
    create_customer_journey_chart,
    create_cohort_analysis
)
from segmentation import perform_customer_segmentation, plot_segmentation_results
from utils import show_info, download_dataframe

# Import advanced analysis modules
from rfm_analysis import perform_rfm_analysis, plot_rfm_analysis
from customer_prediction import (
    train_churn_prediction_model, predict_customer_churn, plot_churn_analysis,
    train_clv_prediction_model, predict_future_clv, plot_clv_prediction_analysis
)
from market_basket_analysis import (
    perform_market_basket_analysis, plot_market_basket_results, 
    generate_product_recommendations
)
from animation import (
    create_animated_sales_trend, create_animated_customer_growth,
    create_animated_customer_segments, create_geographic_animation
)

# Import section modules
from rfm_section import (
    create_rfm_section, 
    create_predictive_analytics_section, 
    create_market_basket_section, 
    create_animations_section
)

# Page configuration
st.set_page_config(
    page_title="Easy Customer Insights Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'segmentation_done' not in st.session_state:
    st.session_state.segmentation_done = False
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'rfm_done' not in st.session_state:
    st.session_state.rfm_done = False
if 'churn_model' not in st.session_state:
    st.session_state.churn_model = None
if 'clv_model' not in st.session_state:
    st.session_state.clv_model = None
if 'mba_results' not in st.session_state:
    st.session_state.mba_results = None

# Sidebar for navigation and data input
with st.sidebar:
    st.title("Customer Behavior Analysis")
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "What would you like to see?",
        options=[
            "Data Input", 
            "Overview Dashboard", 
            "Customer Groups", 
            "Shopping Patterns", 
            "Time Trends", 
            "VIP Customer Analysis", 
            "Future Predictions", 
            "Products Bought Together", 
            "Visual Stories",
            "About"
        ]
    )
    
    # Create mapping for page titles
    page_mapping = {
        "Customer Groups": "Customer Segmentation",
        "Shopping Patterns": "Purchase Patterns",
        "Time Trends": "Time Analysis",
        "VIP Customer Analysis": "RFM Analysis",
        "Future Predictions": "Predictive Analytics",
        "Products Bought Together": "Market Basket Analysis",
        "Visual Stories": "Animations & Visualizations"
    }
    
    st.markdown("---")
    
    # Data options
    st.subheader("Data Options")
    data_option = st.radio(
        "Choose data source:",
        options=["Upload Your Data", "Generate Dummy Data"]
    )
    
    if data_option == "Upload Your Data":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                is_valid, message = validate_data(data)
                
                if is_valid:
                    # Store original data
                    st.session_state.raw_data = data
                    
                    # Perform EDA and get the report
                    cleaned_data, eda_report = perform_eda(data)
                    st.session_state.data = cleaned_data
                    
                    # Store EDA report
                    st.session_state.eda_report = eda_report
                    
                    # Process data for analysis
                    st.session_state.processed_data = preprocess_data(data)
                    st.session_state.metrics = calculate_metrics(st.session_state.processed_data)
                    
                    # Show success message with EDA summary
                    st.success("Data successfully loaded, cleaned, and processed!")
                    
                    # Create EDA report in an expander
                    with st.expander("📊 View Data Cleaning Report"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("### Data Cleaning Summary")
                            st.metric("Initial Rows", eda_report['initial_row_count'])
                            st.metric("Duplicate Rows Removed", eda_report['duplicates_removed'])
                            st.metric("Rows with Missing Values Removed", 
                                      eda_report.get('rows_removed_missing', 0))
                            if 'rows_removed_invalid_amount' in eda_report:
                                st.metric("Rows with Invalid Amounts Removed", 
                                          eda_report['rows_removed_invalid_amount'])
                            st.metric("Final Row Count", eda_report['final_row_count'])
                            st.metric("Total Rows Removed", eda_report['rows_removed_total'])
                        
                        with col2:
                            st.write("### Data Quality Check")
                            st.write("**Missing Values Detected:**")
                            missing_df = pd.DataFrame.from_dict(
                                eda_report['missing_values'], 
                                orient='index', 
                                columns=['Count']
                            )
                            missing_df = missing_df[missing_df['Count'] > 0]
                            if not missing_df.empty:
                                st.dataframe(missing_df)
                            else:
                                st.write("No missing values detected in the dataset.")
                            
                            st.write("**Outliers Detected:**")
                            st.metric("Potential Outliers", eda_report['outliers_detected'])
                            st.write(f"Outlier Boundaries: Lower = {eda_report['outlier_bounds']['lower']:.2f}, " +
                                     f"Upper = {eda_report['outlier_bounds']['upper']:.2f}")
                else:
                    st.error(f"Invalid data format: {message}")
                    st.info("Please upload data with the following columns: customer_id, purchase_date, transaction_amount, product_category, etc.")
            except Exception as e:
                st.error(f"Error loading data: {e}")
                
    elif data_option == "Generate Dummy Data":
        st.write("Configure your dummy data:")
        num_customers = st.slider("Number of customers", 100, 1000, 500)
        time_period = st.slider("Months of data", 3, 24, 12)
        
        if st.button("Generate Data"):
            with st.spinner("Generating customer data..."):
                # Generate dummy data
                data = generate_customer_data(num_customers, time_period)
                st.session_state.raw_data = data
                
                # Perform EDA and get the report
                cleaned_data, eda_report = perform_eda(data)
                st.session_state.data = cleaned_data
                
                # Store EDA report
                st.session_state.eda_report = eda_report
                
                # Process data for analysis
                st.session_state.processed_data = preprocess_data(data)
                st.session_state.metrics = calculate_metrics(st.session_state.processed_data)
                
                # Show success message
                st.success("Dummy data generated and processed successfully!")
                
                # Create EDA report in an expander
                with st.expander("📊 View Data Cleaning Report"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("### Data Cleaning Summary")
                        st.metric("Initial Rows", eda_report['initial_row_count'])
                        st.metric("Duplicate Rows Removed", eda_report['duplicates_removed'])
                        st.metric("Rows with Missing Values Removed", 
                                  eda_report.get('rows_removed_missing', 0))
                        if 'rows_removed_invalid_amount' in eda_report:
                            st.metric("Rows with Invalid Amounts Removed", 
                                      eda_report['rows_removed_invalid_amount'])
                        st.metric("Final Row Count", eda_report['final_row_count'])
                        st.metric("Total Rows Removed", eda_report['rows_removed_total'])
                    
                    with col2:
                        st.write("### Data Quality Check")
                        st.write("**Missing Values Detected:**")
                        missing_df = pd.DataFrame.from_dict(
                            eda_report['missing_values'], 
                            orient='index', 
                            columns=['Count']
                        )
                        missing_df = missing_df[missing_df['Count'] > 0]
                        if not missing_df.empty:
                            st.dataframe(missing_df)
                        else:
                            st.write("No missing values detected in the dataset.")
                        
                        st.write("**Outliers Detected:**")
                        st.metric("Potential Outliers", eda_report['outliers_detected'])
                        st.write(f"Outlier Boundaries: Lower = {eda_report['outlier_bounds']['lower']:.2f}, " +
                                 f"Upper = {eda_report['outlier_bounds']['upper']:.2f}")
    
    st.markdown("---")
    
    # Show data preview if available
    if st.session_state.data is not None:
        if st.checkbox("Show Data Preview"):
            st.dataframe(st.session_state.data.head())
        
        # Download option
        download_dataframe(st.session_state.data, "customer_data.csv")

# Main content area
if page == "Data Input":
    st.title("Customer Behavior Analysis Tool")
    
    st.markdown("""
    ## Welcome to the Customer Behavior Analysis Dashboard
    
    This easy-to-use tool helps you understand how your customers shop and behave. You'll discover valuable patterns and insights without needing to be a data expert.
    
    ### 🚀 Getting Started (Simple Steps):
    1. **Add Your Data**: Upload your sales data or create test data with the buttons on the left
    2. **Choose Analysis**: Select what you want to learn from the menu on the left
    3. **Explore Results**: View colorful charts and clear insights about your customers
    
    ### 📋 What Your Data Should Include:
    Upload a spreadsheet (CSV file) with these columns:
    - **Customer ID**: Who made the purchase (like customer numbers)
    - **Purchase Date**: When they bought (YYYY-MM-DD format)
    - **Amount**: How much they spent
    - **Product Category**: What type of product they bought
    - **Customer Age**: How old they are (optional)
    - **Customer Gender**: Their gender (optional)
    
    ### 🔍 What You'll Discover:
    - Who your best customers are
    - When people shop the most
    - Which products are often bought together
    - Which customers might leave soon
    - How to group customers for better marketing
    """)
    
    # Show example data format
    with st.expander("View Example Data Format"):
        example_data = pd.DataFrame({
            'customer_id': ['C001', 'C001', 'C002', 'C003', 'C003'],
            'purchase_date': ['2023-01-15', '2023-02-20', '2023-01-10', '2023-03-05', '2023-03-25'],
            'transaction_amount': [125.50, 200.00, 75.25, 300.00, 50.75],
            'product_category': ['Electronics', 'Clothing', 'Home Goods', 'Electronics', 'Food'],
            'customer_age': [34, 34, 28, 45, 45],
            'customer_gender': ['M', 'M', 'F', 'F', 'F']
        })
        st.dataframe(example_data)
    
    if st.session_state.data is None:
        st.info("Please upload or generate data using the sidebar to begin analysis.")
    
elif page == "Overview Dashboard" and st.session_state.processed_data is not None:
    st.title("Customer Overview Dashboard")
    
    # Key metrics in the top row
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric(
            "Total Customers", 
            f"{st.session_state.metrics['total_customers']:,}",
            delta=None
        )
    
    with metrics_col2:
        st.metric(
            "Total Revenue", 
            f"${st.session_state.metrics['total_revenue']:,.2f}",
            delta=f"{st.session_state.metrics['revenue_growth']:.1f}%"
        )
    
    with metrics_col3:
        st.metric(
            "Average Order Value", 
            f"${st.session_state.metrics['avg_order_value']:,.2f}",
            delta=None
        )
    
    with metrics_col4:
        st.metric(
            "Purchase Frequency", 
            f"{st.session_state.metrics['purchase_frequency']:.2f} orders/month",
            delta=None
        )
    
    st.markdown("---")
    
    # Sales trend over time
    st.subheader("Sales Trend Over Time")
    sales_chart = create_sales_trend_chart(st.session_state.processed_data)
    st.plotly_chart(sales_chart, use_container_width=True)
    
    # Customer demographics and purchase behavior
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Product Category Distribution")
        category_chart = create_product_category_chart(st.session_state.processed_data)
        st.plotly_chart(category_chart, use_container_width=True)
    
    with col2:
        st.subheader("Customer Purchase Frequency")
        frequency_chart = create_purchase_frequency_chart(st.session_state.processed_data)
        st.plotly_chart(frequency_chart, use_container_width=True)
    
    # Customer Lifetime Value distribution
    st.subheader("Customer Lifetime Value Distribution")
    clv_chart = create_customer_lifetime_value_chart(st.session_state.processed_data)
    st.plotly_chart(clv_chart, use_container_width=True)
    
    # Export option for dashboard
    st.markdown("---")
    st.markdown("### Export Dashboard Data")
    if st.button("Prepare Dashboard Data for Export"):
        # Create export data with key metrics and summaries
        export_data = {
            "Key Metrics": pd.DataFrame([st.session_state.metrics]),
            "Product Category Summary": st.session_state.processed_data.groupby('product_category').agg({
                'transaction_amount': ['sum', 'mean', 'count']
            }).reset_index()
        }
        st.session_state.export_data = export_data
        st.success("Dashboard data prepared for export!")
        
        for name, df in export_data.items():
            st.subheader(f"{name} Export")
            st.dataframe(df)
            download_dataframe(df, f"{name.lower().replace(' ', '_')}.csv")

elif page == "Customer Segmentation" and st.session_state.processed_data is not None:
    st.title("Customer Segmentation Analysis")
    
    st.write("""
    Customer segmentation helps you identify distinct groups within your customer base. 
    This allows for more targeted marketing strategies and personalized customer experiences.
    """)
    
    # Segmentation options
    st.subheader("Segmentation Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        segmentation_features = st.multiselect(
            "Select features for segmentation:",
            options=["Purchase Frequency", "Customer Lifetime Value", "Average Order Value", "Recency"],
            default=["Purchase Frequency", "Customer Lifetime Value", "Average Order Value"]
        )
        
        # Map selected features to actual dataframe columns
        feature_mapping = {
            "Purchase Frequency": "purchase_frequency",
            "Customer Lifetime Value": "customer_lifetime_value",
            "Average Order Value": "avg_order_value",
            "Recency": "recency_days"
        }
        
        selected_features = [feature_mapping[f] for f in segmentation_features]
    
    with col2:
        num_segments = st.slider("Number of segments", min_value=2, max_value=7, value=3)
    
    # Run segmentation when requested
    if st.button("Perform Customer Segmentation"):
        if len(selected_features) < 2:
            st.error("Please select at least 2 features for segmentation.")
        else:
            with st.spinner("Segmenting customers..."):
                # Perform segmentation and store results in session state
                segmentation_results = perform_customer_segmentation(
                    st.session_state.processed_data, 
                    selected_features, 
                    num_segments
                )
                st.session_state.segmentation_results = segmentation_results
                st.session_state.segmentation_done = True
                st.success("Segmentation completed!")
    
    # Display segmentation results if available
    if st.session_state.segmentation_done:
        st.subheader("Segmentation Results")
        
        # Visualization of segments
        st.write("### Customer Segments Visualization")
        segment_charts = plot_segmentation_results(st.session_state.segmentation_results, selected_features)
        
        for chart in segment_charts:
            st.plotly_chart(chart, use_container_width=True)
        
        # Segment characteristics table
        st.write("### Segment Characteristics")
        segment_characteristics = st.session_state.segmentation_results["segment_profiles"]
        st.dataframe(segment_characteristics)
        
        # Download segmentation results
        download_dataframe(
            st.session_state.segmentation_results["customer_segments"], 
            "customer_segments.csv"
        )

elif page == "Purchase Patterns" and st.session_state.processed_data is not None:
    st.title("Purchase Pattern Analysis")
    
    st.write("""
    Understand how customers interact with your products and services over time.
    This section analyzes purchasing patterns, product associations, and customer journeys.
    """)
    
    # Time period filter
    st.subheader("Filter Time Period")
    
    col1, col2 = st.columns(2)
    with col1:
        min_date = st.session_state.processed_data['purchase_date'].min().date()
        max_date = st.session_state.processed_data['purchase_date'].max().date()
        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    
    with col2:
        end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    
    # Filter data based on selected date range
    filtered_data = st.session_state.processed_data[
        (st.session_state.processed_data['purchase_date'].dt.date >= start_date) &
        (st.session_state.processed_data['purchase_date'].dt.date <= end_date)
    ]
    
    if start_date > end_date:
        st.error("Start date must be before end date")
    elif filtered_data.empty:
        st.warning("No data available for the selected date range")
    else:
        # Purchase patterns analysis
        st.subheader("Product Category Trends")
        
        # Category trend over time
        category_trend = filtered_data.groupby([pd.Grouper(key='purchase_date', freq='M'), 'product_category'])['transaction_amount'].sum().reset_index()
        category_trend['month'] = category_trend['purchase_date'].dt.strftime('%b %Y')
        
        fig = px.line(
            category_trend, 
            x='month', 
            y='transaction_amount', 
            color='product_category',
            title='Monthly Sales by Product Category',
            labels={'transaction_amount': 'Sales Amount ($)', 'month': 'Month', 'product_category': 'Product Category'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Customer journey analysis
        st.subheader("Customer Journey Analysis")
        journey_chart = create_customer_journey_chart(filtered_data)
        st.plotly_chart(journey_chart, use_container_width=True)
        
        # Product category transition analysis (showing how customers move between categories)
        st.subheader("Product Category Transition Analysis")
        
        # Get the top 25 customers with most transactions for analysis
        top_customers = filtered_data['customer_id'].value_counts().nlargest(25).index.tolist()
        top_customer_data = filtered_data[filtered_data['customer_id'].isin(top_customers)]
        
        # Create a Sankey diagram of product category transitions
        customer_journeys = top_customer_data.sort_values(['customer_id', 'purchase_date'])
        
        # For each customer, record transitions between product categories
        transitions = []
        for customer in top_customers:
            customer_purchases = customer_journeys[customer_journeys['customer_id'] == customer]
            
            if len(customer_purchases) > 1:
                categories = customer_purchases['product_category'].tolist()
                for i in range(len(categories) - 1):
                    transitions.append((categories[i], categories[i+1]))
        
        # Count transitions
        transition_counts = pd.Series(transitions).value_counts().reset_index()
        transition_counts.columns = ['transition', 'count']
        transition_counts[['source', 'target']] = pd.DataFrame(transition_counts['transition'].tolist(), index=transition_counts.index)
        
        # Only keep transitions with at least 2 occurrences
        transition_counts = transition_counts[transition_counts['count'] >= 2]
        
        if not transition_counts.empty:
            # Create a list of unique categories
            unique_categories = list(set(transition_counts['source'].tolist() + transition_counts['target'].tolist()))
            
            # Create node indices
            category_indices = {category: i for i, category in enumerate(unique_categories)}
            
            # Create Sankey diagram
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=unique_categories
                ),
                link=dict(
                    source=[category_indices[source] for source in transition_counts['source']],
                    target=[category_indices[target] for target in transition_counts['target']],
                    value=transition_counts['count'],
                    hovertemplate='%{source.label} → %{target.label}<br>' +
                                'Count: %{value}<extra></extra>'
                )
            )])
            
            fig.update_layout(title_text="Product Category Transitions", font_size=12)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to show category transitions. Try selecting a larger date range or more customers.")
        
        # Cohort analysis
        st.subheader("Cohort Analysis")
        cohort_plot = create_cohort_analysis(filtered_data)
        st.plotly_chart(cohort_plot, use_container_width=True)

elif page == "Time Analysis" and st.session_state.processed_data is not None:
    st.title("Time-Based Analysis")
    
    st.write("""
    Understand how customer behavior changes over time. This section analyzes seasonal trends,
    day of week patterns, and helps identify important time-based insights.
    """)
    
    # Time-based analyses
    
    # 1. Monthly trend analysis
    st.subheader("Monthly Sales Trend")
    
    monthly_data = st.session_state.processed_data.set_index('purchase_date')
    monthly_sales = monthly_data.resample('M')['transaction_amount'].sum().reset_index()
    monthly_sales['month'] = monthly_sales['purchase_date'].dt.strftime('%b %Y')
    monthly_count = monthly_data.resample('M')['transaction_amount'].count().reset_index()
    monthly_count['month'] = monthly_count['purchase_date'].dt.strftime('%b %Y')
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=monthly_sales['month'],
            y=monthly_sales['transaction_amount'],
            name="Sales Amount",
            marker_color='rgb(55, 83, 109)'
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_count['month'],
            y=monthly_count['transaction_amount'],
            name="Transaction Count",
            marker_color='rgb(26, 118, 255)',
            mode='lines+markers'
        ),
        secondary_y=True,
    )
    
    fig.update_layout(
        title_text="Monthly Sales and Transaction Count",
        xaxis=dict(
            title="Month",
            tickfont=dict(size=10),
            tickangle=45
        )
    )
    
    fig.update_yaxes(title_text="Sales Amount ($)", secondary_y=False)
    fig.update_yaxes(title_text="Transaction Count", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. Day of week analysis
    st.subheader("Day of Week Analysis")
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_data = st.session_state.processed_data.copy()
    day_data['day_of_week'] = day_data['purchase_date'].dt.day_name()
    
    day_amount = day_data.groupby('day_of_week')['transaction_amount'].agg(['sum', 'count', 'mean']).reset_index()
    day_amount = day_amount.set_index('day_of_week').reindex(day_order).reset_index()
    
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=("Transaction Count by Day of Week", 
                                         "Average Order Value by Day of Week"),
                        specs=[[{"type": "bar"}, {"type": "bar"}]])
    
    fig.add_trace(
        go.Bar(x=day_amount['day_of_week'], y=day_amount['count'], name="Transaction Count"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=day_amount['day_of_week'], y=day_amount['mean'], name="Average Order Value"),
        row=1, col=2
    )
    
    fig.update_layout(height=500, title_text="Shopping Behavior by Day of Week")
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Average Order Value ($)", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. Hour of day analysis (if time data is available)
    if 'purchase_hour' in st.session_state.processed_data.columns:
        st.subheader("Hour of Day Analysis")
        
        hour_data = st.session_state.processed_data.copy()
        hour_amount = hour_data.groupby('purchase_hour')['transaction_amount'].agg(['sum', 'count', 'mean']).reset_index()
        
        fig = make_subplots(rows=1, cols=2, 
                            subplot_titles=("Transaction Count by Hour", 
                                             "Average Order Value by Hour"),
                            specs=[[{"type": "scatter"}, {"type": "scatter"}]])
        
        fig.add_trace(
            go.Scatter(x=hour_amount['purchase_hour'], y=hour_amount['count'], 
                      mode='lines+markers', name="Transaction Count"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=hour_amount['purchase_hour'], y=hour_amount['mean'], 
                      mode='lines+markers', name="Average Order Value"),
            row=1, col=2
        )
        
        fig.update_layout(height=500, title_text="Shopping Behavior by Hour of Day")
        fig.update_xaxes(title_text="Hour of Day", row=1, col=1)
        fig.update_xaxes(title_text="Hour of Day", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Average Order Value ($)", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 4. Customer retention over time
    st.subheader("Customer Retention Analysis")
    
    # Prepare data for retention analysis
    retention_data = st.session_state.processed_data.copy()
    retention_data['purchase_month'] = retention_data['purchase_date'].dt.to_period('M')
    retention_data['cohort'] = retention_data.groupby('customer_id')['purchase_date'].transform('min').dt.to_period('M')
    
    # Count number of unique customers in each cohort and month
    cohort_data = retention_data.groupby(['cohort', 'purchase_month']).agg(n_customers=('customer_id', 'nunique')).reset_index()
    
    # Calculate periods since first purchase
    cohort_data['period'] = (cohort_data['purchase_month'] - cohort_data['cohort']).apply(lambda x: x.n)
    
    # Create cohort table: cohort on rows, period on columns, values are retention rate
    cohort_pivot = cohort_data.pivot_table(index='cohort', columns='period', values='n_customers')
    
    # Calculate retention rate
    cohort_size = cohort_pivot[0]
    retention_matrix = cohort_pivot.divide(cohort_size, axis=0) * 100
    
    # Select top N cohorts with most data
    top_cohorts = min(8, len(retention_matrix))
    retention_matrix = retention_matrix.iloc[:top_cohorts, :12]  # First 12 periods
    
    # Create heatmap
    fig = px.imshow(
        retention_matrix.values,
        labels=dict(x="Months Since First Purchase", y="Cohort", color="Retention Rate (%)"),
        x=[f"Month {i}" for i in retention_matrix.columns],
        y=[str(cohort) for cohort in retention_matrix.index],
        color_continuous_scale="Blues",
        aspect="auto"
    )
    
    fig.update_layout(
        title="Customer Retention by Cohort (%)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif page == "RFM Analysis" and st.session_state.processed_data is not None:
    create_rfm_section(st, st.session_state)

elif page == "Predictive Analytics" and st.session_state.processed_data is not None:
    create_predictive_analytics_section(st, st.session_state)

elif page == "Market Basket Analysis" and st.session_state.processed_data is not None:
    create_market_basket_section(st, st.session_state)

elif page == "Animations & Visualizations" and st.session_state.processed_data is not None:
    create_animations_section(st, st.session_state)

elif page == "About":
    st.title("About this Application")
    
    st.write("""
    ## Customer Behavior Analysis Dashboard
    
    This application is designed to help businesses analyze customer behavior patterns 
    and gain actionable insights from their customer data. By understanding how customers 
    interact with your business, you can optimize marketing strategies, improve customer 
    experiences, and increase customer retention.
    
    ### Key Features:
    
    - **Data Upload and Visualization**: Upload your own data or generate sample data for analysis
    - **Customer Segmentation**: Group customers based on behavior patterns
    - **Purchase Pattern Analysis**: Understand what, when, and how customers buy
    - **Time-Based Analysis**: Identify trends, seasonality, and time-based patterns
    - **Interactive Dashboards**: Explore data through interactive charts and filters
    
    ### How to Use This Tool:
    
    1. Start by uploading your customer data or generating dummy data in the sidebar
    2. Navigate through different analysis sections using the navigation menu
    3. Interact with visualizations to explore different dimensions of your data
    4. Export results for further analysis or reporting
    
    ### Technologies Used:
    
    - **Streamlit**: Web application framework
    - **Pandas & NumPy**: Data processing and analysis
    - **Plotly**: Interactive data visualization
    - **Scikit-learn**: Machine learning for customer segmentation
    """)
    
    # Contact information
    st.markdown("---")
    st.subheader("Contact Information")
    
    st.write("""
    For questions, feedback, or feature requests, please contact:
    
    📧 support@customerbehavioranalytics.com
    """)
    
    # Version information
    st.markdown("---")
    st.caption("Customer Behavior Analysis Dashboard v1.0")
else:
    if st.session_state.processed_data is None and page != "About" and page != "Data Input":
        st.title("Data Required")
        st.info("Please upload or generate data using the sidebar before accessing this section.")
