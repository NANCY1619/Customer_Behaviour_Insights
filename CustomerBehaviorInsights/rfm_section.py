def create_rfm_section(st, st_session_state):
    """
    Create the RFM Analysis section of the app.
    
    Args:
        st: Streamlit module
        st_session_state: Streamlit session state
    """
    from rfm_analysis import perform_rfm_analysis, plot_rfm_analysis
    from utils import download_dataframe
    
    st.title("RFM Analysis - Customer Value Segmentation")
    
    st.write("""
    RFM (Recency, Frequency, Monetary) analysis is a customer segmentation technique that uses past purchase behavior 
    to identify customer segments. This powerful marketing analysis tool helps you identify your most valuable customers.
    
    - **Recency** - How recently has the customer made a purchase?
    - **Frequency** - How often does the customer make purchases?
    - **Monetary** - How much does the customer spend?
    """)
    
    # Run RFM analysis
    if not st_session_state.rfm_done:
        if st.button("Perform RFM Analysis"):
            with st.spinner("Analyzing customer RFM segments..."):
                # Run RFM analysis
                rfm_data = perform_rfm_analysis(st_session_state.processed_data)
                if rfm_data is not None:
                    st_session_state.rfm_data = rfm_data
                    st_session_state.rfm_done = True
                    st.success("RFM Analysis completed!")
                else:
                    st.error("Could not perform RFM analysis. Please ensure your data contains necessary fields.")
    
    # Display RFM results if available
    if st_session_state.rfm_done:
        # Display key segment counts
        st.subheader("Customer RFM Segments")
        
        # Create a summary of segment counts
        segment_counts = st_session_state.rfm_data['RFM_Segment'].value_counts()
        
        # Set up metrics in rows with colorful cards
        total_customers = len(st_session_state.rfm_data)
        
        # Show metrics for key segments
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            champions_count = segment_counts.get('Champions', 0)
            champions_percent = (champions_count / total_customers * 100) if total_customers > 0 else 0
            st.metric(
                "Champions",
                f"{champions_count}",
                f"{champions_percent:.1f}% of total"
            )
            
        with col2:
            loyal_count = segment_counts.get('Loyal Customers', 0)
            loyal_percent = (loyal_count / total_customers * 100) if total_customers > 0 else 0
            st.metric(
                "Loyal Customers",
                f"{loyal_count}",
                f"{loyal_percent:.1f}% of total"
            )
            
        with col3:
            at_risk_count = segment_counts.get('At Risk', 0)
            at_risk_percent = (at_risk_count / total_customers * 100) if total_customers > 0 else 0
            st.metric(
                "At Risk",
                f"{at_risk_count}",
                f"{at_risk_percent:.1f}% of total"
            )
            
        with col4:
            cant_lose_count = segment_counts.get("Can't Lose Them", 0)
            cant_lose_percent = (cant_lose_count / total_customers * 100) if total_customers > 0 else 0
            st.metric(
                "Can't Lose Them",
                f"{cant_lose_count}",
                f"{cant_lose_percent:.1f}% of total"
            )
        
        # Display RFM visualizations
        rfm_charts = plot_rfm_analysis(st_session_state.rfm_data)
        
        for chart in rfm_charts:
            st.plotly_chart(chart, use_container_width=True)
        
        # Download RFM data
        st.subheader("Download RFM Analysis")
        download_dataframe(st_session_state.rfm_data, "rfm_analysis.csv")


def create_predictive_analytics_section(st, st_session_state):
    """
    Create the Predictive Analytics section of the app.
    
    Args:
        st: Streamlit module
        st_session_state: Streamlit session state
    """
    import pandas as pd
    from customer_prediction import (
        train_churn_prediction_model, predict_customer_churn, plot_churn_analysis,
        train_clv_prediction_model, predict_future_clv, plot_clv_prediction_analysis
    )
    from utils import download_dataframe
    
    st.title("Predictive Analytics")
    
    st.write("""
    This section uses machine learning algorithms to predict future customer behavior.
    You can predict customer churn probability and future customer lifetime value.
    """)
    
    # Tabs for different predictive models
    tabs = st.tabs(["Customer Churn Prediction", "CLV Prediction"])
    
    # Churn Prediction Tab
    with tabs[0]:
        st.subheader("Customer Churn Prediction")
        
        st.write("""
        Predict which customers are likely to churn (stop purchasing) in the near future.
        This can help you take proactive measures to retain these customers.
        """)
        
        if st_session_state.churn_model is None:
            if st.button("Train Churn Prediction Model"):
                with st.spinner("Training churn prediction model... This may take a moment."):
                    model, features, accuracy, feature_importance = train_churn_prediction_model(
                        st_session_state.processed_data
                    )
                    
                    if model is not None:
                        st_session_state.churn_model = model
                        st_session_state.churn_features = features
                        st_session_state.churn_accuracy = accuracy
                        st_session_state.churn_feature_importance = feature_importance
                        
                        # Generate predictions
                        customer_data = st_session_state.processed_data.drop_duplicates(subset=['customer_id'])
                        st_session_state.churn_predictions = predict_customer_churn(
                            model, features, customer_data
                        )
                        
                        st.success(f"Churn prediction model trained successfully! Model accuracy: {accuracy:.2f}")
                    else:
                        st.error("Could not train churn prediction model. Insufficient data or too many missing values.")
        else:
            st.success(f"Churn prediction model is ready. Model accuracy: {st_session_state.churn_accuracy:.2f}")
            
            # Show churn analysis
            churn_charts = plot_churn_analysis(
                st_session_state.churn_predictions, 
                st_session_state.churn_feature_importance
            )
            
            for chart in churn_charts:
                st.plotly_chart(chart, use_container_width=True)
            
            # Show customers with high churn risk
            st.subheader("High Churn Risk Customers")
            high_risk = st_session_state.churn_predictions[
                st_session_state.churn_predictions['churn_risk'] == 'High'
            ].sort_values('churn_probability', ascending=False)
            
            if not high_risk.empty:
                st.dataframe(high_risk[['customer_id', 'churn_probability', 'recency_days', 
                                       'purchase_frequency', 'customer_lifetime_value']])
                download_dataframe(high_risk, "high_churn_risk_customers.csv")
            else:
                st.info("No customers identified with high churn risk.")
    
    # CLV Prediction Tab
    with tabs[1]:
        st.subheader("Customer Lifetime Value Prediction")
        
        st.write("""
        Predict the future value of customers based on their current behavior.
        This helps you identify customers with high growth potential.
        """)
        
        if st_session_state.clv_model is None:
            if st.button("Train CLV Prediction Model"):
                with st.spinner("Training CLV prediction model... This may take a moment."):
                    model, features, r2, feature_importance = train_clv_prediction_model(
                        st_session_state.processed_data
                    )
                    
                    if model is not None:
                        st_session_state.clv_model = model
                        st_session_state.clv_features = features
                        st_session_state.clv_r2 = r2
                        st_session_state.clv_feature_importance = feature_importance
                        
                        # Generate predictions
                        customer_data = st_session_state.processed_data.drop_duplicates(subset=['customer_id'])
                        st_session_state.clv_predictions = predict_future_clv(
                            model, features, customer_data
                        )
                        
                        st.success(f"CLV prediction model trained successfully! Model R² score: {r2:.2f}")
                    else:
                        st.error("Could not train CLV prediction model. Insufficient data or too many missing values.")
        else:
            st.success(f"CLV prediction model is ready. Model R² score: {st_session_state.clv_r2:.2f}")
            
            # Show CLV analysis
            clv_charts = plot_clv_prediction_analysis(
                st_session_state.clv_predictions, 
                st_session_state.clv_feature_importance
            )
            
            for chart in clv_charts:
                st.plotly_chart(chart, use_container_width=True)
            
            # Show customers with high growth potential
            st.subheader("Customers with Highest Growth Potential")
            high_potential = st_session_state.clv_predictions[
                st_session_state.clv_predictions['growth_potential'] == 'High'
            ].sort_values('clv_growth_potential', ascending=False).head(10)
            
            if not high_potential.empty:
                st.dataframe(high_potential[[
                    'customer_id', 'customer_lifetime_value', 'predicted_future_clv', 
                    'clv_growth_potential', 'purchase_frequency'
                ]])
                download_dataframe(high_potential, "high_growth_potential_customers.csv")
            else:
                st.info("No customers identified with high growth potential.")


def create_market_basket_section(st, st_session_state):
    """
    Create the Market Basket Analysis section of the app.
    
    Args:
        st: Streamlit module
        st_session_state: Streamlit session state
    """
    import pandas as pd
    from market_basket_analysis import (
        perform_market_basket_analysis, plot_market_basket_results, 
        generate_product_recommendations
    )
    from utils import download_dataframe
    
    st.title("Market Basket Analysis")
    
    st.write("""
    Market Basket Analysis discovers relationships between products based on how frequently they're purchased together.
    This can help with product recommendations, store layouts, and promotional strategies.
    """)
    
    # Run Market Basket Analysis
    if st_session_state.mba_results is None:
        col1, col2 = st.columns(2)
        
        with col1:
            min_support = st.slider(
                "Minimum Support (%)", 
                min_value=1, 
                max_value=20, 
                value=5,
                help="Minimum percentage of transactions that must contain the itemset"
            ) / 100
            
        with col2:
            min_confidence = st.slider(
                "Minimum Confidence (%)", 
                min_value=10, 
                max_value=90, 
                value=30,
                help="Minimum probability that Y is purchased when X is purchased"
            ) / 100
        
        if st.button("Perform Market Basket Analysis"):
            with st.spinner("Analyzing purchase patterns... This may take a moment."):
                mba_results = perform_market_basket_analysis(
                    st_session_state.processed_data,
                    min_support=min_support,
                    min_confidence=min_confidence
                )
                
                st_session_state.mba_results = mba_results
                st_session_state.mba_params = {
                    'min_support': min_support,
                    'min_confidence': min_confidence
                }
                
                st.success("Market Basket Analysis completed!")
    else:
        # Display current parameters
        st.info(f"Analysis performed with min_support={st_session_state.mba_params['min_support']:.2f} and " +
                f"min_confidence={st_session_state.mba_params['min_confidence']:.2f}")
        
        if st.button("Rerun with Different Parameters"):
            st_session_state.mba_results = None
            st.rerun()
    
    # Display MBA results if available
    if st_session_state.mba_results is not None:
        # Display visualization results
        mba_charts = plot_market_basket_results(st_session_state.mba_results)
        
        for chart in mba_charts:
            st.plotly_chart(chart, use_container_width=True)
        
        # Product recommendation tool
        st.subheader("Product Recommendation Tool")
        
        st.write("Select a customer to get personalized product recommendations based on their purchase history:")
        
        # Get unique customers
        customers = st_session_state.processed_data['customer_id'].unique().tolist()
        
        if customers:
            selected_customer = st.selectbox("Select Customer", customers)
            
            if st.button("Generate Recommendations"):
                with st.spinner("Generating recommendations..."):
                    recommendations = generate_product_recommendations(
                        selected_customer,
                        st_session_state.processed_data,
                        st_session_state.mba_results
                    )
                    
                    if recommendations:
                        st.subheader(f"Recommended Products for {selected_customer}")
                        
                        # Show purchase history
                        customer_history = st_session_state.processed_data[
                            st_session_state.processed_data['customer_id'] == selected_customer
                        ]['product_category'].unique().tolist()
                        
                        st.write("**Purchase History:**", ", ".join(customer_history))
                        
                        # Show recommendations
                        rec_df = pd.DataFrame(recommendations)
                        st.dataframe(rec_df[['product', 'confidence', 'lift', 'based_on']])
                    else:
                        st.info("Could not generate recommendations for this customer. They may not have enough purchase history.")
        else:
            st.info("No customer data available.")
            
        # Download association rules
        if 'rules' in st_session_state.mba_results:
            st.subheader("Download Association Rules")
            download_dataframe(st_session_state.mba_results['rules'], "association_rules.csv")


def create_animations_section(st, st_session_state):
    """
    Create the Animations & Visualizations section of the app.
    
    Args:
        st: Streamlit module
        st_session_state: Streamlit session state
    """
    from animation import (
        create_animated_sales_trend, create_animated_customer_growth,
        create_animated_customer_segments, create_geographic_animation
    )
    
    st.title("Interactive Animations & Visualizations")
    
    st.write("""
    This section features animated visualizations that help you better understand
    customer behavior trends and patterns over time.
    """)
    
    # Tabs for different animations
    tabs = st.tabs([
        "Sales Growth Animation", 
        "Customer Growth Animation", 
        "Geographic Sales Map",
        "Customer Segment Evolution"
    ])
    
    # Sales Growth Animation
    with tabs[0]:
        st.subheader("Animated Sales Growth Over Time")
        
        st.write("""
        Watch how sales for different product categories evolve over time.
        This animation shows cumulative sales growth by category.
        """)
        
        sales_animation = create_animated_sales_trend(st_session_state.processed_data)
        st.plotly_chart(sales_animation, use_container_width=True)
    
    # Customer Growth Animation
    with tabs[1]:
        st.subheader("Customer Acquisition Growth")
        
        st.write("""
        This animation shows the growth in customer base over time,
        displaying both new customer acquisition and cumulative customer count.
        """)
        
        customer_animation = create_animated_customer_growth(st_session_state.processed_data)
        st.plotly_chart(customer_animation, use_container_width=True)
    
    # Geographic Sales Map
    with tabs[2]:
        st.subheader("Geographic Sales Distribution")
        
        st.write("""
        This animated map shows how sales are distributed geographically over time.
        Watch how sales volume evolves across different locations.
        """)
        
        geo_animation = create_geographic_animation(st_session_state.processed_data)
        st.plotly_chart(geo_animation, use_container_width=True)
    
    # Customer Segment Evolution
    with tabs[3]:
        st.subheader("Customer Segment Evolution")
        
        st.write("""
        If you've performed customer segmentation, this animation shows how
        the distribution of customer segments has evolved over time.
        """)
        
        if st_session_state.segmentation_done:
            segment_col = 'segment_label'
            segment_animation = create_animated_customer_segments(
                st_session_state.segmentation_results['customer_segments'],
                segment_col
            )
            st.plotly_chart(segment_animation, use_container_width=True)
        elif st_session_state.rfm_done:
            segment_animation = create_animated_customer_segments(
                st_session_state.rfm_data,
                'RFM_Segment'
            )
            st.plotly_chart(segment_animation, use_container_width=True)
        else:
            st.info("Please perform customer segmentation or RFM analysis first to see segment evolution.")