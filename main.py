# main.py

import streamlit as st
import pandas as pd
import plotly.express as px
from models import collaborative_filtering, content_based_filtering
import random

# Streamlit app title and page configuration setup
st.set_page_config(page_title="computer_vission",
                   page_icon="👓", layout="wide")
st.title("Evaluation of Recommender System Effectiveness and Efficiency")

# Sidebar setup
st.sidebar.header("Upload your CSV file")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset")
    st.dataframe(df, use_container_width=True, height=270)

    st.sidebar.markdown("---")  # Add a separator line
    feature = st.sidebar.selectbox("Select a feature for user rating:", df.columns)
    
    if feature:
        st.sidebar.markdown("---")  # Add a separator line
        st.sidebar.subheader("Rate these values for evaluation:")
        
        # Ask the user to rate 10 random values from the selected feature
        random_samples = df[feature].dropna().sample(10, random_state=5).unique()
        user_ratings = {}
        
        # Create ratings in sidebar
        for value in random_samples:
            rating = st.sidebar.slider(f"Rate '{value}' (0-10):", min_value=0, max_value=10, value=5)
            user_ratings[value] = rating
        
        # Tabs for different model evaluations and recommendations
        tab1, tab2, tab3 = st.tabs(["Model 1: Collaborative Filtering", 
                                    "Model 2: Content-Based Filtering", 
                                    "Compare Models"])
        
        with tab1:
            st.header("Collaborative Filtering")
            effectiveness, efficiency, recommendations = collaborative_filtering(df, user_ratings)
            
            # Create two columns for metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Effectiveness", f"{effectiveness:.2f}")
            with col2:
                st.metric("Efficiency", f"{efficiency:.2f}")
            
            st.subheader("Recommendations")
            st.dataframe(recommendations, use_container_width=True, height=270)

            # Visualization
            fig = px.bar(x=['Effectiveness', 'Efficiency'], y=[effectiveness, efficiency], title="Model 1 Metrics")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("Content-Based Filtering")
            effectiveness, efficiency, recommendations = content_based_filtering(df, user_ratings)
            
            # Create two columns for metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Effectiveness", f"{effectiveness:.2f}")
            with col2:
                st.metric("Efficiency", f"{efficiency:.2f}")
            
            st.subheader("Recommendations")
            st.dataframe(recommendations, use_container_width=True, height=270)

            # Visualization
            fig = px.bar(x=['Effectiveness', 'Efficiency'], y=[effectiveness, efficiency], title="Model 2 Metrics")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("Comparison of Models")
            # Collect data for both models
            effectiveness1, efficiency1, _ = collaborative_filtering(df, user_ratings)
            effectiveness2, efficiency2, _ = content_based_filtering(df, user_ratings)
            
            comparison_df = pd.DataFrame({
                "Metric": ["Effectiveness", "Efficiency"],
                "Model 1": [effectiveness1, efficiency1],
                "Model 2": [effectiveness2, efficiency2]
            })
            
            st.subheader("Comparison Chart")
            st.dataframe(comparison_df, use_container_width=True, height=150)
            
            fig = px.line(comparison_df, x="Metric", y=["Model 1", "Model 2"], markers=True, title="Model Performance Comparison")
            st.plotly_chart(fig, use_container_width=True)



