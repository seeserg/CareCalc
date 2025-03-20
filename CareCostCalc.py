import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

# Set Streamlit page configuration
st.set_page_config(
    page_title="Care Cost & Trust Fund Calculator",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS styles
st.markdown(
    """
    <style>
        .main .block-container {
            padding-top: 2rem;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .metric-container {
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 4px;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2c3e50;
        }
        .sufficient {
            color: #27ae60 !important;
        }
        .insufficient {
            color: #e74c3c !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title("Care Cost & Trust Fund Calculator")
st.markdown("Calculate long-term care costs and trust fund projections")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Calculator", "📋 Detailed Projections", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Parameters")
        current_age = st.number_input("Current Age", min_value=0, max_value=120, value=54, step=1)
        life_expectancy = st.number_input("Projected Lifespan (years)", min_value=1, max_value=50, value=20, step=1)
        
        annual_cost = st.number_input("Annual Care Cost ($)", min_value=0, value=57000, step=1000)
        inflation_rate = st.slider("General Inflation Rate (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
        care_increase_rate = st.slider("Additional Annual Increase in Care Needs (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
        ssi_annual = st.number_input("Annual SSI/Other Benefits ($)", min_value=0, value=10800, step=100)
        ssi_increase_rate = st.slider("SSI/Benefits Increase Rate (%)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
        
        trust_amount = st.number_input("Trust Fund Amount ($)", min_value=0, value=1000000, step=10000)
        investment_return = st.slider("Investment Return Rate (%)", min_value=0.0, max_value=12.0, value=5.0, step=0.1)
        
        calculate_button = st.button("Calculate Projections")
    
    @st.cache_data(ttl=600)
    def calculate_projections(*args):
        """Calculate cost projections and trust fund sufficiency."""
        (current_age, annual_cost, inflation_rate, care_increase_rate,
         trust_amount, investment_return, life_expectancy, ssi_annual, ssi_increase_rate) = args
        
        inflation_rate /= 100
        care_increase_rate /= 100
        investment_return /= 100
        ssi_increase_rate /= 100
        
        years = list(range(1, life_expectancy + 1))
        ages = list(range(current_age, current_age + life_expectancy))
        annual_expenses, annual_ssi, net_costs, trust_balances = [], [], [], []
        
        remaining_trust = trust_amount
        annual_expense, current_ssi = annual_cost, ssi_annual
        deplete_year = None
        
        for year in years:
            annual_expenses.append(annual_expense)
            annual_ssi.append(current_ssi)
            net_cost = max(0, annual_expense - current_ssi)
            net_costs.append(net_cost)
            
            remaining_trust = (remaining_trust * (1 + investment_return)) - net_cost
            trust_balances.append(max(0, remaining_trust))
            
            if remaining_trust < 0 and deplete_year is None:
                deplete_year = year
                
            annual_expense *= (1 + inflation_rate + care_increase_rate)
            current_ssi *= (1 + ssi_increase_rate)
        
        return {
            'years': years,
            'ages': ages,
            'annual_expenses': annual_expenses,
            'annual_ssi': annual_ssi,
            'net_costs': net_costs,
            'trust_balances': trust_balances,
            'deplete_year': deplete_year
        }
    
    if calculate_button or 'results' not in st.session_state:
        st.session_state.results = calculate_projections(
            current_age, annual_cost, inflation_rate, care_increase_rate, 
            trust_amount, investment_return, life_expectancy, ssi_annual, ssi_increase_rate
        )
    
    results = st.session_state.results
    
    with col2:
        st.subheader("Trust Fund Projection")
        
        df = pd.DataFrame({
            'Year': results['years'],
            'Age': results['ages'],
            'Annual Cost': results['annual_expenses'],
            'SSI Benefits': results['annual_ssi'],
            'Net Cost': results['net_costs'],
            'Trust Balance': results['trust_balances']
        })
        
        fig = px.line(df, x='Age', y=['Annual Cost', 'SSI Benefits', 'Net Cost'], 
                      labels={'value': 'Amount ($)', 'Age': 'Age'},
                      title='Annual Cost & Benefits Projection')
        
        st.plotly_chart(fig, use_container_width=True)
        
        fig2 = px.line(df, x='Age', y='Trust Balance', 
                       labels={'Trust Balance': 'Trust Fund Balance ($)', 'Age': 'Age'},
                       title='Trust Fund Balance Over Time')
        
        st.plotly_chart(fig2, use_container_width=True)
        
        if results['deplete_year']:
            st.warning(f"⚠️ Trust fund is expected to deplete in Year {results['deplete_year']}.")
        else:
            st.success("✅ Trust fund is sufficient for the projected timeframe.")

with tab2:
    st.subheader("Detailed Breakdown")
    st.dataframe(df.style.format("${:,.2f}"))

with tab3:
    st.markdown("### About This Calculator")
    st.markdown("This tool helps estimate long-term care costs and project trust fund sustainability.")
