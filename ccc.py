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

@st.cache_data(ttl=600)
def calculate_projections(
    current_age: int,
    annual_cost: float,
    inflation_rate: float,
    care_increase_rate: float,
    trust_amount: float,
    investment_return: float,
    life_expectancy: int,
    ssi_annual: float,
    ssi_increase_rate: float
):
    """Calculate cost projections and trust fund sufficiency."""
    # Convert percentages to decimals
    inflation = inflation_rate / 100
    care_inc = care_increase_rate / 100
    inv_return = investment_return / 100
    ssi_inc = ssi_increase_rate / 100

    years = list(range(1, life_expectancy + 1))
    ages = list(range(current_age, current_age + life_expectancy))
    annual_expenses, annual_ssi, net_costs, trust_balances = [], [], [], []

    remaining_trust = trust_amount
    annual_expense = annual_cost
    current_ssi = ssi_annual
    deplete_year = None

    for year in years:
        annual_expenses.append(annual_expense)
        annual_ssi.append(current_ssi)

        net_cost = max(0, annual_expense - current_ssi)
        net_costs.append(net_cost)

        # Grow trust by inv_return, then subtract net cost
        remaining_trust = (remaining_trust * (1 + inv_return)) - net_cost
        trust_balances.append(max(0, remaining_trust))

        # Capture first depletion moment
        if remaining_trust < 0 and deplete_year is None:
            deplete_year = year

        # Update for next year
        annual_expense *= (1 + inflation + care_inc)
        current_ssi *= (1 + ssi_inc)

    return {
        "years": years,
        "ages": ages,
        "annual_expenses": annual_expenses,
        "annual_ssi": annual_ssi,
        "net_costs": net_costs,
        "trust_balances": trust_balances,
        "deplete_year": deplete_year,
    }


def run_monte_carlo(
    current_age,
    annual_cost,
    inflation_rate,
    care_increase_rate,
    trust_amount,
    investment_return,
    life_expectancy,
    ssi_annual,
    ssi_increase_rate,
    num_simulations=500,
    variation=2.0
):
    """
    Perform a simple Monte Carlo simulation by randomly varying inflation rate,
    investment return, care increase rate, and ssi increase rate within +/- 'variation'.
    Return a DataFrame of simulation outcomes.
    """
    # We'll do uniform distribution around the user-provided rates
    sim_results = []

    for i in range(num_simulations):
        # Randomly vary each parameter within +/- variation
        sim_inflation = np.random.uniform(
            max(0, inflation_rate - variation), inflation_rate + variation
        )
        sim_inv_return = np.random.uniform(
            max(0, investment_return - variation), investment_return + variation
        )
        sim_care_inc = np.random.uniform(
            max(0, care_increase_rate - variation), care_increase_rate + variation
        )
        sim_ssi_inc = np.random.uniform(
            max(0, ssi_increase_rate - variation), ssi_increase_rate + variation
        )

        # Calculate projections with random parameters
        res = calculate_projections(
            current_age,
            annual_cost,
            sim_inflation,
            sim_care_inc,
            trust_amount,
            sim_inv_return,
            life_expectancy,
            ssi_annual,
            sim_ssi_inc,
        )

        final_balance = res["trust_balances"][-1]
        depletion = res["deplete_year"] is not None

        sim_results.append(
            {
                "Simulation": i + 1,
                "Inflation": sim_inflation,
                "Investment Return": sim_inv_return,
                "Care Increase": sim_care_inc,
                "SSI Increase": sim_ssi_inc,
                "Final Balance": final_balance,
                "Depletion Year": res["deplete_year"],
                "Is Depleted": depletion,
            }
        )

    df_sim = pd.DataFrame(sim_results)
    return df_sim

with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Parameters")
        current_age = st.number_input("Current Age", min_value=0, max_value=120, value=54, step=1)
        base_life_expectancy = st.number_input("Projected Lifespan (years)", min_value=1, max_value=50, value=20, step=1)

        # Add a health & mental health checklist
        with st.expander("Health & Mental Health Factors"):
            st.write("Select any applicable factors that may adjust life expectancy.")
            smokes = st.checkbox("Smoking")
            obesity = st.checkbox("Obesity")
            chronic_illness = st.checkbox("Chronic Illness (e.g. Diabetes)")
            heart_disease = st.checkbox("History of Cardiovascular Disease")
            severe_mental = st.checkbox("Severe Mental Health Condition")
            exercise = st.checkbox("Regular Exercise")
            balanced_diet = st.checkbox("Balanced Diet")

            life_expectancy_offset = 0

            if smokes:
                life_expectancy_offset -= 3
            if obesity:
                life_expectancy_offset -= 2
            if chronic_illness:
                life_expectancy_offset -= 2
            if heart_disease:
                life_expectancy_offset -= 2
            if severe_mental:
                life_expectancy_offset -= 1
            if exercise:
                life_expectancy_offset += 2
            if balanced_diet:
                life_expectancy_offset += 1

            adjusted_life_expectancy = base_life_expectancy + life_expectancy_offset
            adjusted_life_expectancy = max(1, adjusted_life_expectancy)
            st.write(f"**Adjusted Life Expectancy:** {adjusted_life_expectancy} years")

        annual_cost = st.number_input("Annual Care Cost ($)", min_value=0, value=57000, step=1000)
        inflation_rate = st.slider("General Inflation Rate (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
        care_increase_rate = st.slider(
            "Additional Annual Increase in Care Needs (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1
        )
        ssi_annual = st.number_input("Annual SSI/Other Benefits ($)", min_value=0, value=10800, step=100)
        ssi_increase_rate = st.slider("SSI/Benefits Increase Rate (%)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)

        trust_amount = st.number_input("Trust Fund Amount ($)", min_value=0, value=1000000, step=10000)
        investment_return = st.slider("Investment Return Rate (%)", min_value=0.0, max_value=12.0, value=5.0, step=0.1)

        calculate_button = st.button("Calculate Projections")

    if calculate_button or "results" not in st.session_state:
        st.session_state.results = calculate_projections(
            current_age,
            annual_cost,
            inflation_rate,
            care_increase_rate,
            trust_amount,
            investment_return,
            adjusted_life_expectancy,
            ssi_annual,
            ssi_increase_rate,
        )

    results = st.session_state.results

    with col2:
        # Key Metrics
        total_cost = sum(results["net_costs"])
        final_balance = results["trust_balances"][-1]
        is_depleted = results["deplete_year"] is not None

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Total Projected Cost", f"${total_cost:,.2f}")
        metric_col2.metric("Final Trust Balance", f"${final_balance:,.2f}")

        if is_depleted:
            st.warning(f"⚠️ Trust fund is expected to deplete in Year {results['deplete_year']}.")
        else:
            st.success("✅ Trust fund is sufficient for the projected timeframe.")

        # Visualization
        st.subheader("Trust Fund Projection")
        df = pd.DataFrame(
            {
                "Year": results["years"],
                "Age": results["ages"],
                "Annual Cost": results["annual_expenses"],
                "SSI Benefits": results["annual_ssi"],
                "Net Cost": results["net_costs"],
                "Trust Balance": results["trust_balances"],
            }
        )

        fig = px.line(
            df,
            x="Age",
            y=["Annual Cost", "SSI Benefits", "Net Cost"],
            labels={"value": "Amount ($)", "Age": "Age"},
            title="Annual Cost & Benefits Projection",
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(
            df,
            x="Age",
            y="Trust Balance",
            labels={"Trust Balance": "Trust Fund Balance ($)", "Age": "Age"},
            title="Trust Fund Balance Over Time",
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Detailed Breakdown")

    # Access the previously created DataFrame
    if "results" in st.session_state:
        results = st.session_state.results
        df = pd.DataFrame(
            {
                "Year": results["years"],
                "Age": results["ages"],
                "Annual Cost": results["annual_expenses"],
                "SSI Benefits": results["annual_ssi"],
                "Net Cost": results["net_costs"],
                "Trust Balance": results["trust_balances"],
            }
        )
        st.dataframe(df.style.format("${:,.2f}"), use_container_width=True)

        # Export Button
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv_data,
            file_name="care_cost_projections.csv",
            mime="text/csv",
        )

        # Add Monte Carlo Simulation
        with st.expander("Monte Carlo Analysis"):
            st.write(
                """This simulation runs multiple scenarios with random variations in key parameters (within ±2% by default)
                to estimate the probability of trust fund depletion and the range of possible outcomes."""
            )

            num_sims = st.slider("Number of Simulations", min_value=100, max_value=2000, value=500, step=100)
            variation = st.slider("Parameter Variation (+/- %)", min_value=1.0, max_value=5.0, value=2.0, step=0.5)

            run_mc = st.button("Run Monte Carlo")

            if run_mc:
                df_sim = run_monte_carlo(
                    current_age,
                    annual_cost,
                    inflation_rate,
                    care_increase_rate,
                    trust_amount,
                    investment_return,
                    results["years"][-1] - results["years"][0] + 1,  # same timespan used in calculation
                    ssi_annual,
                    ssi_increase_rate,
                    num_simulations=num_sims,
                    variation=variation,
                )

                # Probability of depletion
                depleted_count = df_sim["Is Depleted"].sum()
                depletion_prob = depleted_count / num_sims * 100

                st.write(f"**Probability of Depletion:** {depletion_prob:.2f}%")

                # Plot histogram of Final Balance
                fig_mc = px.histogram(
                    df_sim,
                    x="Final Balance",
                    nbins=30,
                    title="Distribution of Final Trust Balance",
                    labels={"Final Balance": "Final Balance ($)"},
                )
                # Add a vertical line for 0 (fully depleted)
                fig_mc.add_vline(
                    x=0,
                    line_width=2,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Depletion",
                )
                st.plotly_chart(fig_mc, use_container_width=True)

                # Download simulation results
                csv_sim = df_sim.to_csv(index=False)
                st.download_button(
                    "Download Monte Carlo Results (CSV)",
                    csv_sim,
                    "monte_carlo_results.csv",
                    "text/csv",
                )
    else:
        st.info("Please run calculations in the Calculator tab first.")

with tab3:
    st.markdown("### About This Calculator")
    st.markdown(
        "This tool helps estimate long-term care costs and project trust fund sustainability. "
        "Use the Calculator tab to enter assumptions and view the results. The Detailed Projections tab "
        "provides an annual breakdown, CSV export, and an optional Monte Carlo analysis."
    )
