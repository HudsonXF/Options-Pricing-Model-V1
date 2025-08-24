pip install scipy
import streamlit as st
import pandas as pd
import numpy as np
from np.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp
import matplotlib.pyplot as plt
import seaborn as sns

#######################
# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")


# Custom CSS to inject into Streamlit
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* Overall page styling */
.main {
    background-color: #f0f2f6; /* Light gray background */
}

/* Sidebar styling */
.css-1d391kg, .css-1lcbmhc { /* Targeting sidebar container */
    background-color: #ffffff; /* White background for sidebar */
    padding: 20px;
    border-right: 1px solid #e0e0e0;
    box-shadow: 2px 0 5px rgba(0,0,0,0.05);
}

/* Main title styling */
h1 {
    color: #2c3e50; /* Dark blue-gray */
    font-weight: 700;
    text-align: center;
    margin-bottom: 20px;
    padding-top: 15px;
}

h2 {
    color: #34495e;
    font-weight: 600;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 10px;
    margin-top: 30px;
}

h3 {
    color: #555;
    font-weight: 600;
}

/* Adjust the size and alignment of the CALL and PUT value containers */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 15px; /* Adjust the padding to control height */
    margin: 10px auto; /* Center the container with some vertical margin */
    border-radius: 12px; /* More rounded corners */
    box-shadow: 0 4px 12px rgba(0,0,0,0.1); /* Soft shadow */
    transition: transform 0.2s ease-in-out;
    height: 100px; /* Fixed height for uniformity */
}

.metric-container:hover {
    transform: translateY(-3px); /* Slight lift on hover */
}

/* Custom classes for CALL and PUT values */
.metric-call {
    background-color: #e6ffe6; /* Very light green background */
    color: #1a5e1a; /* Darker green font color */
    border: 1px solid #90ee90;
}

.metric-put {
    background-color: #ffe6e6; /* Very light red background */
    color: #8c2a2a; /* Darker red font color */
    border: 1px solid #ffcccb;
}

/* Style for the value text */
.metric-value {
    font-size: 2.2rem; /* Larger font size for prominence */
    font-weight: 700;
    margin: 0;
    text-align: center;
}

/* Style for the label text */
.metric-label {
    font-size: 1.1rem; /* Adjust font size */
    font-weight: 600;
    margin-bottom: 5px; /* Spacing between label and value */
    text-align: center;
    color: #555; /* Neutral color for label */
}

/* Input table styling */
.st-emotion-cache-zt5ig { /* Targeting the dataframe display widget */
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    overflow: hidden; /* Ensures rounded corners apply to content */
    margin-bottom: 25px;
}
.st-emotion-cache-zt5ig thead th {
    background-color: #eaf1f7; /* Light blue header */
    color: #333;
    font-weight: 600;
    padding: 12px 15px;
    border-bottom: 1px solid #d0d0d0;
}
.st-emotion-cache-zt5ig tbody td {
    padding: 10px 15px;
    border-bottom: 1px solid #f0f0f0;
}

/* Info box styling */
.st-emotion-cache-1f190u4 { /* Targeting the st.info container */
    background-color: #e0f2f7; /* Light blue */
    border-left: 5px solid #2196f3; /* Blue border */
    color: #212121;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 20px;
}

/* Button styling */
.stButton>button {
    width: 100%;
    padding: 10px 20px;
    background-color: #4CAF50; /* Green */
    color: white;
    border-radius: 8px;
    border: none;
    font-size: 1.1rem;
    font-weight: 600;
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    transition: background-color 0.3s, transform 0.2s;
}
.stButton>button:hover {
    background-color: #45a049;
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

class BlackScholes:
    def __init__(
        self,
        time_to_maturity: float,
        strike: float,
        current_price: float,
        volatility: float,
        interest_rate: float,
    ):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def calculate_prices(
        self,
    ):
        time_to_maturity = self.time_to_maturity
        strike = self.strike
        current_price = self.current_price
        volatility = self.volatility
        interest_rate = self.interest_rate

        d1 = (
            log(current_price / strike) +
            (interest_rate + 0.5 * volatility ** 2) * time_to_maturity
            ) / (
                volatility * sqrt(time_to_maturity)
            )
        d2 = d1 - volatility * sqrt(time_to_maturity)

        call_price = current_price * norm.cdf(d1) - (
            strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(d2)
        )
        put_price = (
            strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(-d2)
        ) - current_price * norm.cdf(-d1)

        self.call_price = call_price
        self.put_price = put_price

        # GREEKS
        # Delta
        self.call_delta = norm.cdf(d1)
        self.put_delta = norm.cdf(d1) - 1 # Corrected formula for put delta

        # Gamma
        # norm.pdf(d1) is the probability density function (N'(d1))
        # Note: Gamma is the same for Call and Put
        self.call_gamma = norm.pdf(d1) / (
            current_price * volatility * sqrt(time_to_maturity)
        )
        self.put_gamma = self.call_gamma

        # Theta (approximated for daily change, then annualized)
        # Note: Theta can be complex, this is a common approximation
        self.call_theta = (
            (-current_price * norm.pdf(d1) * volatility / (2 * sqrt(time_to_maturity)))
            - (interest_rate * strike * exp(-interest_rate * time_to_maturity) * norm.cdf(d2))
        ) / 365 # Daily theta
        self.put_theta = (
            (-current_price * norm.pdf(d1) * volatility / (2 * sqrt(time_to_maturity)))
            + (interest_rate * strike * exp(-interest_rate * time_to_maturity) * norm.cdf(-d2))
        ) / 365 # Daily theta

        # Vega
        self.vega = current_price * norm.pdf(d1) * sqrt(time_to_maturity) / 100 # Change for 1% vol change

        # Rho (approximated for 1% change in interest rate)
        self.call_rho = (strike * time_to_maturity * exp(-interest_rate * time_to_maturity) * norm.cdf(d2)) / 100
        self.put_rho = (-strike * time_to_maturity * exp(-interest_rate * time_to_maturity) * norm.cdf(-d2)) / 100


        return call_price, put_price

def plot_heatmap(bs_model, spot_range, vol_range, strike):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))

    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            bs_temp.calculate_prices()
            call_prices[i, j] = bs_temp.call_price
            put_prices[i, j] = bs_temp.put_price

    # Plotting Call Price Heatmap
    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2),
                annot=True, fmt=".2f", cmap="viridis", ax=ax_call, linewidths=.5, linecolor='gray')
    ax_call.set_title('CALL Option Price', fontsize=14)
    ax_call.set_xlabel('Spot Price', fontsize=12)
    ax_call.set_ylabel('Volatility', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Plotting Put Price Heatmap
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2),
                annot=True, fmt=".2f", cmap="magma", ax=ax_put, linewidths=.5, linecolor='gray')
    ax_put.set_title('PUT Option Price', fontsize=14)
    ax_put.set_xlabel('Spot Price', fontsize=12)
    ax_put.set_ylabel('Volatility', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    return fig_call, fig_put


# Sidebar for User Inputs
with st.sidebar:
    st.title("📊 Black-Scholes Model")
    st.markdown("---")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/mprudhvi/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Prudhvi Reddy, Muppala`</a>', unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("Option Parameters")
    current_price = st.number_input("Current Asset Price (S)", value=100.0, min_value=0.01)
    strike = st.number_input("Strike Price (K)", value=100.0, min_value=0.01)
    time_to_maturity = st.number_input("Time to Maturity (T in Years)", value=1.0, min_value=0.01)
    volatility = st.number_input("Volatility (σ - Annualized)", value=0.2, min_value=0.01, max_value=1.0)
    interest_rate = st.number_input("Risk-Free Interest Rate (r - Annualized)", value=0.05, min_value=0.0)

    st.markdown("---")
    st.subheader("Heatmap Settings")

    # Use st.expander for heatmap specific settings to keep sidebar clean
    with st.expander("Adjust Heatmap Ranges"):
        spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
        spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
        vol_min = st.slider('Min Volatility', min_value=0.01, max_value=1.0, value=volatility*0.5, step=0.01)
        vol_max = st.slider('Max Volatility', min_value=0.01, max_value=1.0, value=volatility*1.5, step=0.01)

        spot_range = np.linspace(spot_min, spot_max, 10) # 10 points for readability
        vol_range = np.linspace(vol_min, vol_max, 10) # 10 points for readability


# Main Page for Output Display
st.title("Black-Scholes Option Pricing Model")
st.markdown("A dynamic tool to calculate option prices and visualize their sensitivity to key market parameters.")

# Table of Inputs
st.markdown("## Input Parameters")
input_data = {
    "Parameter": ["Current Asset Price (S)", "Strike Price (K)", "Time to Maturity (T)", "Volatility (σ)", "Risk-Free Rate (r)"],
    "Value": [current_price, strike, time_to_maturity, volatility, interest_rate],
    "Unit": ["$", "$", "Years", "Annualized", "Annualized"]
}
input_df = pd.DataFrame(input_data)
st.dataframe(input_df.set_index('Parameter'), use_container_width=True)


# Calculate Call and Put values
bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
call_price, put_price = bs_model.calculate_prices()

# Display Call and Put Values in colored containers
st.markdown("---")
st.markdown("## Option Prices")
col_call, col_put = st.columns([1,1], gap="large")

with col_call:
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Option Value</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col_put:
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Option Value</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Display Greeks
st.markdown("---")
st.markdown("## Option Greeks")
st.info("The 'Greeks' measure the sensitivity of an option's price to changes in underlying parameters.")

greeks_data = {
    "Greek": ["Delta (Call)", "Delta (Put)", "Gamma", "Theta (Daily Call)", "Theta (Daily Put)", "Vega", "Rho (Call)", "Rho (Put)"],
    "Value": [bs_model.call_delta, bs_model.put_delta, bs_model.call_gamma, bs_model.call_theta, bs_model.put_theta, bs_model.vega, bs_model.call_rho, bs_model.put_rho],
    "Description": [
        "Sensitivity to asset price",
        "Sensitivity to asset price",
        "Sensitivity to Delta's change",
        "Sensitivity to passage of time",
        "Sensitivity to passage of time",
        "Sensitivity to volatility",
        "Sensitivity to interest rate",
        "Sensitivity to interest rate"
    ]
}
greeks_df = pd.DataFrame(greeks_data)
st.dataframe(greeks_df.set_index('Greek'), use_container_width=True)


st.markdown("---")
st.header("Interactive Option Price Heatmaps")
st.info("Explore how option prices fluctuate with varying 'Spot Prices and Volatility' levels, while maintaining a constant 'Strike Price'.")

# Interactive Sliders and Heatmaps for Call and Put Options
col_heatmap_call, col_heatmap_put = st.columns([1,1], gap="large")

with col_heatmap_call:
    st.subheader("Call Price Sensitivity")
    heatmap_fig_call, _ = plot_heatmap(bs_model, spot_range, vol_range, strike)
    st.pyplot(heatmap_fig_call)

with col_heatmap_put:
    st.subheader("Put Price Sensitivity")
    _, heatmap_fig_put = plot_heatmap(bs_model, spot_range, vol_range, strike)
    st.pyplot(heatmap_fig_put)
