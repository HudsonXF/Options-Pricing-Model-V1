import streamlit as st
import pandas as pd
import numpy as np
import math # Import math for sqrt, exp, log, pi, erf
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

# --- Manual Implementations of Standard Normal CDF and PDF ---

def manual_norm_cdf(x):
    """
    Calculates the Cumulative Distribution Function (CDF) for a standard normal distribution.
    phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    """
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def manual_norm_pdf(x):
    """
    Calculates the Probability Density Function (PDF) for a standard normal distribution.
    phi'(x) = (1 / sqrt(2 * pi)) * exp(-x^2 / 2)
    """
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x**2)

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

        # Ensure volatility and time_to_maturity are not zero to prevent division by zero
        # In a real app, you might want more robust error handling or input validation
        if volatility <= 0 or time_to_maturity <= 0:
            st.error("Volatility and Time to Maturity must be greater than zero.")
            self.call_price = 0.0
            self.put_price = 0.0
            self.call_delta = self.put_delta = self.call_gamma = self.put_gamma = 0.0
            self.call_theta = self.put_theta = self.vega = self.call_rho = self.put_rho = 0.0
            return 0.0, 0.0

        d1 = (
            math.log(current_price / strike) +
            (interest_rate + 0.5 * volatility ** 2) * time_to_maturity
            ) / (
                volatility * math.sqrt(time_to_maturity)
            )
        d2 = d1 - volatility * math.sqrt(time_to_maturity)

        # Use manual_norm_cdf instead of norm.cdf
        call_price = current_price * manual_norm_cdf(d1) - (
            strike * math.exp(-(interest_rate * time_to_maturity)) * manual_norm_cdf(d2)
        )
        put_price = (
            strike * math.exp(-(interest_rate * time_to_maturity)) * manual_norm_cdf(-d2)
        ) - current_price * manual_norm_cdf(-d1)

        self.call_price = call_price
        self.put_price = put_price

        # GREEKS - using manual_norm_pdf and manual_norm_cdf
        # Delta
        self.call_delta = manual_norm_cdf(d1)
        self.put_delta = manual_norm_cdf(d1) - 1 

        # Gamma
        self.call_gamma = manual_norm_pdf(d1) / (
            current_price * volatility * math.sqrt(time_to_maturity)
        )
        self.put_gamma = self.call_gamma

        # Theta (approximated for daily change, then annualized)
        self.call_theta = (
            (-current_price * manual_norm_pdf(d1) * volatility / (2 * math.sqrt(time_to_maturity)))
            - (interest_rate * strike * math.exp(-interest_rate * time_to_maturity) * manual_norm_cdf(d2))
        ) / 365 # Daily theta
        self.put_theta = (
            (-current_price * manual_norm_pdf(d1) * volatility / (2 * math.sqrt(time_to_maturity)))
            + (interest_rate * strike * math.exp(-interest_rate * time_to_maturity) * manual_norm_cdf(-d2))
        ) / 365 # Daily theta

        # Vega
        self.vega = current_price * manual_norm_pdf(d1) * math.sqrt(time_to_maturity) / 100 # Change for 1% vol change

        # Rho (approximated for 1% change in interest rate)
        self.call_rho = (strike * time_to_maturity * math.exp(-interest_rate * time_to_maturity) * manual_norm_cdf(d2)) / 100
        self.put_rho = (-strike * time_to_maturity * math.exp(-interest_rate * time_to_maturity) * manual_norm_cdf(-d2)) / 100

        return call_price, put_price

def plot_heatmap(bs_model, spot_range, vol_range, strike):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))
    
    # Handle cases where bs_model might have invalid parameters, resulting in 0 prices
    if bs_model.volatility <= 0 or bs_model.time_to_maturity <= 0:
        fig_call, ax_call = plt.subplots(figsize=(10, 8))
        ax_call.text(0.5, 0.5, "Invalid parameters for heatmap generation.", horizontalalignment='center', verticalalignment='center', transform=ax_call.transAxes, fontsize=12, color='red')
        ax_call.set_title('CALL Option Price (Error)', fontsize=14)
        ax_call.axis('off')
        plt.close(fig_call) # Close figure to prevent displaying empty plot

        fig_put, ax_put = plt.subplots(figsize=(10, 8))
        ax_put.text(0.5, 0.5, "Invalid parameters for heatmap generation.", horizontalalignment='center', verticalalignment='center', transform=ax_put.transAxes, fontsize=12, color='red')
        ax_put.set_title('PUT Option Price (Error)', fontsize=14)
        ax_put.axis('off')
        plt.close(fig_put) # Close figure to prevent displaying empty plot
        return fig_call, fig_put


    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            # Ensure temporary model also has valid parameters for calculation
            if vol > 0 and bs_model.time_to_maturity > 0:
                bs_temp = BlackScholes(
                    time_to_maturity=bs_model.time_to_maturity,
                    strike=strike,
                    current_price=spot,
                    volatility=vol,
                    interest_rate=bs_model.interest_rate
                )
                bs_temp.calculate_prices()
                call_prices[i, j] = bs_temp.call_pr
