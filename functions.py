"""
Live option pricer functions
"""

## Import modules
import numpy as np
from datetime import datetime
from scipy.stats import norm
import requests
from bs4 import BeautifulSoup
import pandas as pd

## Functions
# Function for calculating black scholes price of call or put
"""
input:
contract_type = {
    "call",
    "put"
    } # String of type of option
stock_price = {
    float
    } # Float of current stock price
K = {
     float
     } # Float of strike price of option
r = {
     float
     } # Float of interest rate
vol = {
       float
       } # Float of volatility of stock
tau = [
       float
       ] # Float of time to expiry
"""
def Black_Scholes_price(contract_type, stock_price, K, r, vol, tau):
    
    d1 = (np.log(stock_price/K) + (r + vol**2/2)*tau) / (vol * np.sqrt(tau))
    d2 = d1 - vol * np.sqrt(tau)
    
    if contract_type == "call":
        option_price = stock_price * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    elif contract_type == "put":
        option_price = -stock_price * norm.cdf(-d1) + K * np.exp(-r * tau) * norm.cdf(-d2)
        
    return option_price

# Function for checking if year is in leap year
"""
input:
expiry_date = {
    "YYYY-MM-DD"
    } # String of date of expiry of contract
"""
def check_if_leap_year(expiry_date):
    # convert date string to integer of year
    year = int(expiry_date[:4])
    
    # Check if leap year
    if (year%4 == 0 and (year&100 != 0 or year%400 == 0)):
        leap_year = True
    else:
        leap_year = False
    
    # Return whether it is leap year
    return leap_year

# Function for checking seconds in month
"""
input:
expiry_date = {
    "YYYY-MM-DD"
    } # String of date of expiry of contract
"""
def check_seconds_in_month(expiry_date):
    # Convert date string to integer month number
    month = int(expiry_date[5:7])
        
    # Find number of seconds in current month
    if month in [1,3,5,7,8,10,12]:
        seconds_in_month = 2678400
    elif month in [4,6,9,11]:
        seconds_in_month = 2592000
    else:
        # Check for leap year
        leap_year = check_if_leap_year(expiry_date) 
        # If leap year feb has 29 days, else 28
        if leap_year == True:
            seconds_in_month = 2505600
        else:
            seconds_in_month = 2419200
    
    # Return seconds in current month
    return seconds_in_month

# Function for calculating time to expiry with precision of nanoseconds wrt a specific timeframe
"""
input:
expiry_date = {
    "YYYY-MM-DD"
    } # String of date of expiry of contract
timeframe = {
    "years",
    "months",
    "weeks",
    "days",
    "hours",
    "minutes",
    "seconds"
    } # String of desired timeframe
"""
def time_to_expiry(expiry_date, timeframe="years"):
    # Get current date
    current_date_and_time = datetime.now()
    
    # Convert expiry date to datetime format
    expiry_date_and_time = expiry_date + " 23:59:59.999999"
    expiry_date_and_time = datetime.strptime(expiry_date_and_time, "%Y-%m-%d %H:%M:%S.%f")
    
    # Compute time to expiry
    time_to_expiry = expiry_date_and_time - current_date_and_time
    total_seconds = time_to_expiry.total_seconds()
    
    # If yearly timeframe
    if timeframe == "years":
        # Check for leap year
        leap_year = check_if_leap_year(expiry_date)
        
        # Seconds in (non) leap year
        if leap_year == True:
            seconds_in_year = 31622400
        else:
            seconds_in_year = 31536000
        
        # Time to expiry in years is:
        time_to_expiry = total_seconds / seconds_in_year
    
    # If monthly timeframe
    elif timeframe == "months":
        # Seconds in month is:
        seconds_in_month = check_seconds_in_month(expiry_date)
    
        # Time to expiry in months is:
        time_to_expiry = total_seconds / seconds_in_month    
    
    # If weekly timeframe
    elif timeframe == "weeks":
        # Seconds in week is:
        seconds_in_week = 604800
        
        # Time to expiry in weeks is:
        time_to_expiry = total_seconds / seconds_in_week   
    
    # If daily timeframe
    elif timeframe == "day":
        # Seconds in day is:
        seconds_in_day = 86400
        
        # Time to expiry in days is:
        time_to_expiry = total_seconds / seconds_in_day   
    
    # If hourly timeframe
    elif timeframe == "hours":
        # Seconds in hour is:
        seconds_in_hour = 3600
        
        # Time to expiry in hours is:
        time_to_expiry = total_seconds / seconds_in_hour    
    
    # If minute timeframe
    elif timeframe == "minutes":
        # Seconds in minute is:
        seconds_in_minute = 60
        
        # Time to expiry in minutes is:
        time_to_expiry = total_seconds / seconds_in_minute
    
    # If seconds timeframe
    elif timeframe == "seconds":
        # Time to expiry in seconds is:
        time_to_expiry = total_seconds

    # Return time to expiry in desired unit format
    return time_to_expiry    

# Function for estimating interest rate
"""
input:
None
"""
def get_current_yield_data():
    url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=2024"

    # Make connection to website
    response = requests.get(url)

    # Check if connection was succesfull
    if not response.status_code == 200:
        raise Exception("could not get bond data")

    # Read content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract yield information
    table = soup.find("table")
    rows = table.find_all("tr")

    maturities = rows[0].get_text().split("\n")
    mats = []
    for maturity in maturities:
        if "Mo" in maturity or "Yr" in maturity:
            mats.append(maturity)
    yields = rows[-1].get_text().split("\n")
    ylds = []
    for yld in yields:
        try:
            ylds.append(float(yld)/100)
        except:
            pass
    yield_data = pd.DataFrame(ylds, index=mats, columns=["yield"])
    
    # Return yield data
    return yield_data     
    
# Function for estimating interest rate from given maturity and yield data
"""
input:
yield_data = {
    1 Mo,
    2 Mo,
    3 Mo,
    4 Mo,
    6 Mo,
    1 Yr,
    2 Yr,
    3 Yr,
    5 Yr,
    7 Yr,
    10 Yr,
    20 Yr,
    30 Yr,
    } # Dataframe for current yield data
option_time_to_maturity = {
    float
    } # Float for time to maturity
"""
def estimate_interest_rate(yield_data, option_time_to_maturity):
    # Extract maturities in yr format and yields per maturity
    maturities = [float(yld.strip(" Mo"))/12 if " Mo" in yld else float(yld.strip(" Yr")) for yld in yield_data.index]
    yields = yield_data
    
    # Interpolate if expiry date in between smallest and largest bond maturity
    if option_time_to_maturity < maturities[-1] and option_time_to_maturity > maturities[0]:
        interpolated_yield = np.interp(option_time_to_maturity, maturities, yields)
        interest_rate = np.log(1 + interpolated_yield)
    # Else extrapolate
    elif option_time_to_maturity < maturities[0]:
        short_term_yield = yields.iloc[0]
        interest_rate = np.log(1 + short_term_yield)
    else:
        long_term_yield = yields.iloc[-1]
        interest_rate = np.log(1 + long_term_yield)

    # Return estimated interest rate
    return float(interest_rate.iloc[0])

# Function for estimating volatility based on historical data
"""
Input:
stock_data = {
    Dataframe
    } # Dataframe with at least the Adjusted close prices in column named "Adj Close"
time_to_expiry_yrs = {
    float
    } # Time to expiry of option in years
"""
def estimate_volatility_from_historical_data(stock_data, time_to_expiry_yrs):
    # Extract adjusted close data from historical data
    adj_close_data = stock_data["Adj Close"].ffill()
    
    # Transform close prices to log returns
    returns = adj_close_data / adj_close_data.shift(1)
    ln_returns = np.log(returns)
    
    # Depending on time to maturity of option, base estimation on different data
    try:
        # If tau smaller than 1 month base on past 2 months of data
        if time_to_expiry_yrs < 1/12:
            data_ln_returns = ln_returns.iloc[-42:]
            vol = data_ln_returns.std() * np.sqrt(252)
        # If tau smaller than 1 year base on past 1,5 years of data
        elif time_to_expiry_yrs < 1:
            data_ln_returns = ln_returns.iloc[-350:]
            vol = data_ln_returns.std() * np.sqrt(252)
        # If tau smaller than 3 years base on past 6 years of data
        elif time_to_expiry_yrs < 3:
            data_ln_returns = ln_returns.iloc[-1500]
            vol = data_ln_returns.std() * np.sqrt(252)
        # Else base on all data
        else:
            data_ln_returns = ln_returns
            vol = data_ln_returns.std() * np.sqrt(252)
    # If exception is raised due to out of bound error, base on all available data
    except:
        data_ln_returns = ln_returns
        vol = data_ln_returns.std() * np.sqrt(252)
    # Return estimated volatility
    return float(vol.iloc[0])

# Function for estimating mean return based on historical data
"""
Input:
stock_data = {
    Dataframe
    } # Dataframe with at least the Adjusted close prices in column named "Adj Close"
time_to_expiry_yrs = {
    float
    } # Time to expiry of option in years
"""
def estimate_mean_return_from_historical_data(stock_data, time_to_expiry_yrs):
    # Extract adjusted close data from historical data
    adj_close_data = stock_data["Adj Close"].ffill()
    
    # Transform close prices to log returns
    returns = adj_close_data / adj_close_data.shift(1)
    ln_returns = np.log(returns)
    
    # Estimate mean return
    mu = ln_returns.iloc[-1000:].mean() * 252
    
    # Return estimated volatility
    return float(mu.iloc[0])

# Function for estimating option price using Monte Carlo simulation
"""
Input:
contract_type = {
    "call",
    "put"
    } # String of the type of option contract
stock_price = {
    float
    } # Float with the current stock price
K = {
    float
     } # Strike price of the option
r = {
     float
     } # Estimated interest rate over contract period
mu = {
      float
      } # Estimated mean annualized return of underlying stock over contract period
vol = {
       float
       } # Estimated volatility of underlying stock over contract period
time_to_expriy_yrs = {
    float
    } # Time to expiry of the option in years
"""
def Monte_Carlo_price(contract_type, stock_price, K, r, mu, vol, time_to_expiry_yrs):
    # Set simulation parameters
    N = 1000000 # One million
    mu = mu
    vol = vol
    S0 = stock_price
    tau = time_to_expiry_yrs
    
    # Simulate stock prices
    Z = np.random.normal(0, 1, [N, 1])
    S_maturity = S0 * np.exp( (mu - 0.5 * vol**2) * tau + vol * np.sqrt(tau) * Z)

    # Find simulated option payoffs
    if contract_type == "call":
        C_maturity = np.maximum(0, S_maturity - K)
    elif contract_type == "put":
        C_maturity = np.maximum(0, K - S_maturity)
    
    # Return average of maturity payoff and discount back to present
    C_maturity_est = np.mean(C_maturity)
    
    # Discount estimated option price to present
    C_now_est = C_maturity_est * np.exp(-r * tau)
    
    # Return estimated option price
    return C_now_est

# Function for extracting actual market makers' option price from option chain data given strike and contract type
"""
Input:
option_chain = {
    object
    } # yfinance extracted option chain of the option
contract_type = {
    "call",
    "put"
    } # String of the contract type of the option
K = {
     float
     } # Float of the strike price of the option
"""
def actual_option_price(option_chain, contract_type, K):
    # Look in call or put option chain
    if contract_type == "call":
        option_chain = option_chain[0]
    elif contract_type == "put":
        option_chain = option_chain[1]
    
    # For given strike extract option data of bid and ask
    row = option_chain[option_chain["strike"] == K]
    bid = row["bid"].values[0]
    ask = row["ask"].values[0]
    
    # Calculate market makers' fair value
    option_value = (bid + ask) /2
    
    # Return actual option value
    return option_value

# Function for estimating option price using the binomial tree method
"""
Input:
contract_type = {
    "call",
    "put"
    } # String of the type of option contract
stock_price = {
    float
    } # Float with the current stock price
K = {
    float
     } # Strike price of the option
r = {
     float
     } # Estimated interest rate over contract period
mu = {
      float
      } # Estimated mean annualized return of underlying stock over contract period
vol = {
       float
       } # Estimated volatility of underlying stock over contract period
time_to_expriy_yrs = {
    float
    } # Time to expiry of the option in years
"""
def Binomial_tree_price(contract_type, stock_price, K, r, mu, vol, time_to_expiry_yrs):
    # Number of steps taken
    N=10000
    
    # Initialize model parameters
    dt = time_to_expiry_yrs / N # Step size
    u = np.exp(mu * dt + vol * np.sqrt(dt)) # Up factor
    d = 1/u # Down factor
    q = (np.exp(r*dt)-d)/(u-d) # Risk neutral probability
    S0=stock_price # Initial stock price
    discount = np.exp(-r*dt) # preinitialize discount amount for faster computation

    # Initialize stock price tree
    ST = S0 * d**(np.arange(N,-1,-1)) * u**(np.arange(0,N+1,1))
    
    # Compute terminal stock price
    if contract_type == "call":
        C = np.maximum(ST-K, np.zeros(N+1))
    else:
        C = np.maximum(K-ST, np.zeros(N+1))
    
    # Compute option price backwards until current time step
    for i in np.arange(N,0,-1):
        C = discount * (q*C[1:i+1] + (1-q)*C[0:i])
        
    # Return estimated option price
    return C[0]





    