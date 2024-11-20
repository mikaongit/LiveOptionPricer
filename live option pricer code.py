"""
Live option pricer by Mika Cardinaal
"""

# Import packages
import yfinance as yf

# Import functions
from functions import Black_Scholes_price
from functions import time_to_expiry
from functions import get_current_yield_data
from functions import estimate_interest_rate
from functions import estimate_volatility_from_historical_data
from functions import estimate_mean_return_from_historical_data
from functions import Monte_Carlo_price
from functions import actual_option_price
from functions import Binomial_tree_price

# Global initial parameters
ticker = "AAPL"


# Extracting real prices to compare theoratical prices to
stock = yf.Ticker(ticker)
stock_data = yf.download(ticker)

expiration_dates = stock.options

#option_chain = stock.option_chain(expiration_dates)


###################################### FOR one expiration date and one strike price
exp_date = expiration_dates[4]
option_chain = stock.option_chain(exp_date)

calls = option_chain[0]
puts = option_chain[1]
extra_info = option_chain[2]
stock_info = stock.info

contract_type = "call"

strike = calls["strike"].iloc[45]

time_to_expiry_yrs = time_to_expiry(exp_date)
time_to_expiry_mo = time_to_expiry(exp_date, timeframe="months")

# Option
stock_price = stock_info["currentPrice"]
yield_data = get_current_yield_data()
r = estimate_interest_rate(yield_data, time_to_expiry_yrs)
vol = estimate_volatility_from_historical_data(stock_data, time_to_expiry_yrs)
mu = estimate_mean_return_from_historical_data(stock_data, time_to_expiry_yrs)

K = 200

#%%
bs_price = Black_Scholes_price(contract_type, stock_price, K, r, vol, time_to_expiry_yrs)
mc_price = Monte_Carlo_price(contract_type, stock_price, K, r, mu, vol, time_to_expiry_yrs)
bt_price = Binomial_tree_price(contract_type, stock_price, K, r, mu, vol, time_to_expiry_yrs)

actual_price = actual_option_price(option_chain, contract_type, K)












