"""
Additions to model:
- Correct expiry dates incorporate different expiry timezones and markets depending on time zone
- expand for multiple strikes
- expand for puts
- expand for multiple expiries
- expand for multiple tickers
- modelling r: implied market rate from futures or other
- black scholes only works for european options while most options are american, so other methods:
    .Monte Carlo simulations
    .Machine Learning
    .Finite Differences Method
    .Fourier Transform-based model
    .Binomial Option Pricing model
- estimate volatility by: implied volatility OR historical data. Note implied vol on yfinance option chain but sometimes inaccurate.
- incorporate dividends in black scholes model
- expand Blach scholes for dividend stocks
- expand Monte Carlo for american options and check if method is implemented correctly (done quickly in the evening)
- expand monte carlo simulations for dividend paying stocks
- return standard errors for all option price estimations
- expand binomial tree method for american options
- expand binomial tree method for dividend paying stocks
"""














