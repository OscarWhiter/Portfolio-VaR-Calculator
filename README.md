# Portfolio VaR Calculator

A Python tool to calculate **Value-at-Risk (VaR)** for a portfolio of stocks, using **Historical, Monte Carlo, and Parametric methods**.  
This project demonstrates a practical application of risk management techniques in finance, showcasing the calculation of potential losses in a portfolio over a one-day horizon.

---

## Project Description

This Python project calculates **Historical, Monte Carlo, and Parametric Value-at-Risk (VaR)** for a portfolio of stocks.  
Interactive and robust, it handles invalid tickers, user-defined weights, and customizable confidence levels, with clear visualizations of portfolio risk.

Key features of the code:

- **Interactive input** for tickers, portfolio weights, and confidence level.
- **Automatic handling of invalid tickers**, skipping them and notifying the user.
- Computes **Historical, Monte Carlo, and Parametric VaR**.
- Outputs VaR results in **percentages** for clarity.
- Visualizes portfolio returns and compares VaR methods using **histograms and VaR lines**.
- Fully robust **error handling** for invalid inputs, empty returns, or zero standard deviation in simulations.

---

## About Value-at-Risk (VaR)

**Value-at-Risk (VaR)** is a widely used risk metric in finance that estimates the **maximum expected loss** of a portfolio over a given time period, at a specified confidence level.  

- **Historical VaR** uses actual past returns to compute the percentile corresponding to the desired confidence level.  
- **Monte Carlo VaR** simulates portfolio returns based on the mean and standard deviation of historical returns, allowing for flexible modeling of potential outcomes.  
- **Parametric VaR** assumes returns follow a **normal distribution** and calculates VaR analytically using the mean and standard deviation.  

This project not only calculates all three VaR types but also **visualizes the results**, helping users compare different risk estimation approaches and understand portfolio risk.

---

## Requirements

This project requires **Python 3.8+** and the following packages:

- numpy  
- pandas  
- matplotlib  
- scipy  
- yfinance  

You can install all dependencies at once using:

```bash
pip install -r requirements.txt
