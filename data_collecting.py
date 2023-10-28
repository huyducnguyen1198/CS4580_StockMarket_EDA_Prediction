import time
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import  numpy as np
def fetch_earnings_data(symbol):
    """Fetches earnings data for the given stock symbol from Yahoo Finance."""

    # Initialize WebDriver
    driver = webdriver.Chrome()

    # Navigate to the earnings calendar page
    url = f'https://finance.yahoo.com/calendar/earnings?symbol={symbol}'
    driver.get(url)

    # Add an explicit wait to ensure JavaScript content loads
    time.sleep(10)

    # Parse the page source with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Close the browser
    driver.quit()

    # Extract earnings data
    table = soup.find('table', class_='W(100%)')
    if table:
        rows = table.tbody.find_all('tr')
        earnings_data = []

        for row in rows:
            columns = row.find_all('td')
            date = columns[2].text
            earnings_estimate = columns[3].text
            reported_earnings = columns[4].text
            surprise = columns[5].text
            earnings_data.append((date, earnings_estimate, reported_earnings, surprise))

        return earnings_data
    else:
        print("Unable to locate the table.")
        return None

if __name__ == "__main__":
    symbol = "NVDA"
    earnings_data = fetch_earnings_data(symbol)

    # Convert earnings data to DataFrame
    earnings_df = pd.DataFrame(earnings_data, columns=['Date', 'Earnings Estimate', 'Reported Earnings', 'Surprise'])

    # Load stock data
    stock_df = pd.read_csv("NVDA.csv")
    amd_df = pd.read_csv("AMD.csv")
    nasdaq_df = pd.read_csv("NASDAQ.csv")
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    amd_df['Date'] = pd.to_datetime(amd_df['Date'])
    nasdaq_df['Date'] = pd.to_datetime(nasdaq_df['Date'])
    earnings_df['Date'] = earnings_df['Date'].str.split(',').str[0] + ',' + earnings_df['Date'].str.split(',').str[1]

    # Convert the Date column to datetime format
    earnings_df['Date'] = pd.to_datetime(earnings_df['Date'])
    earnings_df = earnings_df.iloc[::-1].reset_index(drop=True)
    # Calculate days to the next earnings for each date in stock data
    def days_to_next_earnings(stock_date):
        next_earnings_date = earnings_df[earnings_df['Date'] > stock_date]['Date'].min()
        if pd.isna(next_earnings_date):
            return None
        return (next_earnings_date - stock_date).days
    def find_first_est_date(stock_date):
        prev_earnings_date = earnings_df[earnings_df['Date'] < stock_date]['Date'].max()
        return prev_earnings_date

    stock_df['Days to Next Earnings'] = stock_df['Date'].apply(days_to_next_earnings)
    #remove all earning data before the current stock data date
    date = find_first_est_date(stock_df["Date"].iloc[0])
    earnings_df = earnings_df[earnings_df['Date'] >= date]
    earnings_df = earnings_df[earnings_df['Date'] <= stock_df['Date'].iloc[-1]]


    def get_previous_earnings(stock_date):
        # Filter the earnings_df for dates before the current stock_date
        previous_earnings = earnings_df[earnings_df["Date"] < stock_date]

        # If there are no previous earnings, return NaN for both
        if previous_earnings.empty:
            return np.nan, np.nan

        # Return the "Reported Earnings" and "Surprise" of the latest earnings report before the stock_date
        return previous_earnings.iloc[-1]["Reported Earnings"], previous_earnings.iloc[-1]["Surprise"]


    # Apply the function to each row in stock_df
    stock_df["Previous Reported EPS"], stock_df["Previous Surprise Earning"] = zip(
        *stock_df["Date"].apply(get_previous_earnings))
    #zip help to unpack tupe
    #calculate change percentage of open current day and close previous day
    stock_df['Close_Open Prev_Next Day %'] = ((stock_df['Open'] - stock_df['Close'].shift(1)) / stock_df['Close'].shift(1)) * 100
    stock_df['Close_Open Prev_Next Day %'] = stock_df['Close_Open Prev_Next Day %'].shift(periods=-1)

    stock_df['5_day_avg'] = stock_df['Close'].rolling(window=5).mean()
    stock_df['10_day_avg'] = stock_df['Close'].rolling(window=10).mean()
    stock_df['20_day_avg'] = stock_df['Close'].rolling(window=20).mean()
    stock_df['daily_return'] = stock_df['Close'].pct_change() * 100
    rolling_mean = stock_df['Close'].rolling(window=20).mean()
    rolling_std = stock_df['Close'].rolling(window=20).std()
    stock_df['Bollinger_Upper'] = rolling_mean + (rolling_std * 2)
    stock_df['Bollinger_Lower'] = rolling_mean - (rolling_std * 2)
    delta = stock_df['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    stock_df['RSI'] = 100 - (100 / (1 + rs))

    exp1 = stock_df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = stock_df['Close'].ewm(span=26, adjust=False).mean()
    stock_df['MACD'] = exp1 - exp2
    stock_df['Signal_Line'] = stock_df['MACD'].ewm(span=9, adjust=False).mean()

    stock_df['Historical_Volatility'] = stock_df['daily_return'].rolling(window=21).std()
    stock_df['OpenVolume'] = stock_df['Open'] * stock_df['Volume']

    amd_df['AMD_Close_Open Prev_Next Day %'] = ((amd_df['AMD_Open'] - amd_df['AMD_Close'].shift(1)) / amd_df['AMD_Close'].shift(1)) * 100
    amd_df['AMD_Close_Open Prev_Next Day %'] = amd_df['AMD_Close_Open Prev_Next Day %'].shift(periods=-1)
    amd_df['AMD_Daily_Return'] = amd_df['AMD_Close'].pct_change() * 100

    nasdaq_df['NASDAQ_Close_Open Prev_Next Day %'] = ((nasdaq_df['NASDAQ_Open'] - nasdaq_df['NASDAQ_Close'].shift(1)) / nasdaq_df[
        'NASDAQ_Close'].shift(1)) * 100
    nasdaq_df['NASDAQ_Close_Open Prev_Next Day %'] = nasdaq_df['NASDAQ_Close_Open Prev_Next Day %'].shift(periods=-1)
    nasdaq_df['NASDAQ_Daily_Return'] =  nasdaq_df['NASDAQ_Close'].pct_change() * 100
    stock_df = stock_df.merge(amd_df, on='Date', how='inner')
    stock_df = stock_df.merge(nasdaq_df, on='Date', how='inner')

    stock_df['Diff'] = stock_df['Close'] - stock_df['Open']
    stock_df['Stock rise for the next day?'] = stock_df['Diff'].shift(-1)
    stock_df['Stock rise for the next day?'] = stock_df['Stock rise for the next day?'] > 0

    # Drop the temporary 'Diff' column
    stock_df = stock_df.drop(columns='Diff')



    print(earnings_df.head(5))
    print(stock_df.head(40))
    stock_df.to_csv('NVDA_With_Earning.csv', index=False)


