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

    stock_df['Date'] = pd.to_datetime(stock_df['Date'])

    earnings_df['Date'] = earnings_df['Date'].str.split(',').str[0] + ',' + earnings_df['Date'].str.split(',').str[1]

    # Convert the Date column to datetime format
    earnings_df['Date'] = pd.to_datetime(earnings_df['Date'])
    earnings_df = earnings_df.iloc[::-1].reset_index(drop=True)
    def find_first_est_date(stock_date):
        prev_earnings_date = earnings_df[earnings_df['Date'] < stock_date]['Date'].max()
        return prev_earnings_date

    # #remove all earning data before the current stock data date
    date = find_first_est_date(stock_df["Date"].iloc[0])
    earnings_df = earnings_df[earnings_df['Date'] > date]
    earnings_df = earnings_df[earnings_df['Date'] <= stock_df['Date'].iloc[-1]]

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

    stock_df = stock_df.merge(earnings_df, on='Date', how='outer')

    # print(earnings_df.head(5))
    # print(stock_df.head(40))
    stock_df.to_csv('NVDA_With_Extra_Features.csv', index=False)


