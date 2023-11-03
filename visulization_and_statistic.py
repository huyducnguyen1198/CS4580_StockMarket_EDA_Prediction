
import pandas as pd
import numpy as np
import warnings

from plotly.express.trendline_functions import ols
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison




import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif



def statistic_analyze(df):
    macd_signal_line_dailyreturn_corr = df['diff_macd_sinal_line'].corr(df['daily_return'])
    print("Correlation between difference of signal line and macd and time daily return ")
    print(macd_signal_line_dailyreturn_corr)
    macd_dailyreturn_corr = df['MACD'].corr(df['daily_return'])
    df_2022 = df[df['Date'].dt.year == 2022]

    # Extract only the month and 'daily_return' column
    df_2022 = df_2022[['Date', 'daily_return']].copy()
    df_2022['Date'] = df_2022['Date'].dt.month_name()  # Get month name. Use dt.month for month number.
    print(df_2022.head())
    print(df_2022.tail())

    model = ols('daily_return~ C(Date)', data=df_2022).fit()


    # Actually run the anova:
    aov_table = sm.stats.anova_lm(model)
    print('ANOVA results of Date and daily_return:\n', aov_table)
    mc = MultiComparison(df_2022['daily_return'], df_2022['Date'])
    result = mc.tukeyhsd()
    print("\nTukey HSD results:")
    print(result)
    print("-" * 100)



def visualization(df):

    #--------------------------------------------------
    #Close Price vs. Bollinger Bands
    fig1 = go.Figure()

    # Use 'x' argument to specify the date column for the scatter plots
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price'))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Bollinger_Upper'], mode='lines', name='Bollinger Upper',
                              line=dict(width=0.5, color='red')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Bollinger_Lower'], mode='lines', name='Bollinger Lower',
                              line=dict(width=0.5, color='red')))

    fig1.update_layout(title="Close Price and Bollinger Bands", xaxis_title="Date", yaxis_title="Stock Price")
    fig1.write_html("nvidia_close_bollinger.html")
    #RSI Indicator
    fig2 = go.Figure()

    # Use 'x' argument to specify the date column for the scatter plot
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                         subplot_titles=("RSI Indicator", "Close Price Movement"))

    # Add RSI trace to the first subplot (on the top)
    fig2.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI'), row=1, col=1)

    # Add threshold lines to the RSI plot
    fig2.add_shape(type='line', line=dict(dash='dash'), y0=70, y1=70, x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1],
                   xref='x1', yref='y1', line_color='gray')
    fig2.add_shape(type='line', line=dict(dash='dash'), y0=30, y1=30, x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1],
                   xref='x1', yref='y1', line_color='gray')

    # Add Close price trace to the second subplot (on the bottom)
    fig2.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price'), row=2, col=1)

    # Update the layout
    fig2.update_layout(title="RSI Indicator and Close Price Analysis")
    fig2.update_yaxes(title_text="RSI Value", row=1, col=1)
    fig2.update_yaxes(title_text="Price", row=2, col=1)
    fig2.update_xaxes(title_text="Date", row=2, col=1)

    # Write to an HTML file
    fig2.write_html("nvidia_rsi_close_stacked.html")
    #Macd vs signal line

    fig3 = make_subplots(rows=1, cols=2, shared_xaxes=True, horizontal_spacing=0.1,
                         subplot_titles=("MACD and Signal Line", "Close Price Movement"))

    # Add MACD trace to the first subplot (on the left)
    fig3.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD'), row=1, col=1)

    # Add Signal Line trace to the first subplot (on the left)
    fig3.add_trace(go.Scatter(x=df['Date'], y=df['Signal_Line'], mode='lines', name='Signal Line'), row=1, col=1)

    # Add Close price trace to the second subplot (on the right)
    fig3.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price'), row=1, col=2)

    # Update the layout
    fig3.update_layout(title="MACD, Signal Line, and Close Price Analysis")
    fig3.update_yaxes(title_text="Value", row=1, col=1)
    fig3.update_yaxes(title_text="Price", row=1, col=2)
    fig3.update_xaxes(title_text="Date", row=1, col=2)

    # Write to an HTML file
    fig3.write_html("nvidia_macd_signal_close.html")
#----------


    # Create the main line plot for close prices
    fig4 = go.Figure(data=[go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price')])

    # Highlight earnings events using markers
    earnings_dates = df[df['Earnings Estimate'].notna()]['Date']
    earnings_values = df[df['Earnings Estimate'].notna()]['Close']
    surprises = df[df['Earnings Estimate'].notna()]['Surprise']

    # Add Earnings Event Scatter plot with date text on top
    fig4.add_trace(go.Scatter(x=earnings_dates, y=earnings_values, mode='markers+text', name='Earnings Event',
                             marker=dict(size=10, color='red', symbol='star'),
                             text=earnings_dates,
                             textposition="top center"))

    # Surprise Bars
    fig4.add_trace(go.Bar(x=earnings_dates, y=surprises, name='Surprise', marker_color='blue'))

    # Customize the layout
    fig4.update_layout(title='Close Prices with Earnings Events and Surprises', xaxis_title='Date',
                      yaxis_title='Close Price', xaxis_rangeslider_visible=False, barmode='overlay')

    fig4.write_html("nvidia_close_earning_event.html")


pd.set_option('display.max_columns', None)
df = pd.read_csv("NVDA_With_Extra_Features.csv")
df['Date'] = pd.to_datetime(df['Date'])
description = df.describe()
description.to_csv("description.csv")
df['diff_macd_sinal_line'] = df['MACD'] - df["Signal_Line"]
columns_with_dash = df.columns[df.isin(['-']).any()].tolist()


statistic_analyze(df)
df_40 = df.tail(int(len(df) * 0.4))
df_40.to_csv("40%_data.csv")

visualization(df)

