import pandas as pd
import numpy as np
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import webbrowser, os
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.offline as po
from sklearn import preprocessing


def correlation_analyze(df):
    amd_nvdia_price_corr = df['Close'].corr(df['AMD_Close'])
    print("Correlation close price between AMD and NVIDIA: ")
    print(amd_nvdia_price_corr)
    volume_earningday_corr = df["Days to Next Earnings"].corr(df['Volume'])
    print("Correlation between days to next earning and volumn")
    print(volume_earningday_corr)
    nvidia_amd_volume_corr = df['Volume'].corr(df['AMD_Volume'])
    print("Correlation between nvidia and amd volume")
    print(nvidia_amd_volume_corr)


def visualization(df):
    fig = px.scatter(df, x='Days to Next Earnings', y='Volume', title='Days to Next Earning vs Volumn')
    filename = "days_volume.html"
    po.plot(fig, filename=filename)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["Volume"], name="Volume"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["Days to Next Earnings"], name="Days to Next Earnings"),
        secondary_y=True,
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Date")

    # Set y-axes titles
    fig.update_yaxes(title_text="Volume", secondary_y=False)
    fig.update_yaxes(title_text="Days to Next Earnings", secondary_y=True)

    # Show the figure
    po.plot(fig, "days_volume_with_date")




df = pd.read_csv("NVDA_With_Earning.csv")
correlation_analyze(df)
visualization(df)





