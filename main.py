
import pandas as pd
import numpy as np
import warnings

from plotly.subplots import make_subplots

import webbrowser, os
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.offline as po
from scipy import stats
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE, SelectKBest

import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif

######## keras ########
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout



pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def correlation_analyze(df):
    # amd_nvdia_price_corr = df['Close'].corr(df['AMD_Close'])
    # print("Correlation close price between AMD and NVIDIA: ")
    # print(amd_nvdia_price_corr)
    volume_earningday_corr = df["Days to Next Earnings"].corr(df['Volume'])
    print("Correlation between days to next earning and volume")
    print(volume_earningday_corr)
    # nvidia_amd_volume_corr = df['Volume'].corr(df['AMD_Volume'])
    # print("Correlation between nvidia and amd volume")
    #print(nvidia_amd_volume_corr)
    diff_stock_price_rise_corr = df["is diff close and open more than 2%"].corr(df["Stock rise for the next day?"])
    print("Correlation between is diff close and open more than 2% and Stock rise for the next day?")
    print(diff_stock_price_rise_corr)


def visualization(df):
    fig = px.scatter(df, x='Days to Next Earnings', y='Volume', title='Days to Next Earning vs Volumn')
    filename = "days_volume.html"
    po.plot(fig, filename=filename, auto_open=False)

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
    po.plot(fig, "days_volume_with_date", auto_open=False)

    crosstab_result = pd.crosstab(df["is diff close and open more than 2%"], df["Stock rise for the next day?"])

    # Modify the index for more descriptive labels
    crosstab_result.index = ['Diff < 2%' if idx == False else 'Diff > 2%' for idx in crosstab_result.index]

    # Define the bars
    bar1 = go.Bar(x=crosstab_result.index, y=crosstab_result[True], name='Stock Rise (Yes)', marker_color='blue')
    bar2 = go.Bar(x=crosstab_result.index, y=crosstab_result[False], name='Stock Rise (No)', marker_color='red')

    # Combine bars and plot
    fig = go.Figure(data=[bar1, bar2])
    fig.update_layout(barmode='stack',
                      title="Relationship between 'Diff Close and Open' and 'Stock Rise for the Next Day?'")



    po.plot(fig, filename='diff_close_open_stock_rise.html', auto_open=False)


def chi2_test(df):
    contingency = pd.crosstab(df["is diff close and open more than 2%"], df["Stock rise for the next day?"])
    chi2, p_value, df = stats.chi2_contingency(contingency)[0:3]
    print("\n Chi square value of diff close and open and stock rise: ", chi2)
    print("P_value is ", p_value )
    print("Degree of freedom: ", df)

def train_evaluate(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, cm

def LogisticRegressionTest(new_features, df, y):
    classifer = LogisticRegression(max_iter=1000)
    X = df.loc[:, new_features]
    X = scalar.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    accuracy, cm = train_evaluate(classifer, X_train, X_test, y_train, y_test)
    print("Accuracy: ", accuracy)
    print("Cm: ")
    print(cm)

def RandomForest(new_features, df, y):
    X = df.loc[:, new_features]
    X = scalar.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    accuracy, cm = train_evaluate(rf_classifier, X_train, X_test, y_train, y_test)
    print("RandomForest Classifier Accuracy: ", accuracy)
    print("Cm: ")
    print(cm)

def GradientBoosting(new_features, df, y):
    X = df.loc[:, new_features]
    X = scalar.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
    accuracy, cm = train_evaluate(gb_classifier, X_train, X_test, y_train, y_test)
    print("GradientBoosting Classifier Accuracy: ", accuracy)
    print("Cm: ")
    print(cm)

def SVCTest(new_features, df, y):
    X = df.loc[:, new_features]
    X = scalar.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    svm_classifier = SVC(kernel='linear', C=1, random_state=42)
    accuracy, cm = train_evaluate(svm_classifier, X_train, X_test, y_train, y_test)
    print("Support Vector Machine Classifier Accuracy: ", accuracy)
    print("Cm: ")
    print(cm)



def qui():
    df = pd.read_csv("NVDA_With_Earning.csv")

    df["is diff close and open more than 2%"] = df["Close_Open Prev_Next Day %"] > 2


    correlation_analyze(df)
    visualization(df)
    chi2_test((df))

    le = LabelEncoder()
    df = df.apply(preprocessing.LabelEncoder().fit_transform)
    df.dropna(inplace=True)



    #setting X and y, features
    filtered_column = df.columns.drop(['Stock rise for the next day?', 'is diff close and open more than 2%'])
    features = np.array(filtered_column)
    X = df.loc[:, features]
    y= df['Stock rise for the next day?']
    scalar = StandardScaler()
    X = scalar.fit_transform(X)

    classifer = LogisticRegression(max_iter=1000)

    #setting num_components
    num_components = 10

    rfe = RFE(estimator=classifer, n_features_to_select=num_components)
    rfe = rfe.fit(X, y)

    print(f'\n Top {num_components} features: ')
    new_features = features[rfe.support_]
    print(new_features)
    LogisticRegressionTest(new_features, df, y)

    RandomForest(new_features, df, y)

    GradientBoosting(new_features, df, y)

    SVCTest(new_features, df, y)

    X = df.loc[:, features]
    # 1. Feature Importances from RandomForest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:num_components]
    print("\nTop features using RandomForest importances:")
    new_features = features[indices]
    print(new_features)

    LogisticRegressionTest(new_features, df, y)

    RandomForest(new_features, df, y)

    GradientBoosting(new_features, df, y)

    SVCTest(new_features, df, y)

    # 2. SelectKBest with Mutual Information
    k_best = SelectKBest(mutual_info_classif, k=num_components)
    X_new = k_best.fit_transform(X, y)

    print("\nTop features using Mutual Information:")
    cols = k_best.get_support(indices=True)
    new_features = df.iloc[:, cols].columns.tolist()
    print(new_features)

    LogisticRegressionTest(new_features, df, y)

    RandomForest(new_features, df, y)

    GradientBoosting(new_features, df, y)

    SVCTest(new_features, df, y)

    # 3. Correlation with the Target
    correlations = {}
    for f in features:
        data_temp = df[[f, 'Stock rise for the next day?']]
        x1 = data_temp[f].values
        x2 = data_temp['Stock rise for the next day?'].values
        key = f
        correlations[key] = spearmanr(x1, x2)[0]
    data_correlations = pd.DataFrame(correlations, index=['Value']).T
    new_features = data_correlations['Value'].abs().sort_values(ascending=False)[:num_components].index
    print("\nTop features based on Correlation with target:")
    print(new_features)

    LogisticRegressionTest(new_features, df, y)

    RandomForest(new_features, df, y)

    GradientBoosting(new_features, df, y)

    SVCTest(new_features, df, y)
#############################################   main   #############################################
def renameColumn(df):
    df.rename(columns={'Days to Next Earnings': 'Days_to_Next_Earnings'}, inplace=True)
    df.rename(columns={'Stock rise for the next day?': 'rise_Next_day?'}, inplace=True)
    df.rename(columns={'is diff close and open more than 2%': 'diff_close_open_2%'}, inplace=True)
    df.rename(columns={'Close_Open Prev_Next Day %': 'Close_Open_Prev_Next_Day_%'}, inplace=True)
    df.rename(columns={'Previous Reported EPS': 'Previous_Reported_EPS'}, inplace=True)
    df.rename(columns={'Previous Surprise Earning': 'Previous_Surprise_Earning'}, inplace=True)

def plot(df):

    #histogram, grouping by each month
    df_days = df[['Date', 'Close_Open_Prev_Next_Day_%']]
    #histogram, grouping by each month
    gby = df_days.groupby('Date').mean().reset_index()
    fig = px.histogram(gby, x='Date', y='Close_Open_Prev_Next_Day_%', title='Close_Open_Prev_Next_Day_%',nbins=200)
    #fig = px.histogram(df_days, x='Date', y='Close_Open_Prev_Next_Day_%', title='Close_Open_Prev_Next_Day_%',nbins=200)
    #fig.show()


    #stock NVDIA and AMD, there is a similar trend
    df_compare = df[['Date', 'Open', 'Close', 'High', 'Low', 'AMD_Open', 'AMD_Close', 'AMD_High', 'AMD_Low']]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df_compare['Date'], y=df_compare['Open'], name='NVDA Open'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_compare['Date'], y=df_compare['Close'], name='NVDA Close'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_compare['Date'], y=df_compare['High'], name='NVDA High'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_compare['Date'], y=df_compare['Low'], name='NVDA Low'), secondary_y=False)

    fig.add_trace(go.Scatter(x=df_compare['Date'], y=df_compare['AMD_Open'], name='AMD Open'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_compare['Date'], y=df_compare['AMD_Close'], name='AMD Close'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_compare['Date'], y=df_compare['AMD_High'], name='AMD High'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_compare['Date'], y=df_compare['AMD_Low'], name='AMD Low'), secondary_y=False)

    #fig.show()




    #Corelation between stock by NVDIA and AMD  => highly correlated
    df_corr = df[['Open', 'Close', 'High', 'Low', 'AMD_Open', 'AMD_Close', 'AMD_High', 'AMD_Low']]
    corr = df_corr.corr()
    fig = px.imshow(corr)
    #fig.show()

    #RSI and MACD => RSI > 70 => overbought => sell
    df_rsi = df[['Date', 'RSI', 'MACD', 'Signal_Line']]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add RSI trace
    fig.add_trace(go.Scatter(x=df_rsi['Date'], y=df_rsi['RSI'], name='RSI'), secondary_y=False)

    # Add straight lines for 70 and 30 RSI values
    fig.add_trace(
        go.Scatter(x=df_rsi['Date'], y=[70] * len(df_rsi), name='RSI 70', line=dict(dash='dash', color='red')),
        secondary_y=False)
    fig.add_trace(
        go.Scatter(x=df_rsi['Date'], y=[30] * len(df_rsi), name='RSI 30', line=dict(dash='dash', color='green')),
        secondary_y=False)




    #stock with MACD and Signal Line => MAACD > Signal Line => stock price increase => buy
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Stock Price'), row=1, col=1)

    # Plot MACD and Signal Line
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Signal_Line'], name='Signal Line', line=dict(color='red')), row=2,
                  col=1)

    #fig.show()


    #stockwith volatilitity  => volatility is high when stock price is low
    # Create a subplot with shared x-axis
    fig = go.Figure()

    # Plot Stock Prices
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Stock Price', line=dict(color='blue')))

    # Add a secondary y-axis for the Historical Volatility
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Historical_Volatility'], name='Historical Volatility', line=dict(color='red'),
                   yaxis='y2'))

    # Layout modifications for the second y-axis
    fig.update_layout(
        yaxis2=dict(title='Historical Volatility', overlaying='y', side='right')
    )

    fig.show()



def separateDate(df):
    #get day, month, year from date
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year


def getTimeSeries(df, window_size):
    '''
    get time series data on Day, Month, Open, Close, High, Low, Volume
    :param df: dataframe include all data(Day, Month, Open, Close, High, Low, Volume)
    :param window_size: how many days to in the past to predict the next day
    :return: x_train, x_test, y_train, y_test
    '''

    #might add , 'RSI', 'MACD', 'Signal_Line', 'Historical_Volatility' later
    df_time_series = df[['Day','Month', 'Open', 'Close', 'High', 'Low', 'Volume']]
    scaledData = StandardScaler().fit_transform(df_time_series)

    x = []
    y = []
    for i in range(0, df_time_series.shape[0] - window_size):
        x.append(scaledData[i:i + window_size])
        y.append(scaledData[i + window_size, 2:2 + 4])#get open, close, high, low, start at index 2(Open) and end at index 5(Low) => 4 elements
    x = np.array(x)
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test

def stockPredict(df):
    '''
    predict stock using LSTM model.
    :return:
    '''

    #get data from getTimeSeries()
    x_train, x_test, y_train, y_test = getTimeSeries(df, 12) #MACD is about 12 days
    print(x_train.shape)
    #create LSTM model using keras
    model = Sequential()

    #add LSTM layer
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 7)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, activation='relu'))

    #add output layer
    model.add(Dense(units=4))

    #compile model
    model.compile(optimizer='adam', loss='mean_squared_error')

    #fit model
    history = model.fit(x_train, y_train, epochs=10,validation_data=(x_test, y_test), batch_size=32)

    #predict
    y_pred = model.predict(x_test)

    #plot
    import numpy as np

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    width = 0.4  # width of the bars
    indices = np.arange(len(x_test))  # the label locations

    # Bar Chart for 'Open'
    axes[0, 0].bar(indices - width / 2, y_test[:, 0], width, color='red', label='Real Open')
    axes[0, 0].bar(indices + width / 2, y_pred[:, 0], width, color='blue', label='Predicted Open')
    axes[0, 0].set_title('Open Prediction')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Open')
    axes[0, 0].set_xticks(indices)
    axes[0, 0].legend()

    # Bar Chart for 'Close'
    axes[0, 1].bar(indices - width / 2, y_test[:, 1], width, color='red', label='Real Close')
    axes[0, 1].bar(indices + width / 2, y_pred[:, 1], width, color='blue', label='Predicted Close')
    axes[0, 1].set_title('Close Prediction')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Close')
    axes[0, 1].set_xticks(indices)
    axes[0, 1].legend()

    # Bar Chart for 'High'
    axes[1, 0].bar(indices - width / 2, y_test[:, 2], width, color='red', label='Real High')
    axes[1, 0].bar(indices + width / 2, y_pred[:, 2], width, color='blue', label='Predicted High')
    axes[1, 0].set_title('High Prediction')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('High')
    axes[1, 0].set_xticks(indices)
    axes[1, 0].legend()

    # Bar Chart for 'Low'
    axes[1, 1].bar(indices - width / 2, y_test[:, 3], width, color='red', label='Real Low')
    axes[1, 1].bar(indices + width / 2, y_pred[:, 3], width, color='blue', label='Predicted Low')
    axes[1, 1].set_title('Low Prediction')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Low')
    axes[1, 1].set_xticks(indices)
    axes[1, 1].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

    # Plotting the Mean Squared Error
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training MSE')
    plt.plot(history.history['val_loss'], label='Validation MSE')
    plt.title('MSE over epochs')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()

df = pd.read_csv("NVDA_With_Earning.csv")
separateDate(df)

df = df[df['Year'] >= 2022]

renameColumn(df)
#plot(df)
stockPredict(df)
