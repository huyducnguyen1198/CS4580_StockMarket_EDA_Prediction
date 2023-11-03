
import pandas as pd
import numpy as np
import warnings

from keras.src.callbacks import Callback
from keras.src.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten
from keras.src.metrics import mse
from keras.src.regularizers import l1, l1_l2, l2
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
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
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

    df_20 = df.tail(int(len(df) * 0.4))
    # Assuming df_20 already contains the last 20% of your data and 'RSI' column exists

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Normalization function
    def normalize_series(series, min_val, max_val):
        return (series - series.min()) * (max_val - min_val) / (series.max() - series.min()) + min_val

    # Assuming df_20 already contains the last 20% of your data
    max_close = df_20['Close'].max()
    min_close = df_20['Close'].min()

    scaled_MACD = normalize_series(df_20['MACD'], min_close, max_close)
    scaled_Signal = normalize_series(df_20['Signal_Line'], min_close, max_close)

    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Stock Close Prices with Scaled MACD & Signal Line", "RSI"))

    # Plot Stock Close Prices
    fig.add_trace(go.Scatter(x=df_20['Date'], y=df_20['Close'], name='Close', line=dict(color='black')), row=1, col=1)

    # Plot Scaled MACD & Signal Line on the same subplot
    fig.add_trace(go.Scatter(x=df_20['Date'], y=scaled_MACD, name='Scaled MACD', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_20['Date'], y=scaled_Signal, name='Scaled Signal Line', line=dict(color='red')),
                  row=1, col=1)

    # Plot RSI
    fig.add_trace(go.Scatter(x=df_20['Date'], y=df_20['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_shape(go.layout.Shape(type="line", x0=df_20['Date'].iloc[0], x1=df_20['Date'].iloc[-1], y0=70, y1=70,
                                  line=dict(color="Red", width=0.5, dash="dash")), row=2, col=1)
    fig.add_shape(go.layout.Shape(type="line", x0=df_20['Date'].iloc[0], x1=df_20['Date'].iloc[-1], y0=30, y1=30,
                                  line=dict(color="Green", width=0.5, dash="dash")), row=2, col=1)

    # Update layout
    fig.update_layout(
        legend=dict(
            x=0,  # Adjust this value to move legend left/right
            y=1,  # Adjust this value to move legend up/down
            bgcolor='rgba(255, 255, 255, 0.5)'        ),
        height=600,
        width=800,
    ) # Adjust the width
    fig.show()

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

    #fig.show()


    #stock with volume => volume is high when stock price is lowfig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    fig.add_trace(go.Scatter(x=df['Date'], y=df['High'], name='Stock Price'), row=1, col=1)

    # Plot MACD and Signal Line
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Previous_Surprise_Earning'], name='MACD', line=dict(color='blue')), row=2, col=1)

    #fig.show()
def separateDate(df):
    #get day, month, year from date
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year


def getTimeSeries(df, window_size, feature_name='High'):
    '''
    get time series data on Day, Month, Open, Close, High, Low, Volume
    :param df: dataframe include all data(Day, Month, Open, Close, High, Low, Volume)
    :param window_size: how many days to in the past to predict the next day
    :return: x_train, x_test, y_train, y_test
    '''

    #might add , 'RSI', 'MACD', 'Signal_Line', 'Historical_Volatility' later
    df_time_series = df[['Day','Month', 'Open', 'Close', 'High', 'Low', 'Volume']]
    if feature_name not in ['Day','Month', 'Open', 'Close', 'High', 'Low', 'Volume']:
        raise Exception('feature_name must be in [Day, Month, Open, Close, High, Low, Volume]')

    scaledData = MinMaxScaler().fit_transform(df_time_series)

    #scaledDf = pd.DataFrame(scaledData, columns=['Day','Month', 'Open', 'Close', 'High', 'Low', 'Volume'])
    col = 4
    if feature_name == 'High':
        col = 4
    elif feature_name == 'Low':
        col = 5
    elif feature_name == 'Open':
        col = 2
    elif feature_name == 'Close':
        col = 3

    x = []
    y = []
    for i in range(0, df_time_series.shape[0] - window_size):
        x.append(scaledData[i:i + window_size])
        y.append(scaledData[i + window_size, col])#get open, close, high, low, start at index 2(Open) and end at index 5(Low) => 4 elements
    x = np.array(x)
    y = np.array(y)
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    #reduce dimension
    '''samples, time_steps, features = x.shape
    x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
    x = PCA(n_components=5).fit_transform(x)
    x = np.reshape(x, (samples, time_steps, -1))'''
    #x = PCA(n_components=5).fit_transform(x)
    #split data into train and test
    split = int(0.8 * len(x))
    x_train = x[:split]
    x_test = x[split:]
    y_train = y[:split]
    y_test = y[split:]
    return x_train, x_test, y_train, y_test


def plotPrediction(y_test, y_pred, feature_name='High'):
    #plot
    import numpy as np

    # Assuming 'feature_name' is a string, e.g. 'High' or 'Low'
    plt.figure(figsize=(10, 5))

    # Line plot for actual and predicted values
    plt.plot(range(len(y_test)), y_test[:], color='red', label=f'Real {feature_name}')
    plt.plot(range(len(y_pred)), y_pred[:], color='blue', label=f'Predicted {feature_name}')

    # Add title and labels
    plt.title(f'{feature_name} Prediction')
    plt.xlabel('Date Index')  # Assuming you don't have the actual date and only the index
    plt.ylabel(feature_name)
    plt.legend()

    # Display the plot
    plt.show()


def plotLoss(history):
    # Plotting the Mean Squared Error
    plt.figure(figsize=(5, 3))
    plt.plot(history.history['loss'], label='Training MSE')
    plt.plot(history.history['val_loss'], label='Validation MSE')
    plt.title('MSE over epochs')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()


################################################
#               lstm model                     #
################################################
def stockPredictLSTMHybrid(df, feature_name='High'):
    '''
    predict stock using LSTM model.
    :return:
    '''

    #get data from getTimeSeries()
    x_train, x_test, y_train, y_test = getTimeSeries(df, 15, feature_name) #MACD is about 12 days
    print(x_train.shape)
    #create LSTM model using keras
    model = Sequential()

    #fileter 8, kernel_size 5, input_shape (12, 7), lstm units 20
    #add CNN
    model.add(Conv1D(filters=8, kernel_size=5, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(TimeDistributed(Flatten()))
    model.add(Dropout(0.2))

    #add Dropout
    model.add(LSTM(units=50, activation='relu', return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(LSTM(units=50, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    #add LSTM layer
    model.add(LSTM(units=50, activation='relu'))


    #add hidden layer
    #add output layer
    model.add(Dense(units=1, activation='linear'))

    #compile model
    model.compile(optimizer='adam', loss='mean_squared_error')

    #fit model
    history = model.fit(x_train, y_train ,epochs=50,validation_data=(x_test, y_test), batch_size=64)

    #predict
    y_pred = model.predict(x_test)

    print(mean_squared_error(y_test, y_pred))
    plotPrediction(y_test, y_pred, feature_name)
    plotLoss(history)

class StopWhenLossBelowThreshold(Callback):
    def __init__(self, threshold):
        super(StopWhenLossBelowThreshold, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get('val_loss')
        if val_loss is not None:
            if val_loss <= self.threshold:
                print(f"\nStopping training: validation loss {val_loss:.6f} is below threshold {self.threshold:.6f}")
                self.model.stop_training = True
def stockPredictLSTM(df, feature_name='High'):
    '''
    predict stock using LSTM model. todo, use inver_transform to get the real value
    :return:
    '''

    #get data from getTimeSeries()
    x_train, x_test, y_train, y_test = getTimeSeries(df, 12, feature_name) #MACD is about 12 days
    print(x_train.shape)
    #create LSTM model using keras
    model = Sequential()

    #add LSTM layer for 1 year use 15 units( now 1 output(high) => 6 units).
    model.add(LSTM(units=30, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.1))
    model.add(LSTM(units=30, activation='relu', return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(units=30, activation='relu'))
    #add hidden layer
    #add output layer
    model.add(Dense(units=1, activation='linear'))

    #compile model
    model.compile(optimizer='adam', loss='mean_squared_error')

    #fit model
    callback = StopWhenLossBelowThreshold(threshold=0.01)
    history = model.fit(x_train, y_train ,epochs=100,validation_data=(x_test, y_test), batch_size=32, callbacks=[callback] )

    #predict
    y_pred = model.predict(x_test)

    #plot
    #plotPrediction(y_test, y_pred, feature_name)
    #plotLoss(history)

    return y_test, y_pred


import keras_tuner as kt
from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch


class LSTMHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()

        # Tune the number of units in the first LSTM layer
        units_1 = hp.Int('units_1', min_value=20, max_value=80, step=5)
        model.add(LSTM(units=units_1, activation='relu', return_sequences=True, input_shape=self.input_shape))
        model.add(Dropout(rate=hp.Float('dropout_1', 0.1, 0.5, step=0.1)))

        # Tune the number of units in the second LSTM layer
        units_2 = hp.Int('units_2', min_value=20, max_value=80, step=5)
        model.add(LSTM(units=units_2, activation='relu', return_sequences=True))
        model.add(Dropout(rate=hp.Float('dropout_2', 0.1, 0.5, step=0.1)))

        # Tune the number of units in the third LSTM layer
        units_3 = hp.Int('units_3', min_value=20, max_value=80, step=5)
        model.add(LSTM(units=units_3, activation='relu'))

        model.add(Dense(units=1, activation='linear'))

        optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        return model

'''
x_train, x_test, y_train, y_test = getTimeSeries(df, 12, 'Close')
input_shape = (x_train.shape[1], x_train.shape[2])
hypermodel = LSTMHyperModel(input_shape)

tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=20,
    executions_per_trial=2,
    directory='random_search',
    project_name='stock_prediction_lstm')'''

#tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
from keras.callbacks import LearningRateScheduler

def LSTMTest(df, feature_name = 'Close'):
    # get data from getTimeSeries()
    x_train, x_test, y_train, y_test = getTimeSeries(df, 3, feature_name)  # MACD is about 12 days
    print(x_train.shape)
    # create LSTM model using keras
    model = Sequential()

    # add LSTM layer for 1 year use 15 units( now 1 output(high) => 6 units).
    model.add(
        LSTM(units=40, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.1))
    model.add(LSTM(units=40, activation='relu', return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=15, activation='relu',kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.1))

    # add output layer
    model.add(Dense(units=1, activation='linear'))

    # compile model
    model.compile(optimizer='rmsprop', loss='mean_squared_error')

    def scheduler(epoch, lr):
        if epoch % 10 != 0 or epoch == 0:
            return lr
        else:
            return lr * 0.85

    callback_alpha = LearningRateScheduler(scheduler)

    # fit model
    callback = StopWhenLossBelowThreshold(threshold=0.006)
    history = model.fit(x_train, y_train, epochs=120, validation_data=(x_test, y_test), batch_size=16,
                        callbacks=[callback, callback_alpha])

    # predict
    y_pred = model.predict(x_test)

    # plot
    #plotPrediction(y_test, y_pred, feature_name)
    # plotLoss(history)

    return y_test, y_pred

##############################################
#               SVR                          #
##############################################
def svr(df, feature_name):

    x_train, x_test, y_train, y_test = getTimeSeries(df, 12, feature_name)

    #connect x_train and x_test
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    tscv = TimeSeriesSplit(n_splits=5)

    # Parameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1, 1],
        'epsilon': [0.001, 0.01, 0.1, 0.2]
    }
    from sklearn.svm import SVR

    # Initialize SVR
    svr = SVR(kernel='linear')
    gsearch = GridSearchCV(estimator=svr, cv=tscv, param_grid=param_grid, scoring='neg_mean_squared_error', verbose=1)

    x = np.reshape(x, (x.shape[0], -1))
    x_train = np.reshape(x_train, (x_train.shape[0], -1))

    # Fit grid search
    gsearch.fit(x_train, y_train)

    # Get best parameters and score
    best_params = gsearch.best_params_
    best_score = gsearch.best_score_
    print(f'Best parameters: {best_params}')
    print(f'Best score: {best_score}')

    '''Best
    parameters: {'C': 1, 'epsilon': 0.01, 'gamma': 'scale'}
'''
    # Create SVR with best parameters
    svr = SVR(kernel='linear', C=best_params['C'], gamma=best_params['gamma'], epsilon=best_params['epsilon'], verbose=1)


    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    # Fit SVR
    svr.fit(x_train, y_train)

    # Predict
    y_pred = svr.predict(x_test)

    # Plot
    #plotPrediction(y_test, y_pred, feature_name)

    return y_test, y_pred


###############################################
#               Random Forest                  #
###############################################
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

def random_forest(df, feature_name):

    x_train, x_test, y_train, y_test = getTimeSeries(df, 12, feature_name)

    # Connect x_train and x_test
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    tscv = TimeSeriesSplit(n_splits=5)

    # Parameter grid
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize RandomForestRegressor
    rfr = RandomForestRegressor()

    # GridSearchCV
    '''gsearch = GridSearchCV(estimator=rfr, cv=tscv, param_grid=param_grid, scoring='neg_mean_squared_error', verbose=1)

    x = np.reshape(x, (x.shape[0], -1))
    # Fit grid search
    gsearch.fit(x, y)

    # Get best parameters and score
    best_params = gsearch.best_params_
    best_score = gsearch.best_score_
    print(f'Best parameters: {best_params}')
    print(f'Best score: {best_score}')

    # Create RandomForestRegressor with best parameters
    rfr = RandomForestRegressor(
        criterion='friedman_mse',
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf']
    )'''
    rfr = RandomForestRegressor(
        criterion='friedman_mse',
        n_estimators=50,
        min_samples_split=4,
        min_samples_leaf=10
    )

    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    # Fit RandomForestRegressor
    rfr.fit(x_train, y_train)

    # Predict
    y_pred = rfr.predict(x_test)

    # Plot
    #plotPrediction(y_test, y_pred, feature_name)

    return y_test, y_pred


###############################################
#               Gradient Boosting              #
###############################################

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def xgboost_model(df, feature_name):

    x_train, x_test, y_train, y_test = getTimeSeries(df, 12, feature_name)

    # Reshape x_train and x_test
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Parameter grid
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.05, 0.1]
    }

    # Initialize XGBoost
    model = xgb.XGBRegressor(objective ='reg:squarederror')  # 'reg:squarederror' is a common objective for regression tasks
    gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_grid, scoring='neg_mean_squared_error', verbose=1)

    # Fit grid search
    gsearch.fit(x_train, y_train)

    # Get best parameters and score
    best_params = gsearch.best_params_
    best_score = gsearch.best_score_
    print(f'Best parameters: {best_params}')
    print(f'Best score: {best_score}')

    '''Best
    parameters: {'learning_rate': 0.1, 'n_estimators': 100}
'''
    # Create XGBoost with best parameters
    model = xgb.XGBRegressor(
        booster = 'gblinear',
        objective ='reg:squarederror',
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate']
    )

    # Fit XGBoost
    model.fit(x_train, y_train)

    # Predict
    y_pred = model.predict(x_test)

    # Plot
    #plotPrediction(y_test, y_pred, feature_name)

    return y_test, y_pred



###############################################
#               comparision                    #
###############################################
def comparision(df):
    y_test, lstm_pred = LSTMTest(df, 'Close')
    lstm_pred = np.reshape(lstm_pred, (-1))
    _, svr_pred = svr(df, 'Close')

    _, rf_pred = random_forest(df, 'Close')

    _, xgb_pred = xgboost_model(df, 'Close')

    # Create a new 2x2 subplots
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=(
                        'SVR Prediction', 'Random Forest Prediction', 'XGBoost Prediction', 'LSTM Prediction'))

    # SVR Prediction subplot
    fig.add_trace(go.Scatter(y=svr_pred, mode='lines', name='SVR Predicted Close', line=dict(color='blue')), row=1,
                  col=1)
    fig.add_trace(go.Scatter(y=y_test, mode='lines', name='Real Close', line=dict(color='red')), row=1, col=1)

    # Random Forest Prediction subplot
    fig.add_trace(go.Scatter(y=rf_pred, mode='lines', name='Random Forest Predicted Close', line=dict(color='green')),
                  row=1, col=2)
    fig.add_trace(go.Scatter(y=y_test, mode='lines', showlegend=False, line=dict(color='red')), row=1, col=2)

    # XGBoost Prediction subplot
    fig.add_trace(go.Scatter(y=xgb_pred, mode='lines', name='XGBoost Predicted Close', line=dict(color='orange')),
                  row=2, col=1)
    fig.add_trace(go.Scatter(y=y_test, mode='lines', showlegend=False, line=dict(color='red')), row=2, col=1)

    # LSTM Prediction subplot
    fig.add_trace(go.Scatter(y=lstm_pred, mode='lines', name='LSTM Predicted Close', line=dict(color='purple')), row=2,
                  col=2)
    fig.add_trace(go.Scatter(y=y_test, mode='lines', showlegend=False, line=dict(color='red')), row=2, col=2)

    # Add title and update layout
    fig.update_layout(title='Close Prediction by Date',
                      height=600,  # Adjust the height
                      width=900,
                      legend=dict(
                          x=0,
                          y=0.6,
                          orientation='h',
                          bgcolor='rgba(255, 255, 255, 0.5)')
                      )

    # Display the figure
    fig.show()

#########################################
#               main                    #
#########################################

df = pd.read_csv("NVDA_With_Earning.csv")

#this necessary to get the right date
separateDate(df)
df = df[df['Year'] >= 2021]
renameColumn(df)

