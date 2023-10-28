
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

# pca = PCA(n_components=0.75)
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)
#
# accuracy, cm = train_evaluate(classifer, X_train_pca, X_test_pca, y_train, y_test)
# print("PCA Accuracy: ", accuracy)
# print("Cm: ")
# print(cm)

