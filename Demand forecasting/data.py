import math

import pandas as pd
from sklearn import metrics
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, normalize
import numpy as np
from sklearn.svm._libsvm import cross_validation
import matplotlib.pyplot as plt


def random_forest(data):
    data_1 = data.copy()
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=20, random_state=4)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Mean absolute per error ', metrics.r2_score(y_test, y_pred))
    #cross = cross_val_score(model, XX, yy, cv=5, scoring='neg_mean_absolute_error')
    #print(cross.mean())
    x_ax = range(len(X_test))
    plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
    plt.plot(x_ax, y_pred, lw=0.8, color="red", label="predicted")
    plt.legend()
    plt.show()

def elastic_model(data):
    X = data.iloc[:, 1:].values
    y = data[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]
    for a in alphas:
        model = ElasticNet(alpha=a).fit(X, y)
        score = model.score(X, y)
        pred_y = model.predict(X)
        mse = mean_squared_error(y, pred_y)
        print("Alpha:{0:.4f}, R2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}"
              .format(a, score, mse, np.sqrt(mse)))

    elastic = ElasticNet(alpha=0.01).fit(X_train, y_train)
    y_pred = elastic.predict(X_test)
    score = elastic.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    print("R2:{0:.3f}, MSE:{1:.2f}, RMSE:{2:.2f}"
          .format(score, mse, np.sqrt(mse)))

    x_ax = range(len(X_test))
    plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
    plt.plot(x_ax, y_pred, lw=0.8, color="red", label="predicted")
    plt.legend()
    plt.show()

    elastic_cv = ElasticNetCV(alphas=alphas, cv=5)
    model = elastic_cv.fit(X_train, y_train)
    print(model.alpha_)
    print(model.intercept_)

    ypred = model.predict(X_test)
    score = model.score(X_test, y_test)
    mse = mean_squared_error(y_test, ypred)
    print("R2:{0:.3f}, MSE:{1:.2f}, RMSE:{2:.2f}"
          .format(score, mse, np.sqrt(mse)))

    x_ax = range(len(X_test))
    plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
    plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
    plt.legend()
    plt.show()


def grad_boost(data):
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(random_state=228, loss='huber')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Mean absolute per error ', metrics.r2_score(y_test, y_pred))
    x_ax = range(len(X_test))
    plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
    plt.plot(x_ax, y_pred, lw=0.8, color="red", label="predicted")
    plt.legend()
    plt.show()

data = pd.read_csv('test1.txt', header=None, delim_whitespace=True)


#random_forest(data)
#elastic_model(data)
grad_boost(data)

