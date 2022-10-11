import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


class App:

    def __init__(self, csv_path):
        self.standard_data = None
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.csv_path = csv_path
        self.dataset = pd.read_csv(csv_path)

        self.regressor = RandomForestRegressor()
        self.test_data_prediction = None

    def standardize_data(self):
        self.x = self.dataset.drop(['Date', 'GLD'], axis=1)
        self.y = self.dataset['GLD']

    def fit(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.1,
                                                                                random_state=2)

        self.regressor.fit(self.x_train, self.y_train)

    def predict(self):
        self.test_data_prediction = self.regressor.predict(self.x_test)

    def error_score(self):
        error_score = metrics.r2_score(self.y_test, self.test_data_prediction)
        print("R scored Error: ", error_score)

    def get_y_test(self):
        return self.y_test

    def get_test_prediction(self):
        return self.test_data_prediction

