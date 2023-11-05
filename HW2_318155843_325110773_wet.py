import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class PerceptronClassifier:
    def __init__(self):
        """
        Constructor for the PerceptronClassifier.
        """
        # TODO - Place your student IDs here. Single submitters please use a tuple like so: self.ids = (123456789,)
        self.ids = (318155843, 325110773)
        self.k = []
        self.weights = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This method trains a multiclass perceptron classifier on a given training set X with label set y.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
        Array datatype is guaranteed to be np.uint8.
        """

        # TODO - your code here
        n_features = X.shape[1]

        self.k = set(y)

        for i in self.k:
            self.weights.append([np.zeros(n_features)])

        not_seperated = True
        while not_seperated:
            changed = True
            for i, x in enumerate(X):
                pred_y = self.predict(np.array([x]))[0]
                true_y = y[i]
                if pred_y != true_y:
                    changed = False
                    self.weights[true_y] = self.weights[true_y] + x
                    self.weights[pred_y] = self.weights[pred_y] - x
            if changed:
                not_seperated = False

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call PerceptronClassifier.fit before calling this method.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """

        # TODO - your code here
        labels = []
        for x in X:
            max_value = np.dot((self.weights[0]), x)[0]
            max_index = 0
            for i in self.k:
                value = np.dot((self.weights[i]), x)[0]
                if value > max_value:
                    max_value = value
                    max_index = i
            labels.append(max_index)
        return labels

        ### Example code - don't use this:
        # return np.random.randint(low=0, high=2, size=len(X), dtype=np.uint8)


if __name__ == "__main__":
    print("*" * 20)
    print("Started HW2_318155843_325110773_wet.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}")

    print("Initiating PerceptronClassifier")
    model = PerceptronClassifier()
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    is_separable = model.fit(X, y)
    print("Done")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y.ravel()) / y.shape[0]
    print(f"Train accuracy: {accuracy * 100 :.2f}%")

    print("*" * 20)
