import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import DictionaryLearning, NMF, PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import scale, normalize
from sklearn.metrics.pairwise import euclidean_distances as ed
from sklearn.model_selection import GridSearchCV
from ENWrapper import  ENSim
from matplotlib import pyplot as plt
import json

class ResidualType:
    ABSOLUTE = 0
    RELATIVE = 1
    STANDARDIZED = 2
    NORMALIZED = 3
    SCALED = 4

TRAIN_PATH = 'train_set.json'
TEST_PATH = 'test_set.json'
TEST2_PATH = 'test2_set.json'

class FaultClassifier(object):

    def __init__(self, estimator, train_dataset_path=None, test_dataset_path=None, network_file='data/hanoi.inp',
                 feature_extraction=ResidualType.ABSOLUTE):
        """

        :param estimator: an sklearn estimator object
        :param dataset_path: path to the training JSON dataset
        :param network_file: path to the network file
        """
        self.estimator = estimator
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.network_file = network_file
        self.feature_extraction = feature_extraction
        if self.network_file is not None:
            self.sim_eng = ENSim(self.network_file)

        self.dataset = None

    def build_dataset(self, element='NODE_VALUES', feature='EN_PRESSURE', method=ResidualType.ABSOLUTE):
        print("Building dataset")
        with open(self.train_dataset_path) as f:
            json_str = f.read()

        train_data = json.loads(json_str)

        ref = np.array(train_data[element][0][feature])
        print("ref size is {}".format(np.shape(ref)))

        with open(self.test_dataset_path) as f:
            json_str = f.read()

        test_data = json.loads(json_str)

        X_train, y_train = self.get_features(train_data, element, feature, method)
        X_test, y_test = self.get_features(test_data, element, feature, method)
        self.dataset = {}
        self.dataset["X_train"] = X_train
        self.dataset["y_train"] = y_train
        self.dataset["X_test"] = X_test
        self.dataset["y_test"] = y_test

    def train_model(self, element='NODE_VALUES', feature='EN_PRESSURE', method=ResidualType.ABSOLUTE):
        """
        method used to train the estimator object provided
        :return:
        """
        if self.dataset == None:
            self.build_dataset(element, feature, method)

        X = self.dataset["X_train"]
        y = self.dataset["y_train"]
        self.estimator.fit(X, y)

    def get_scores(self):

        X_test = self.dataset["X_test"]
        y_test = self.dataset["y_test"]


        y_pred = self.estimator.predict(X_test)
        labels = None
        print(classification_report(y_test, y_pred, labels=labels))
        # print(confusion_matrix(y_test, y_pred))
        print("Model accuracy is", self.estimator.score(X_test, y_test))

    def get_features(self, data, element='NODES', feature='EN_PRESSURE',method=ResidualType.ABSOLUTE):

        # extracting refference from data
        ref = data[element][0][feature]
        X = []
        y = []

        for vals in data[element]:
            if vals["EMITTER_NODE"] == 1:
                continue
            residual = FaultClassifier.residual_func(ref, vals[feature],  method=method)
            # residual = scale(residual, 1)
            residual = normalize(residual)
            residual = np.mean(residual[35:65], axis=0)
            X.append(residual)
            y.append(vals["EMITTER_NODE"])

        X = np.array(X)
        y = np.array(y)
        shuffle = list(range(1, len(y)))
        np.random.shuffle(shuffle)
        X = X[shuffle]
        y = y[shuffle]
        y = np.squeeze(y)
        X = np.squeeze(X)

        return X, y

    def grid_search(self, param_grid=None):

        if param_grid is None:
            param_grid = [
                {
                    'C': [1, 10, 100, 1000],
                    'kernel': ['linear']
                },
                {
                    'C': [1, 10, 100, 1000],
                    'gamma': [0.001, 0.0001],
                    'kernel': ['rbf']
                }
            ]


        # finding the optimal parameter combination for the desired model
        grid_clf = GridSearchCV(self.estimator, param_grid,)

        grid_clf.fit()

    @staticmethod
    def residual_func(ref, measured, method=ResidualType.ABSOLUTE):
        """
        utility function used to compute the residual of the network
        :param ref: reference vector/matrix
        :param measured: measured vector/matrix
        :param method: method of computing
        :return: calculated residual
        """
        residuals = []
        ref = np.array(ref)
        measured = np.array(measured)
        if method == ResidualType.ABSOLUTE:
            residuals = ref - measured
            # residuals = ed(measured, ref)
        elif method == ResidualType.RELATIVE:
            residuals = (ref - measured)/ref
        elif method == ResidualType.NORMALIZED:
            residuals = ref - measured
        elif method == ResidualType.SCALED:
            pass
        elif method == ResidualType.STANDARDIZED:
            pass

        return residuals


if __name__ == '__main__':


    svm = SVC(kernel='linear', C=2.5, verbose=False)

    clf = FaultClassifier(svm, TRAIN_PATH, TEST_PATH, feature_extraction=ResidualType.ABSOLUTE )
    clf.train_model()
    clf.get_scores()
