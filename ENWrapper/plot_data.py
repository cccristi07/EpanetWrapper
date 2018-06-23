from ENWrapper import *
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier as nn
from sklearn.preprocessing import normalize, scale,MinMaxScaler
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.cluster import KMeans
from sklearn.manifold import t_sne
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit as omp
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib2tikz import save
from scipy.io import savemat
from sklearn.decomposition import MiniBatchDictionaryLearning, DictionaryLearning, NMF
import pickle
import matplotlib2tikz.save as tsave
# TODO:
# cplex, gurobi, mosek - programe implementate in C solvere - Python matlab etc.


# pentru matlab - YALMIP

# sa incerc sa plotez si sa clasific pentru un singur defect - s
# a vad daca se poate face clar separatie intre defect si functionare normala

# dict learning

# codul pentru fault detection & <<fault isolation>>


# ascending order of nodes - Neural Net
#nodes = [6, 13, 29, 22,  5, 25, 26, 14, 12, 30,  4, 21, 24, 23,  7, 18, 27, 28,  8, 19, 20, 15,  3, 16,  9,  1,  2, 17, 10, 11,  0]

# most important nodes
# nodes = [1,  2, 17, 10, 11,  0]

# # least important nodes
# nodes = [2, 17, 10, 11, 0]


# ascending order of nodes -SVM Linear

# ma intereseaza sa fie clasificarea buna, nu neap raritatea dictionarului
# X = D * V

nodes = list(range(1,32))

ELEMENT = "NODE_VALUES"
FEATURE = "EN_PRESSURE"
time_span = list(range(1, 35))

P1 = [11, 10]

# partitions for the network
S1 = {1,2,3,4,17, 18}
S2 = {5,6,7,8}
S3 = {19, 20, 21, 22, 27, 23, 28}
S4 = {13, 14, 16, 15, 26,25,24, 31,30, 29}
S5 = {9, 10, 11, 12}
S = [{22, 23, 24, 27, 28, 29, 30, 31}, {1, 2, 3, 16, 17, 18, 19}, {20, 21}, {4, 5, 6, 7, 8, 9, 10, 11, 12, 13}, {25, 26, 14, 15}]
# S = [{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}, {19, 22, 23, 24, 25, 27, 28, 29, 30, 31}, {14, 15, 16, 17, 18, 26}, {20, 21}]
def residual(measured, ref):
    r = (np.mean(measured[time_span], axis=0) - np.mean(ref[time_span], axis=0))
    return r


def get_data(train_path='data/train_set.json', test_path='data/test_set.json', element='NODE_VALUES', feature='EN_PRESSURE', time_span=None, load_pkl=False,
             train_pkl='data/train_set.pkl', test_pkl='data/test_set.pkl', reduce_singularities=False, partitioned=False):

    if load_pkl:
        with open(train_pkl, 'rb') as f:
            train_data = pickle.load(f)
        with open(test_pkl, 'rb') as f:
            test_data = pickle.load(f)
    else:

        with open(train_path) as f:
            json_str = f.read()

        train_data = json.loads(json_str)

        with open(test_path) as f:
            json_str = f.read()

        test_data = json.loads(json_str)

    ref = np.array(train_data[ELEMENT][0][FEATURE])
    print("ref size is {}".format(np.shape(ref)))


    X_train = []
    y_train = []
    mag_train = [] # vector care retine magnitudinile defectului

    X_test = []
    y_test = []
    mag_test = []

    for val in train_data[element][0:]:
        measured = np.array(val[feature])

        residue = residual(ref, measured)

        if reduce_singularities:
            if val["EMITTER_VAL"] > 3:
                X_train.append(residue)
                if partitioned:
                    for partition, s in enumerate(S):

                        if val["EMITTER_NODE"] in s:
                            y_train.append(partition+1)
                            break
                else:
                    y_train.append(val["EMITTER_NODE"])
                mag_train.append(val["EMITTER_VAL"])
        else:
            X_train.append(residue)
            if partitioned:
                for partition, s in enumerate(S):

                    if val["EMITTER_NODE"] in s:
                        y_train.append(partition+1)
                        break
            else:
                y_train.append(val["EMITTER_NODE"])
                mag_train.append(val["EMITTER_VAL"])

    for val in test_data[ELEMENT][2:]:
        measured = np.array(val[FEATURE])

        residue = residual(ref, measured)

        if reduce_singularities:
            if val["EMITTER_VAL"] > 3:
                X_test.append(residue)
                if partitioned:
                    for partition, s in enumerate(S):
                        if val["EMITTER_NODE"] in s:
                            y_test.append(partition+1)
                            break
                else:
                    y_test.append(val["EMITTER_NODE"])
                    mag_train.append(val["EMITTER_VAL"])
        else:
            X_test.append(residue)
            if partitioned:
                for partition, s in enumerate(S):
                    if val["EMITTER_NODE"] in s:
                        y_test.append(partition+1)
                        break
            else:
                y_test.append(val["EMITTER_NODE"])
                mag_train.append(val["EMITTER_VAL"])

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    X_train = normalize(X_train)
    X_test = normalize(X_test)

    shuffle = list(range(1, len(y_train)))
    np.random.shuffle(shuffle)
    X_train = X_train[shuffle]
    y_train = y_train[shuffle]

    X_train = np.squeeze(X_train)
    X_test = np.squeeze(X_test)

    y_test = np.squeeze(y_test)
    y_train = np.squeeze(y_train)

    return X_train, y_train, X_test, y_test, mag_train, mag_test

def rfe_scores(clf, X_train, y_train, X_test, y_test, n_nodes=None):

    if n_nodes is None:
        n_nodes = [1, 4, 6, 10, 15, 22]

    clfs = []
    for n in n_nodes:
        rclf = RFE(clf, n_features_to_select=n)
        rclf.fit(X_train, y_train)

        print("RFE {} selected nodes {}".format(n, rclf.get_support(True)))
        print("RFE {} score is {}".format(n, rclf.score(X_test, y_test)))
        clfs.append(rclf)

    return clfs

def dim_reduction(X, y, n_comp=2, fault_nodes=None):
    tsne = t_sne.TSNE(n_components=n_comp, perplexity=5)

    X_red = tsne.fit_transform(X)

    if fault_nodes:
        flt_plots = []

        X_tsne = []
        y_tsne = []
        for node in fault_nodes:
            X_node = [X_red[n, :] for n in range(len(X_red)) if y[n] == node]
            X_tsne.extend(X_node)
            y_node = [y_train[n] for n in range(len(X_red)) if y[n] == node]
            y_tsne.extend(y_node)
            X_1 = [X_red[n, 0] for n in range(len(X_red)) if y[n] == node]
            X_2 = [X_red[n, 1] for n in range(len(X_red)) if y[n] == node]
            fig = plt.scatter(X_1, X_2)
            flt_plots.append(fig)

        leg = ["f{}".format(n) for n in fault_nodes]
        plt.legend(leg)
        plt.show()

    else:
        return X_red

def clf_grid_search(clf, X_train, y_train, X_test, y_test, param_grid=None):
    gs = GridSearchCV(clf, param_grid=param_grid, )

    gs.fit(X_train, y_train)

    print("{} bets params are {}".format(type(clf).__name__, gs.best_params_))
    print("{} score is {}".format(type(gs).__name__, gs.score(X_test, y_test)))


def clf_comparison(clfs, X_train, y_train, X_test, y_test):

    for clf in clfs:
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print("{} scored {} on test set".format(type(clf).__name__, score))


def dl_classification(dl,X_train, y_train, X_test, y_test, H_train=None, H_test=None, n_comp=256, sparsity=4,):

    r, c = X_train.shape

    if r > c:
        X_train = X_train.transpose()
        X_test = X_test.transpose()

    if H_train is None:
        # building label matrix H_train
        H_train = np.zeros(shape=X_train.shape)

        for index, flt in enumerate(y_train):
            H_train[flt-1, index] = 1
    if H_test is None:
        H_test = np.zeros(shape=X_test.shape)
        for index, flt in enumerate(y_test):
            H_test[flt-1, index] = 1


    # learn the dictionary from test data
    Xd_train = dl.fit_tranform(X_train)
    Xd_test = dl.transform(X_test)

    D = dl.components_

    # now we use NMF to do classification
    nmf = NMF()


param_grid = [
  {'C': [1, 5, 10, 15, 25, 50, 100, 1000], 'kernel': ['linear'], 'class_weight': ['balanced', None]},
  {'C': [1, 10, 15, 25, 50, 100, 1000], 'gamma': [0.001, 0.0001, 0.01, 0.00001], 'kernel': ['rbf'] , 'class_weight':['balanced', None]},
 ]
X_train, y_train, X_test, y_test, mag_train, mag_test = get_data(time_span=time_span, load_pkl=True, reduce_singularities=False, partitioned=True)
# dim_reduction(X_train, y_train, fault_nodes=[3, 7, 11, 15, 17, 27, 28])
# rf = RandomForestClassifier(n_estimators=10,)
# ab = AdaBoostClassifier()
# gb = GradientBoostingClassifier()
# dt = DecisionTreeClassifier()
# clfs = [rf, ab, gb, dt]
# clf_comparison(clfs, X_train, y_train, X_test, y_test)
clf_grid_search(SVC(), X_train, y_train, X_test, y_test, param_grid)


svm_rfe = SVC(kernel='linear', C=15, class_weight='balanced')

rfe_scores(svm_rfe, X_train, y_train, X_test, y_test, n_nodes=None)


# testing for MSC sensors

MSC = [[11, 15,21,28],
[12,13,16,21,26,28],
[6,12,13,14,15,16,21,26,27,28]]

for m in MSC:

    X = X_train[:, m]
    svm = SVC(kernel='linear', C=15, class_weight='balanced')

    svm.fit(X, y_train)

    print(svm.score(X_test[:, m], y_test))






# # dl1 = DictionaryLearning()
# # Xd = dl.fit_transform(X_train, )
# # Xd_test = dl.transform(X_test)
# # D = dl.components_
# # print(D.shape)
#
# feature_names = ["node" + str(no) for no in range(0, 31)]
#
# def f_importances(coef, names):
#     imp = coef
#     imp,names = zip(*sorted(zip(imp,names)))
#     plt.barh(range(len(names)), imp, align='center')
#     plt.yticks(range(len(names)), names)
#     plt.show()
#
#
#
# print("SVM_DL Score is", svm.score(Xd_test, y_test))
# print("RF_DL score is ", rf.score(Xd_test, y_test))
# print("GB_DL score is", gb.score(Xd_test, y_test))
# print("AB_DL score is", ab.score(Xd_test, y_test))
# print("DT_DL score is", dt.score(Xd_test, y_test))
# # TODO add dictionary learning method
# # TODO set covering problem
#
#
#
# #min alpha*|| H - W*X || + ||Ri - D*x||
#
# import matplotlib.pyplot as plt
#
#
# def make_meshgrid(x, y, h=.02):
#     """Create a mesh of points to plot in
#
#     Parameters
#     ----------
#     x: data to base x-axis meshgrid on
#     y: data to base y-axis meshgrid on
#     h: stepsize for meshgrid, optional
#
#     Returns
#     -------
#     xx, yy : ndarray
#     """
#     x_min, x_max = x.min() - 1, x.max() + 1
#     y_min, y_max = y.min() - 1, y.max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
#     return xx, yy
#
#
# def plot_contours(ax, clf, xx, yy, **params):
#     """Plot the decision boundaries for a classifier.
#
#     Parameters
#     ----------
#     ax: matplotlib axes object
#     clf: a classifier
#     xx: meshgrid ndarray
#     yy: meshgrid ndarray
#     params: dictionary of params to pass to contourf, optional
#     """
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     out = ax.contourf(xx, yy, Z, **params)
#     return out
#
#
#
# # Take the first two features. We could avoid this by using a two-dim dataset
# X_tsne = normalize(np.array(X_tsne))
# y_tsne = y_tsne
#
# # we create an instance of SVM and fit out data. We do not scale our
# # data since we want to plot the support vectors
# C = 1.0  # SVM regularization parameter
# clf = SVC(kernel='linear', C=10)
# clf.fit(X_tsne, y_tsne)
#
# levels = sorted(list(set(y_tsne)))
# # Set-up 2x2 grid for plotting.
# fig, ax = plt.subplots()
#
# X0, X1 = X_tsne[:, 0], X_tsne[:, 1]
# xx, yy = make_meshgrid(X0, X1)
#
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.contourf(xx, yy, Z)
# b_plots = []
# y_node
# for node in fault_nodes:
#     y_node.append(node)
#     X_1 = [X_tsne[n, 0] for n in range(len(X_tsne)) if y_tsne[n] == node]
#     X_2 = [X_tsne[n, 1] for n in range(len(X_tsne)) if y_tsne[n] == node]
#     fig = plt.scatter(X_1, X_2)
#     flt_plots.append(fig)
# plt.legend(['a', 'b','c','d'])
#
#
# # plt.scatter(X0[:], X1[:], c=y_tsne, s=20, edgecolors='k')
# # plt.scatter(X0[17:33], X1[17:33], s=20, edgecolors='k')
# # plt.scatter(X0, X1, c=y_tsne, s=20, edgecolors='k')
# # plt.scatter(X0, X1, c=y_tsne, s=20, edgecolors='k')
#
# # plt.set_xlim(xx.min(), xx.max())
# # plt.set_ylim(yy.min(), yy.max())
# plt.xlabel('x1 tsne')
# plt.ylabel('x2 tsne')
# # plt.show()
#
# #min alpha*|| H - W*X || + ||Ri - D*x||