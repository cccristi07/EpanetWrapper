from ENWrapper import *
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier as nn
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
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
import gc

# TODO:
# cplex, gurobi, mosek - programe implementate in C solvere - Python matlab etc.


# pentru matlab - YALMIP

# sa incerc sa plotez si sa clasific pentru un singur defect - s
# a vad daca se poate face clar separatie intre defect si functionare normala

# dict learning

# codul pentru fault detection & <<fault isolation>>

from sklearn.decomposition import MiniBatchDictionaryLearning, DictionaryLearning
# ascending order of nodes - Neural Net
#nodes = [6, 13, 29, 22,  5, 25, 26, 14, 12, 30,  4, 21, 24, 23,  7, 18, 27, 28,  8, 19, 20, 15,  3, 16,  9,  1,  2, 17, 10, 11,  0]

# most important nodes
# nodes = [1,  2, 17, 10, 11,  0]

# # least important nodes
# nodes = [2, 17, 10, 11, 0]


# ascending order of nodes -SVM Linear
dl = DictionaryLearning(n_components=256,
                        transform_n_nonzero_coefs=4,
                        )
# ma intereseaza sa fie clasificarea buna, nu neap raritatea dictionarului
# X = D * V

nodes = [30, 25, 15, 26, 11, 27, 22,  3, 19, 13, 10,  1,  7,  4, 12, 23,  0,  6, 20,  9, 14, 21, 18,  5, 8,  2, 24, 17, 16, 28, 29]
nodes = list(range(1,31))

ELEMENT = "NODE_VALUES"
FEATURE = "EN_PRESSURE"


with open("train_set.json") as f:
    json_str = f.read()

data = json.loads(json_str)

ref = np.array(data[ELEMENT][0][FEATURE])
print("ref size is {}".format(np.shape(ref)))

with open("test_set.json") as f:
    json_str = f.read()

testdata = json.loads(json_str)


time_span = list(range(1,55))
X = []
y = []

X_test = []
y_test = []

def residual(measured, ref):
    return (np.mean(measured[time_span], axis=0) - np.mean(ref[time_span], axis=0))

#TODO:
#
# Pot aplica diferite filtre pentru a calcula reziduul
# diferenta absoluta
# diferenta relativa (elem_afectat - elem_ref) / elem_ref
#
# Filtram vectorii de reziduuri - Cel mai simplu media aritmetica pe o regiune constanta
# FTJ - care sa taie din spike-uri
#
#
#
X_tsne = []
y_tsne = []
for val in data[ELEMENT][0:]:
    measured = np.array(val[FEATURE])
    residue = measured - ref


    # we take the values where the system is stationary
    residue = residue[time_span]



    #normalization
    residue = normalize(residue, axis=1)
    #mean
    residue = np.mean(residue, axis=0)
    residue = residual(ref, measured)


    X.append(residue)
    y.append(val["EMITTER_NODE"])


for val in testdata[ELEMENT][2:]:
    measured = np.array(val[FEATURE])
    residue = measured - ref


    # we take the values where the system is stationary
    residue = residue[time_span]

    # normalization
    residue = normalize(residue, axis=1)
    # mean
    residue = np.mean(residue, axis=0)
    residue = residual(ref, measured)

    X_test.append(residue)
    y_test.append(val["EMITTER_NODE"])


X = np.array(X)
print(X.shape)
y = np.array(y)

X_test = np.array(X_test)
y_test = np.array(y_test)

shuffle = list(range(1, len(y)))
np.random.shuffle(shuffle)
X = X[shuffle]
y = y[shuffle]
X = normalize(X)
X_test = normalize(X_test)
y = np.squeeze(y)
X = np.squeeze(X)

y_test = np.squeeze(y_test)
X_test = np.squeeze(X_test)


# plot data in 2 dimensions
# de plotat pentru emitter + demand variat
# ce se intampla
# ce reprezinta axele
# o mica discute in legatura cu algoritmii alesi
pca = PCA(n_components=2)
# perplex = [5, 30]
# for p in perplex:
#     # plot only for emitters in node 11 and 17
#     tsne = t_sne.TSNE(perplexity=p)
#     X_red = tsne.fit_transform(X)
#     print(X_red.shape)
#     plt.scatter(X_red[:,0], X_red[:, 1], c=y)
#     plt.show()

del data
del testdata
del json_str

# network = nn(hidden_layer_sizes=(100,), activation='relu', warm_start=True, verbose=True)
svm_rfe = SVC(kernel='linear',C=10.5,verbose=False, max_iter=-1)


rfe = RFE(estimator=svm_rfe, n_features_to_select=1, step=1)
rfe.fit(X, y)
print("RFE SVM score is", rfe.score(X_test, y_test))
ranking = rfe.ranking_
print(ranking.shape)

# pt sel de senzori folosim toate datele
#

param_grid = [
  {'C': [1, 5, 10, 15, 25, 50, 100, 1000], 'kernel': ['linear'], 'class_weight': ['balanced', None]},
  {'C': [1, 10, 15, 25, 50, 100, 1000], 'gamma': [0.001, 0.0001, 0.01, 0.00001], 'kernel': ['rbf'] , 'class_weight':['balanced', None]},
 ]
gs = GridSearchCV(SVC(), param_grid=param_grid, )
plt.stem(ranking)
plt.title('ranking of each node')
plt.show()
svm = SVC(kernel='linear', C=10.5, verbose=False, max_iter=-1)
rf = RandomForestClassifier(n_estimators=10,)
ab = AdaBoostClassifier()
gb = GradientBoostingClassifier()
dt = DecisionTreeClassifier()
gb.fit(X, y)
ab.fit(X, y)
rf.fit(X, y)
dt.fit(X, y)
print("RF score is ", rf.score(X_test, y_test))
print("Adaboost score is ", ab.score(X_test, y_test))
print("GradientBoost score is ", gb.score(X_test, y_test))
print("DecTree score is ", dt.score(X_test, y_test))
dl1 = DictionaryLearning()
Xd = dl.fit_transform(X, )
Xd_test = dl.transform(X_test)
D = dl.components_
print(D.shape)

# de folosit functiile de Dict learning din repo-ul de git


# logreg = LogisticRegression(solver='liblinear', max_iter=1500, dual=True, C=1, multi_class='ovr', verbose=True)
svm.fit(Xd, y)
rf.fit(Xd, y)
ab.fit(Xd, y)
dt.fit(Xd, y)
gb.fit(Xd, y)
# logreg.fit(X, y)
feature_names = ["node" + str(no) for no in range(0, 31)]

def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()



# for epoch in range(1500):
#     #network.fit(X, y)
#     pass


# weights = svm.coef_
# log_weights = logreg.coef_

# print("Log weights - ", np.shape(log_weights))
# print("Log score : ", logreg.score(X_test,y_test))
# plt.plot(log_weights[0,:], marker='o', label='fault at 0')
# plt.plot(log_weights[1,:], marker='o', label='fault at 1')
# plt.plot(log_weights[2,:], marker='o', label='fault at 2')
# plt.plot(log_weights[3,:], marker='o', label='fault at 3')
# plt.legend()
# plt.show()


# print("SVM weights shape ", np.shape(weights))

# node_importance = np.sum(np.absolute(log_weights), axis=0)
# node_importance = node_importance / np.linalg.norm(node_importance)
# f_importances(node_importance, feature_names)


# imp_nodes = np.argsort(node_importance)
# print(imp_nodes)

# plt.plot(np.log(node_importance), 'rx')
# plt.show()
print("SVM_DL Score is", svm.score(Xd_test, y_test))
print("RF_DL score is ", rf.score(Xd_test, y_test))
print("GB_DL score is", gb.score(Xd_test, y_test))
print("AB_DL score is", ab.score(Xd_test, y_test))
print("DT_DL score is", dt.score(Xd_test, y_test))
# TODO add dictionary learning method
# TODO set covering problem



#min alpha*|| H - W*X || + ||Ri - D*x||