from ENWrapper import *
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier as nn
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

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



X = []
y = []

X_test = []
y_test = []

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

for val in data[ELEMENT][0:]:
    residue = (np.array(val[FEATURE]) - ref)/ref
    # plt.figure()
    # plt.plot(residue, marker='x')
    # plt.title("Demand in node {} with emitter {}".format(val["EMITTER_NODE"], val["EMITTER_VAL"]))
    # plt.show()


    residue = residue[:, nodes]
    y.append(val["EMITTER_NODE"])

    # we take the values where the system is stationary
    residue = residue[1:35]

    #normalization
    #residue = normalize(residue, axis=1)
    residue = np.mean(residue, axis=0)

    X.append(residue)


for val in testdata[ELEMENT][2:]:
    # plt.figure()
    # plt.plot(np.array(val["EN_PRESSURE"]) - ref)
    # plt.title("Demand in node {} with emitter {}".format(val["EMITTER_NODE"], val["EMITTER_VAL"]))

    residue = (np.array(val[FEATURE]) - ref)/ref
    residue = residue[:, nodes]
    y_test.append(val["EMITTER_NODE"])

    # we take the values where the system is stationary
    residue = residue[1:35]
    # normalization
    #residue = normalize(residue, axis=1)
    # mean
    residue = np.mean(residue, axis=0)

    X_test.append(residue)

X = np.array(X)
print(X.shape)
y = np.array(y)

X_test = np.array(X_test)
y_test = np.array(y_test)

shuffle = np.random.shuffle(list(range(1, len(y))))
print(shuffle)
X = X[shuffle]
y = y[shuffle]

y = np.squeeze(y, 0)
X = np.squeeze(X, 0)



# network = nn(hidden_layer_sizes=(100,), activation='relu', warm_start=True, verbose=True)
svm = SVC(kernel='linear',C=2.5,verbose=True, max_iter=200)

# logreg = LogisticRegression(solver='liblinear', max_iter=1500, dual=True, C=1, multi_class='ovr', verbose=True)
svm.fit(X, y)
# logreg.fit(X, y)
feature_names = ["node" + str(no) for no in range(0, 31)]

def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()



for epoch in range(1500):
    #network.fit(X, y)
    pass


# weights = svm.coef_
# log_weights = logreg.coef_

# print("Log weights - ", np.shape(log_weights))
# print("Log score : ", logreg.score(X_test,y_test))
# plt.plot(log_weights[0,:], marker='o', label='fault at 0')
# plt.plot(log_weights[1,:], marker='o', label='fault at 1')
# plt.plot(log_weights[2,:], marker='o', label='fault at 2')
# plt.plot(log_weights[3,:], marker='o', label='fault at 3')
plt.legend()
plt.show()


# print("SVM weights shape ", np.shape(weights))

node_importance = np.sum(np.absolute(log_weights), axis=0)
node_importance = node_importance / np.linalg.norm(node_importance)
f_importances(node_importance, feature_names)


# imp_nodes = np.argsort(node_importance)
# print(imp_nodes)

# y_pred = svm.predict(X_test)

# y_pred = y_pred - y_test
# corect = len([ x for x in y_pred if x == 0])
# acc = corect / len(y_pred)

# print("Test accuracy is {}%".format(acc*100))
# plt.plot(np.log(node_importance), 'rx')
# plt.show()

# TODO add dictionary learning method
# TODO set covering problem