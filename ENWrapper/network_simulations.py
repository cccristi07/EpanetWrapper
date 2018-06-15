from ENWrapper import ENSim
from pandas import Series
import pandas as pd
import datetime
import matplotlib2tikz
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier as nn
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

NETFILE = 'data/hanoi.inp'

# class NetworkSimulation(object):
#
#     def __init__(self, network_file=None):
#         self.en_sim = ENWrapper.ENSim(network_file)
#
#
#     def get_data(self):
#         pass
#
#     def set_attr(self):
#         pass


if __name__ == '__main__':

    nodes = list(range(1, 31))

    vals = [5, 15, 20, 40]

    emitters = [(6, val) for val in vals]
    query = {

        "simulation_name": "Hanoi intense leak simulation",
        "simulation_type": "H",
        "emitter_values": emitters,
        "query": {

            "nodes": ["EN_PRESSURE"]
        }

    }

    # for node in nodes:
    #     for val in vals:
    #         query["query"]["nodes"] = [(node, val)]
    #
    #         simulation = ENSim(NETFILE)
    #
    #         sim_dict = simulation.query_network(query)
    #         data



    simulation = ENSim(NETFILE)

    sim_dict = simulation.query_network(query)
    date_range = pd.date_range('1/1/2018', periods=97, freq='15min')

    for data in sim_dict["NODE_VALUES"]:
        pressures = np.array(data["EN_PRESSURE"])

        for node in range(31):
            p = data[:, node]
            ts = Series(p, index= date_range)
            ts.plot()



    # with open("node_5_sim.json", "w") as f:
    #     f.write(data)
