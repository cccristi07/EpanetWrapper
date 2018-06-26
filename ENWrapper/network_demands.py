import ENWrapper
import wntr
from wntr import network
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier


inp_file = 'data/large_net.inp'

wn = wntr.network.WaterNetworkModel(inp_file)
wntr.graphics.plot_network(wn, title=wn.name)
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()
pressure = results.node["pressure"]
plt.plot(pressure)
plt.show()
# add a leak to a node

wn.split_pipe('7', '7_B', '7_leak_node')
leak_node = wn.get_node('7_leak_node')
leak_node.add_leak(wn, area=10.05, start_time=15*60*3, end_time=12*3600)
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()
pressure = results.node["pressure"]
plt.plot(pressure)
plt.show()