import ctypes
import matplotlib.pyplot as plt
from epanettools.epanettools import *
from epanettools import pdd
from epanettools.epanet2 import *
import numpy as np

class NodeProperty:
    EN_ELEVATION = 0
    EN_BASEDEMAND = 1
    EN_PATTERN = 2
    EN_EMITTER = 3
    EN_INITQUAL = 4
    EN_SOURCEQUAL = 5
    EN_SOURCEPAT = 6
    EN_SOURCETYPE = 7
    EN_TANKLEVEL = 8
    EN_DEMAND = 9
    EN_HEAD = 10
    EN_PRESSURE = 11
    EN_QUALITY = 12
    EN_SOURCEMASS = 13
    EN_INITVOLUME = 14
    EN_MIXMODEL = 15
    EN_MIXZONEVOL = 16
    EN_TANKDIAM = 17
    EN_MINVOLUME = 18
    EN_VOLCURVE = 19
    EN_MINLEVEL = 20
    EN_MAXLEVEL = 21
    EN_MIXFRACTION = 22
    EN_TANK_KBULK = 23
    EN_TANKVOLUME = 24
    EN_MAXVOLUME = 25


class CountType:
    EN_NODECOUNT = 0
    EN_TANKCOUNT = 1
    EN_LINKCOUNT = 2
    EN_PATCOUNT = 3
    EN_CURVECOUNT = 4
    EN_CONTROLCOUNT = 5


class ENWrapper:

    def __init__(self, lib_path, network_path, report_path, bin_path):

        # loading library
        self.lib = ctypes.cdll.LoadLibrary(lib_path)

        # load input file
        err_code = self.lib.ENopen(ENWrapper._cstring_convert(network_path),
                        ENWrapper._cstring_convert(report_path),
                        ENWrapper._cstring_convert(bin_path))

        if err_code == 0:
            print("Network Loaded succesfuly!")
        else:
            print("Error loading network! - {}".format(err_code))


    def load_network(self, network_path, report_path, bin_path):

        # load input file
        err_code = self.lib.ENopen(ENWrapper._cstring_convert(network_path),
                                   ENWrapper._cstring_convert(report_path),
                                   ENWrapper._cstring_convert(bin_path))

        if err_code == 0:
            print("Network Loaded succesfuly!")
        else:
            print("Error loading network! - {}".format(err_code))

    def get_count(self, type):

        ptr = ENWrapper._cptr()
        err_code = self.lib.ENgetcount(type, ptr)

        if err_code == 0:
            print("Succes")
        else:
            print("Error performing querry! - {}".format(err_code))

        return ENWrapper._getval(ptr)

    @staticmethod
    def _c_ptr():
        """
        todo - make it compatible with other types
        :return: function that returns a C pointer used for argument retrieval
        """

        return ctypes.pointer(ctypes.c_int(0))

    @staticmethod
    def _getval(c_ptr):
        """

        :return: value of a c_ptr
        """

        return c_ptr.contents.value
    @staticmethod
    def _check_errcode(code):
        if code == 0:
            print("Succes!")
        else:
            print("Error performing querry! - {}".format(code))

    @staticmethod
    def _cstring_convert(string):
        return ctypes.c_char_p(string.encode('utf-8'))

# we are extending the EPANetSimulation class to ease acces to
# simulation routines
class ENSim(EPANetSimulation):

    EN_INIT = 10

    def __init__(self, inputFileName, pdd=False):
        super().__init__(inputFileName, pdd)

    def set_emitter(self, node_index, emitter_val):
        self.ENsetnodevalue(node_index, EN_EMITTER, emitter_val)


    def set_emitters(self, emitter_info=None):

        if emitter_info is None:
            # if arg is none reset emitter values
            for node_index in self.network.nodes:
                self.ENsetnodevalue(node_index, EN_EMITTER, 0)
        else:

            for node_index, emitter_val in emitter_info:
                self.set_emitter(node_index, emitter_val)

    def get_nodes_data(self, data_query):

        no_nodes = self.ENgetcount(EN_NODECOUNT)[1] - self.ENgetcount(EN_TANKCOUNT)[1]
        t_step = 1
        node_values = {}


        for queries in data_query:
            node_values[queries] = [[] for _ in range(no_nodes)]

        # initialize network for hydraulic process

        self.ENinitH(ENSim.EN_INIT)

        while t_step > 0:

            self.ENrunH()

            for node_index in range(1, no_nodes + 1):
                for query_type in data_query:
                    ret_val = self.ENgetnodevalue(node_index, eval(query_type))[1]
                    node_values[query_type][node_index-1].append(ret_val)

            t_step = self.ENnextH()
            t_step = t_step[1]

        return node_values

    def get_links_data(self, data_query):

        no_links = self.ENgetcount(EN_LINKCOUNT)[1]
        t_step = 1
        link_values = {}

        for queries in data_query:
            link_values[queries] = [[] for _ in range(no_links)]

        while t_step > 0:
            self.ENrunH()

            for link_index in range(1, no_links + 1):
                for query_type in data_query:
                    ret_val = self.ENgetnodevalue(link_index, eval(query_type))[1]
                    link_values[query_type][link_index-1].append(ret_val)

            t_step = self.ENnextH()
            t_step = t_step[1]

        return link_values

    def query_network(self, sim_dict, ret_type="JSON"):
        '''
        :param sim_dict: a dict containing info about the network
        has the form
        {
            simulation_name : "name",
            simulation_type: "H" or "Q"
            emitter_values : [ (node_index, emitter_value) ]
            query : {
                nodes : [ "EN_PRESSURE"
                ]
                links : [ "EN_VELOCITY"
                ]
            }
        }
        :param ret_type: JSON or numpy array
        :return:
        '''

        # for the moment i'll treat only hydraulic simulations :)

        # initialize network simulaton
        self.ENopenH()

        # initialize session
        self.ENinitH(ENSim.EN_INIT)


        # check json for querried data

        # node info:
        try:
            if sim_dict["query"]["nodes"]:
                node_query = True
        except:
            node_query = False

        # link info
        try:
            if sim_dict["query"]["links"]:
                link_query = True
        except:
            link_query = False

        # emitter info:
        try:
            simulations = sim_dict["emitter_values"]
        except:
            simulations = False

        if simulations:
            node_values = []
            link_values = []

            for node_index, emitter_value in  simulations:
                print("Simulating emitter in node no{}".format(node_index))

                self.set_emitter(node_index, emitter_value)

                if node_query:
                    node_values.append(self.get_nodes_data(sim_dict["query"]["nodes"]))

                if link_query:
                    link_values.append(self.get_nodes_data(sim_dict["query"]["links"]))

                # reset emitter values everywhere in network
                self.set_emitters()

        else:

            if node_query:
                node_values = self.get_nodes_data(sim_dict["query"]["nodes"])
            else:
                node_values = []

            if link_query:
                link_values = self.get_links_data(sim_dict["query"]["links"])
            else:
                link_values = []

        self.ENcloseH()
        self.ENclose()

        return {
            "NODE_VALUES": node_values,
            "LINK_VALUES": link_values
        }


if __name__ == '__main__':

    es = ENSim("ENWrapper/data/hanoi.inp")

    emitters = [(5, -10), (5, 0), (5, 33)]

    query_dict = {
        "simulation_name": "name",

        "simulation_type": "H",
        "emitter_values" : emitters,

        "query": {

            "nodes": ["EN_PRESSURE", "EN_DEMAND"],
        }

    }

    simulations = es.query_network(query_dict)

    values = simulations["NODE_VALUES"]

    import pprint

    pprint.pprint(simulations)

    for i,vals in enumerate(values):
        plt.figure()
        plt.plot(vals["EN_PRESSURE"])
        plt.title("Demand = {}".format(emitters[i]))

    plt.show()