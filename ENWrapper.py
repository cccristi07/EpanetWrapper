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
class EPANetSimulation(EPANetSimulation):

    def query_network(self, query, ret_type="JSON"):
        """

        :param query: a dict containing info about the network
        has the form
        {
            simulation_name : "name",
            simulation_type: "H" or "Q"
            query_data : {

                nodes : [ "EN_PRESSURE"
                ]

                links : [ "EN_VELOCITY"
                ]

            }

        }
        :param ret_type: JSON or numpy array
        :return:
        """

        # for the moment i'll treat only hydraulic simulations :)

        # initialize network simulaton
        self.ENopenH()

        # initialize session
        self.ENinitH(10)


        # get info about the network
        no_nodes = len(self.network.nodes)
        no_links = len(self.network.links)

        # check json for querried data



        try:
            if query["query_data"]["nodes"]:
                node_values = {}
                for info_type in query["query_data"]["nodes"]:
                    node_values[info_type] = [[] for _ in range(no_nodes)]

        except:
            node_values = False


        try:
            if query["query_data"]["links"]:
                link_values = {}
                for info_type in query["query_data"]["links"]:
                    link_values[info_type] = [[] for _ in range(no_nodes)]
        except:
            link_values = False



        # time step

        t_step = 1

        while t_step > 0:

            self.ENrunH()

            if node_values:
                for node_index in range(no_nodes):
                    for info_type in query["query_data"]["nodes"]:
                        ret_val = self.ENgetnodevalue(node_index, eval(info_type))
                        ret_val = ret_val[1]
                        node_values[info_type][node_index].append(ret_val)


            if link_values:
                for link_index in range(no_nodes):
                    for info_type in query["query_data"]["links"]:
                        ret_val = self.ENgetnodevalue(link_index, eval(info_type))
                        ret_val = ret_val[1]
                        link_values[info_type][link_index].append(ret_val)

            t_step = self.ENnextH()
            t_step = t_step[1]




        return {
            "NODE_VALUES": node_values,
            "LINK_VALUES": link_values
        }



if __name__ == '__main__':




    from epanettools import epanettools as et

    es = EPANetSimulation("hanoi.inp")


    query_dict = {
        "simulation_name": "name",

        "simulation_type": "H",

        "query_data": {

            "nodes": ["EN_PRESSURE", "EN_DEMAND"],

            "links": ["EN_VELOCITY"]

        }

    }
    ret_val = es.query_network(query_dict)
    import pprint

    pprint.pprint(ret_val)
