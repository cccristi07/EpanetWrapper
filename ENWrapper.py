import ctypes

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

        x = ctypes.c_int(0)
        x = ctypes.pointer(x)
        err_code = self.lib.ENgetcount(type, x)

        if err_code == 0:
            print("Succes")
        else:
            print("Error performing querry! - {}".format(err_code))

        return x.contents.value

    @staticmethod
    def _check_errcode(code):
        if code == 0:
            print("Succes!")
        else:
            print("Error performing querry! - {}".format(code))

    @staticmethod
    def _cstring_convert(string):
        return ctypes.c_char_p(string.encode('utf-8'))

if __name__ == '__main__':

    net = ENWrapper('win/64/epanet2.dll', 'hanoi.inp', 'log', 'bin')
    nodes = net.get_count(CountType.EN_NODECOUNT)

    print("Number of Tanks is ",  nodes)