# This module implements classes and functions to parse, modify and print
# raw and dyr files.

import datetime
import xml.etree.ElementTree as ET

class BusRaw(object):

    def __init__(self, line):
        
        line = line.strip('\n').split(',')
        self.busn     = int(line[0])
        self.name     = line[1]
        self.baseKV   = float(line[2])
        self.type      = int(line[3])
        self.area     = int(line[4])
        self.zone     = int(line[5])
        self.owner    = int(line[6])
        self.vm       = float(line[7])
        self.va       = float(line[8])

    def display(self):

        print("Line number %i. Name: %s" % (self.busn, self.name))
        print("\tvm: %f, va: %f" % (self.vm, self.va))

    def toraw(self):

        raw_str = "%d,%s,%f,%d,%d,%d,%d,%f,%f" % (self.busn, self.name,
                self.baseKV, self.type, self.area, self.zone, self.owner,
                self.vm, self.va)
        return raw_str

class LoadRaw(object):

    def __init__(self, line):

        line = line.strip('\n').split(',')
        self.busn   = int(line[0])
        self.name   = line[1].strip('\'')
        self.status = int(line[2])
        self.area   = int(line[3])
        self.zone   = int(line[4])
        self.pl     = float(line[5])
        self.ql     = float(line[6])
        self.ip     = float(line[7])
        self.iq     = float(line[8])
        self.yp     = float(line[9])
        self.yq     = float(line[10])
        self.owner  = int(line[11])


    def toraw(self):

        raw_str = "%d,%s,%d,%d,%d,%f,%f,%f,%f,%f,%f,%d" % (self.busn, self.name,
                self.status, self.area, self.zone, self.pl, self.ql, self.ip,
                self.iq, self.yp, self.yq, self.owner)

        return raw_str

class ShuntRaw(object):

    def __init__(self, line):

        line = line.strip('\n').split(',')
        self.busn    = int(line[0])
        self.name    = line[1]
        self.status  = int(line[2])
        self.gshunt  = float(line[3])
        self.bshunt  = float(line[4])

    def toraw(self):

        raw_str = "%d,%s,%d,%5.3f,%5.3f" % (self.busn, self.name,
                self.status, self.gshunt, self.bshunt)
        return raw_str


class GenRaw(object):

    def __init__(self, line):

        line = line.strip('\n').split(',')
        
        self.busn   = int(line[0])
        self.name   = line[1]
        self.pg     = float(line[2])
        self.qg     = float(line[3])
        self.qt     = float(line[4])
        self.qb     = float(line[5])
        self.vs     = float(line[6])
        self.ireg   = int(line[7])
        self.mbase  = float(line[8])
        self.zr     = float(line[9])
        self.zx     = float(line[10])
        self.rt     = float(line[11])
        self.xt     = float(line[12])
        self.gtap   = float(line[13])
        self.status = int(line[14])
        self.rmpct  = float(line[15])
        self.pt     = float(line[16])
        self.pb     = float(line[17])
        self.o1     = int(line[18])
        self.f1     = float(line[19])

    def toraw(self):

        raw_str = ("%d,%s,%f,%f,%f,%f,%f,%d,%f,%f,%f,%f,%f,%f,%d,"
                "%f,%f,%f,%d,%f") % (self.busn, self.name, self.pq,
                self.qg, self.qt, self.qb, self.vs, self.ireg,
                self.mbase, self.zr, self.zx, self.rt, self.xt,
                self.gtap, self.status, self.rmpct, self.pt,
                self.pb, self.o1, self.f1)

        return raw_str

class BranchRaw(object):

    def __init__(self, line):

        line = line.strip('\n').split(',')

        self.fbus   = int(line[0])
        self.tbus   = int(line[1])
        self.ckt    = line[2]
        self.r      = float(line[3])
        self.x      = float(line[4])
        self.b      = float(line[5])
        self.rateA  = float(line[6])
        self.rateB  = float(line[7])
        self.rateC  = float(line[8])
        self.gi     = float(line[9])
        self.bi     = float(line[10])
        self.gj     = float(line[11])
        self.bj     = float(line[12])
        self.status = int(line[13])
        self.MET    = int(line[14])
        self.lenght = float(line[15])
        self.o1     = int(line[16])
        self.f1     = float(line[16])


    def toraw(self):

        raw_str = "%d,%d,%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d,%f,%i,%f" % (
                self.fbus, self.tbus, self.ckt, self.r, self.x, self.b,
                self.rateA, self.rateB, self.rateC, self.gi, self.bi,
                self.gj, self.bj, self.status, self.MET, self.lenght,
                self.o1, self.f1)
        return raw_str

class TransformerRaw(object):

    def __init__(self, line, line2, line3, line4):
        # note: fow now we assume there are no three-way transformers

        line = line.strip('\n').split(',')
        self.fbus    = int(line[0])
        self.tbus    = int(line[1])
        self.k       = int(line[2])
        self.ckt     = line[3]
        self.CW      = int(line[4])
        self.CZ      = int(line[5])
        self.CM      = int(line[6])
        self.MAG1    = float(line[7])
        self.MAG2    = float(line[8])
        self.NMETR   = int(line[9])
        self.tname   = line[10]
        self.status  = int(line[11])
        self.o1      = float(line[12])

        line = line2.strip('\n').split(',')
        self.r       = float(line[0])
        self.x       = float(line[1])
        self.sbase12 = float(line[2])

        line = line3.strip('\n').split(',')
        self.WINDV1  = float(line[0])
        self.NOMV1   = float(line[1])
        self.ANG1    = float(line[2])
        self.rateA   = float(line[3])
        self.rateB   = float(line[4])
        self.rateC   = float(line[5])
        self.COD1    = int(line[6])
        self.CONT1   = int(line[7])
        self.RMA1    = float(line[8])
        self.RMIT    = float(line[9])
        self.VMA1    = float(line[10])
        self.VMI1    = float(line[11])
        self.NTP1    = int(line[12])
        self.TAB1    = int(line[13])
        self.CR1     = float(line[14])
        self.CX1     = float(line[15])
        self.CNXA1   = float(line[16])

        line = line4.strip('\n').split(',')
        self.WINDV2  = float(line[0])
        self.NOMV2   = float(line[1])
        


class PsystemRaw(object):
    """ class: system
        Implements a base system case, composed by buses, genrators,
        transmission lines and loads.
    """
    def __init__(self, line):
        baseMVA = line[0:12].strip('\t').split(',')
        self.baseMVA = float(baseMVA[1])

        self.buses        = []
        self.loads        = []
        self.shunts       = []
        self.branches     = []
        self.gens         = []
        self.transformers = []

    def add_buses(self, f):

        while(True):
            line = f.readline()
            if "0 / END OF BUS DATA, BEGIN LOAD DATA" in line: break
            if not line: break
            else:
                self.buses.append(BusRaw(line))
    
    def add_loads(self, f):

        while(True):
            line = f.readline()
            if "0 / END OF LOAD DATA, BEGIN FIXED SHUNT DATA" in line: break
            if not line: break
            else:
                self.loads.append(LoadRaw(line))

    def add_shunts(self, f):
        
        while(True):
            line = f.readline()
            if "0 / END OF FIXED SHUNT DATA, BEGIN GENERATOR DATA" in line: break
            if not line: break
            else:
                self.shunts.append(ShuntRaw(line))
    
    def add_gens(self, f):
        
        while(True):
            line = f.readline()
            if "0 / END OF GENERATOR DATA, BEGIN BRANCH DATA" in line: break
            if not line: break
            else:
                self.gens.append(GenRaw(line))

    def add_branches(self, f):
        
        while(True):
            line = f.readline()
            if "0 / END OF BRANCH DATA, BEGIN TRANSFORMER DATA" in line: break
            if not line: break
            else:
                self.branches.append(BranchRaw(line))
    
    def add_transformers(self, f):
        
        while(True):
            line = f.readline()
            if "0 / END OF TRANSFORMER DATA, BEGIN AREA DATA" in line: break
            if not line: break
            else:
                line2 = f.readline()
                line3 = f.readline()
                line4 = f.readline()
                self.transformers.append(TransformerRaw(line, line2, line3, line4))

    def nbus(self):
        return len(self.buses)


    def to_raw(self, filename):
        """ Prints system into a PSS/E raw datafile
            input: filename string
        """

        with open(filename, 'w') as f:

            now = datetime.datetime.now()

            f.write("0,  %3.3f      / PSS/E-33.2\n" % (self.baseMVA))
            f.write("\n\n")
            
            for bus in self.buses: f.write(bus.toraw() + '\n')
            f.write("0 / END OF BUS DATA, BEGIN LOAD DATA\n")

            for load in self.loads: f.write(load.toraw() + '\n')
            f.write("0 / END OF LOAD DATA, BEGIN FIXED SHUNT DATA\n")
            
            for shunt in self.shunts: f.write(shunt.toraw() + '\n')
            f.write("0 / END OF FIXED SHUNT DATA, BEGIN GENERATOR DATA\n")
            
            for gen in self.gens: f.write(gen.toraw() + '\n')
            f.write("0 / END OF GENERATOR DATA, BEGIN BRANCH DATA\n")
            
            for branch in self.branches: f.write(branch.toraw() + '\n')
            f.write("0 / END OF BRANCH DATA, BEGIN TRANSFORMER DATA\n")

            #if len(self.transformers):
            #    raise NameError('Transformers not tested yet')
            f.write("0 / END OF TRANSFORMER DATA, BEGIN AREA DATA\n")
            f.write("0 / END OF AREA DATA, BEGIN TWO-TERMINAL DC DATA\n")
            f.write("0 / END OF TWO-TERMINAL DC DATA, BEGIN VSC DC LINE DATA\n")
            f.write("0 / END OF VSC DC LINE DATA, " 
                    "BEGIN IMPEDANCE CORRECTION DATA\n")
            f.write("0 / END OF IMPEDANCE CORRECTION DATA, "
                     "BEGIN MULTI-TERMINAL DC DATA\n")
            f.write("0 / END OF MULTI-TERMINAL DC DATA, " 
                    "BEGIN MULTI-SECTION LINE DATA\n")
            f.write("0 / END OF MULTI-SECTION LINE DATA, BEGIN ZONE DATA\n")
            f.write("0 / END OF ZONE DATA, BEGIN INTER-AREA TRANSFER DATA\n")
            f.write("0 / END OF INTER-AREA TRANSFER DATA, BEGIN OWNER DATA\n")
            f.write("0 / END OF OWNER DATA, BEGIN FACTS DEVICE DATA\n")
            f.write("0 / END OF FACTS DEVICE DATA, BEGING SWITCHED SHUNT DATA\n")
            f.write("0 / END OF SWITCHED SHUNT DATA, BEGIN GNE DATA\n")
            f.write("0 / END OF GNE DATA, BEGIN INDUCTION MACHINE DATA\n")
            f.write("0 / END OF INDUCTION MACHINE DATA\n")
            f.write("Q\n")
            f.close()


def read_raw(filename):
    """ Reads raw data case file and creates a system object.
    """

    with open(filename, 'r') as f:
        sys = PsystemRaw(f.readline())
        
        # skip two spaces
        f.readline()
        f.readline()

        sys.add_buses(f)
        sys.add_loads(f)
        sys.add_shunts(f)
        sys.add_gens(f)
        sys.add_branches(f)
        sys.add_transformers(f)

        f.close()

    return sys