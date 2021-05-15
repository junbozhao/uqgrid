from .psysdef import Psystem, GenGENROU, ExcESDC1A, GovIEESGO, MotCIM5
from .parse_psse import read_raw
import scipy.io as sio
import numpy as np

def load_psse(raw_filename):

    case = read_raw(raw_filename)

    nbus = len(case.buses)
    nbranch = len(case.branches)
    nloads = len(case.loads)
    psse_to_int = {}
    
    baseMVA = case.baseMVA
    psys = Psystem(basemva=float(baseMVA))

    # add buses
    for i in range(nbus):
        psys.add_bus(i, bus_type=case.buses[i].type)
        psys.buses[i].set_vinit(case.buses[i].vm, (np.pi/180.0)*case.buses[i].va)

        psse_to_int[case.buses[i].busn] = i

    # add branches
    for branch in case.branches:
        fr_internal = psse_to_int[int(branch.fbus)]
        to_internal = psse_to_int[int(branch.tbus)]

        psys.add_branch(fr_internal, to_internal, branch.r, branch.x, 
                sh=branch.b)
    # add transformers
    for tran in case.transformers:
        
        fr_internal = psse_to_int[int(tran.fbus)]
        to_internal = psse_to_int[int(tran.tbus)]

        if tran.CW == 2:
            assert False, "Not implemented yet"
            volt2 = tran.WINDV2
            volt1 = tran.WINDV1
        else:
            volt2 = tran.WINDV2
            volt1 = tran.WINDV1

        if tran.CZ == 1:
            r12 = tran.r*(volt2)**2.0
            x12 = tran.x*(volt2)**2.0
        elif tran.CZ == 2:
            r12 = tran.r*(basemva/case.sbase12)*(volt2)**2.0
            x12 = tran.x*(basemva/case.sbase12)*(volt2)**2.0
        elif tran.CZ == 3:
            assert False, "Not implemented yet"

        tap = (volt1/volt2)
        psys.add_branch(fr_internal, to_internal, r12, x12, 
                0.0, tap=tap, shift=tran.ANG1)

        if tran.COD1 == 1:
            psys.add_shunt(fr_internal, tran.MAG1*baseMVA, tran.MAG2*baseMVA)
        else:
            # We don't have this implemented. Ensure it it 0
            MAG1 = tran.MAG1
            MAG2 = tran.MAG2
            assert np.isclose(np.abs(MAG1) + np.abs(MAG2), 0.0), "Not implemented"

    # add generators
    for gen in case.gens:
        bus = psse_to_int[int(gen.busn)]
        psys.add_gen(bus, gen.pg, gen.qg, mbase=gen.mbase)

    # add loads
    for i in range(nloads):
        bus = psse_to_int[case.loads[i].busn]
        if case.loads[i].status == 1:
            # Considering only constant-power loads
            #psys.add_load(bus, case.loads[i].name, case.loads[i].pl, -case.loads[i].ql)
            psys.add_load(bus, case.loads[i].name, case.loads[i].pl + case.loads[i].yp + case.loads[i].ip, -case.loads[i].ql)
    
    # adjust alpha for buses where there are multiple loads.
    # Example, in bus 2 there are two loads:
    # ID "A" 100 MW
    # ID "B" 200 MW
    # We calculate the total power -> 300 MW
    # Then alpha = 100/300 = 100.


    for shunt in case.shunts:
        if shunt.status == 1:
            bus = psse_to_int[busn.busn]
            psys.add_shunt(bus, shunt.gshunt, shunt.bshunt) 

    psys.add_ext2int(psse_to_int)
    psys.assemble()

    return psys

def load_matpower(mat_file):

    """
        The files loaded here are the result executing the following commands in MATPOWER
        mpc = loadcase('casefile.m')
        save('casefile.mat', mpc)
    """

    case = sio.loadmat(mat_file)

    basemva = case['mpc'][0][0][1][0][0]
    mat_buses = np.array(case['mpc'][0][0][2])
    mat_gens = np.array(case['mpc'][0][0][3])
    mat_branches = np.array(case['mpc'][0][0][4])
        
    nbus = mat_buses.shape[0]
    nbranch = mat_branches.shape[0]
    ngens = mat_gens.shape[0]
    mat_to_int = {}
 
    psys = Psystem(basemva=float(basemva))

    for i in range(nbus):
        psys.add_bus(i, bus_type=mat_buses[i, 1])
        psys.buses[i].set_vinit(mat_buses[i, 7], (np.pi/180.0)*mat_buses[i, 8])
 
        # add shunt
        if mat_buses[i, 4] > 0.0 or mat_buses[i, 5] > 0.0:
                psys.add_shunt(i, mat_buses[i, 4], mat_buses[i, 5])
        # add load
        psys.add_load(i, str(i), mat_buses[i, 2], -mat_buses[i, 3])
        mat_to_int[mat_buses[i, 0]] = i

    for i in range(ngens):
        bus = mat_to_int[int(mat_gens[i, 0])]
        psys.add_gen(bus, mat_gens[i, 1], mat_gens[i, 2])

    for i in range(nbranch):
        fr_internal = mat_to_int[int(mat_branches[i, 0])]
        to_internal = mat_to_int[int(mat_branches[i, 1])]

        psys.add_branch(fr_internal, to_internal, mat_branches[i, 2], mat_branches[i, 3], 
                sh=mat_branches[i, 4], tap=mat_branches[i, 8], shift=mat_branches[i, 9])

    psys.assemble()

    return psys


def return_dyr_device(data, dev, ptr):
    ptr += 1
    while dev[-1] != '/':
        dev.extend(data[ptr].strip('\n').split())
        ptr = ptr + 1
    return ptr, dev

def add_dyr(psys, dyr_filename, verbose=False):

    assert isinstance(psys, Psystem)
    
    devices = []

    with open(dyr_filename) as f:
        data = f.readlines()
        ptr = 0
        data_len = len(data)
    
        while ptr < data_len:
            if ',' in data[ptr]:
                # Comma delimited file
                dev = data[ptr].strip('\n').split(",")
            else:
                dev = data[ptr].strip('\n').split()

            if len(dev) == 0:
                # Empty
                ptr = ptr + 1
            elif dev[0][0:2] == '//':
                # Comment
                ptr = ptr + 1
            else:
                ptr, dev = return_dyr_device(data, dev, ptr)
                devices.append(dev)
    
    for device in devices:

        if 'GENROU' in device[1]:
            if verbose:
                print("Adding GENROU at bus %d." % (int(device[0])))
            bus = psys.ext2int[int(device[0])]
            idx = str(device[2])
            T_d0p = float(device[3])
            T_d0dp = float(device[4])
            T_q0p = float(device[5])
            T_q0dp = float(device[6])
            H = float(device[7])
            D = float(device[8])
            x_d = float(device[9])
            x_q = float(device[10])
            x_dp = float(device[11])
            x_qp = float(device[12])
            x_ddp = float(device[13])
            xl = float(device[14])
            S1 = float(device[15])
            S2 = float(device[16])

            for i in range(len(psys.gens)):
                if psys.gens[i].bus == bus:
                    psys.add_gen_dynamics(psys.gens[i],
                        GenGENROU(idx, x_d, x_q, x_dp, x_qp, x_ddp,
                        xl, H, D, T_d0p, T_q0p, T_d0dp, T_q0dp))
                    break

        if 'IEESGO' in device[1]:
            bus = psys.ext2int[int(device[0])]
            gen_id = str(device[2])
            if verbose:
                print("Adding IEESGO at bus %d. GENID %s." % (int(device[0]), gen_id))

            T1 = float(device[3])
            T2 = float(device[4])
            T3 = float(device[5])
            T4 = float(device[6])
            T5 = float(device[7])
            T6 = float(device[8])
            K1 = float(device[9])
            K2 = float(device[10])
            K3 = float(device[11])

            for gen in psys.gendyn:
                if gen.bus == bus and gen.id_tag == gen_id:
                    psys.add_gov(gen, GovIEESGO(gen_id, T1, T2, T3, T4, T5, T6,
                        K1, K2, K3))
                    break

        if 'ESDC1A' in device[1]:

            bus = psys.ext2int[int(device[0])]
            gen_id = str(device[2])

            if verbose:
                print("Adding ESDC1A at bus %d. LOADID %s." % (int(device[0]), load_id))

            TR = float(device[3])
            KA = float(device[4])
            TA = float(device[5])
            TB = float(device[6])
            TC = float(device[7])
            VRMAX = float(device[8])
            VRMIN = float(device[9])
            KE = float(device[10])
            TE = float(device[11])
            KF = float(device[12])
            TF1 = float(device[13])
            SW = float(device[14])
            E1 = float(device[15])
            SE1 = float(device[16])
            E2 = float(device[17])
            SE2 = float(device[18])

            for gen in psys.gendyn:
                if gen.bus == bus and gen.id_tag == gen_id:
                    #psys.add_exc(gen, ExcESDC1A(gen_id, KA, TA, KF, TF1, KE, TE, TR, E1, E2))
                    psys.add_exc(gen, ExcESDC1A(gen_id, 20.0, 1.0, 0.7, 0.7, 7.0, 0.5, 20.4, 0.006, 0.9))
                    break


        if 'CIM5BL' in device[1]:
            bus = psys.ext2int[int(device[0])]
            load_id = str(device[2])
            if verbose:
                print("Adding CIM5BL at bus %d. LOADID %s." % (int(device[0]), load_id))
            ra = float(device[4])
            xa = float(device[5])
            xm = float(device[6])
            r1 = float(device[7])
            x1 = float(device[8])
            r2 = float(device[9])
            x2 = float(device[10])
            E1 = float(device[11])
            SE1 = float(device[12])
            E2 = float(device[13])
            SE2 = float(device[14])
            MBASE = float(device[15])
            PMULT = float(device[16])
            Hin = float(device[17])
            V1 = float(device[18])
            T1 = float(device[19])
            TB = float(device[20])
            Damp = float(device[21])
            TNOM = float(device[22])

            for load in psys.loads:
                if load.bus == bus and load.id_tag == load_id:
                    psys.add_load_dynamics(load, MotCIM5(load_id, ra, xa, xm, r1,
                        x1, Hin, Damp))
                    break
