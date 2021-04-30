from __future__ import print_function

import numpy as np
import sys
from scipy import optimize
import numdifftools as nd
from numpy import linalg as LA
from scipy import optimize
import numba
from numba import jit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import scipy as sp
import math

# optional include: PETSC4py
try:
    import petsc4py
except:
    petsc4py = None
    print("Warning: using uqgrid without PETSc4py. \
            Some functionality will not be available")
if petsc4py:
    petsc4py.init(sys.argv)
    from petsc4py import PETSc

#from psysdef import Psystem
from .psysdef import Psystem, GenGENROU, ExcESDC1A, GovIEESGO, MotCIM5
from .pflow import runpf, compute_pinj, compute_pinj_alt
from .tools import matprint, csr_mult_row, csr_add_row, csr_set_row, csr_to_zeros
# supress annoying LAPACK warning on MACOS
import warnings
warnings.filterwarnings(action="ignore",
                        module="scipy",
                        message="^internal gelsd")

# Test flags
TEST_JACOBIAN = False
VERIFY_HESSIAN = False
SECONDORDER = True


def gradient_p(psys, z, theta, load_idx=0):
    """Generates gradient of the r.h.s with respect to a single parameter p.
    """

    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    pow_size = 2*psys.nbuses  # power balance equations
    sys_size = alg_size + dif_size + 2*psys.nbuses

    v = z[dif_size + alg_size:]

    # alocate gradient
    G = np.zeros(sys_size)
    psys.loads[load_idx].gradient_alpha(G[alg_size + dif_size:], z, v, theta)

    # This gradient is for CIM5 + Z load implementation.
    # Commenting out until I have a better way to select sensitivity parameters.

    # G = np.zeros(sys_size)

    # if len(psys.loads) == 1:
    #     ptot = psys.loads[0].pload
    #     qtot = psys.loads[0].qload
    # else:
    #     ptot = psys.loads[0].pload + psys.loads[1].pload
    #     qtot = psys.loads[0].qload + psys.loads[1].qload

    # v0 = psys.loads[0].v0
    # vm = v[2]

    # This would be only the derivative of the load w.r.t the weight.
    # Note that
    # pinj = (weight)*ptot + (1 - weight)*ptot
    # but (1 - weight)*ptot = pmot
    # so we only take derivative w.r.t load.

    # G[alg_size + dif_size + 2] = -(vm/v0)**2.0*ptot
    # G[alg_size + dif_size + 3] = (vm/v0)**2.0*qtot

    return G


def gradient_xp(psys, z, theta, load_idx=0):
    """Generates matrix of partial derivatives

    GX = [ d^2f_1/dx1dp d^2f_1/dx2dp ...
           d^2f_2/dx1dp d^2f_2/dx2dp ...
           ...
           d^2f_n/dx1dp d^2f_n/dx2dp ...]

    """

    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    pow_size = 2*psys.nbuses  # power balance equations
    sys_size = alg_size + dif_size + 2*psys.nbuses

    dev = alg_size + dif_size
    v = z[dif_size + alg_size:]

    # F_dalpha,dalpha (vector)
    G = np.zeros(sys_size)
    # F_dalpha, x
    GX = np.zeros((sys_size, sys_size))

    psys.loads[load_idx].gradient_pp_alpha(GX, z, v, theta, dev)

    # CIM 5 code
    ################################
    # if len(psys.loads) == 1:
    #     ptot = psys.loads[0].pload
    #     qtot = psys.loads[0].qload
    # else:
    #     ptot = psys.loads[0].pload + psys.loads[1].pload
    #     qtot = psys.loads[0].qload + psys.loads[1].qload

    # vm_idx = alg_size + dif_size + 2
    # v0 = psys.loads[0].v0
    # vm = v[2]

    # GX[vm_idx, vm_idx] =  -2.0*ptot*(vm/v0)**2.0/vm
    # GX[vm_idx + 1, vm_idx] =  2.0*qtot*(vm/v0)**2.0/vm

    return GX

def gradient_pp(psys, z, theta, idx_a=0, idx_b=0):
    """Generates matrix of partial derivatives
    """
    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    pow_size = 2*psys.nbuses  # power balance equations
    sys_size = alg_size + dif_size + 2*psys.nbuses

    # F_dalpha,dalpha (vector)
    # This is 0 for all load models. regardless
    G = np.zeros(sys_size)

    return G

def residual_function(F, z, theta, psys):

    F.fill(0.0)
    # Lock system vector
    z.flags.writeable = False

    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    pow_size = 2*psys.nbuses  # power balance equations
    sys_size = alg_size + dif_size + 2*psys.nbuses

    # Assign vectors
    x = z[:dif_size]
    y = z[dif_size:dif_size + alg_size]
    v = z[dif_size + alg_size:]

    # should I write another function?
    #compute_pinj(v[0::2], v[1::2], F[alg_size + dif_size::2],
    #        F[alg_size + dif_size + 1::2], psys.ybus)
    compute_pinj_alt(v, F[alg_size + dif_size:], psys.ybus_mat, psys.graph_mat,
                     psys.nbuses)
    F[alg_size + dif_size:] = -1.0*F[alg_size + dif_size:]

    idxs = np.zeros(4, dtype=np.int64)

    for i in range(psys.num_devices):
        idxs[0] = psys.devices[i].dif_ptr
        idxs[1] = dif_size + psys.devices[i].alg_ptr
        idxs[2] = psys.devices[i].par_ptr
        idxs[3] = psys.devices[i].bus

        ctrl_idx = psys.devices[i].ctrl_idx
        ctrl_var = psys.devices[i].ctrl_var

        psys.devices[i].residual_diff(F, z, v, theta, idxs, ctrl_idx, ctrl_var)
        psys.devices[i].residual_pinj(F[alg_size + dif_size:], z, v, None,
                                      idxs)

    for load in psys.loads:
        if load.dynamic == 0:
            load.residual_pinj(F[alg_size + dif_size:], z, v, None, None)

    for fault in psys.fault_events:
        if fault.active:
            fault.residual_pinj(F[alg_size + dif_size:], v)

    # Restore write access to system vector
    z.flags.writeable = True

    return None


@jit(nopython=True, cache=True)
def power_flow_jacobian(ybus_data, ybus_ptr, ybus_idx, J_data, J_ptr, J_idx,
                        dev, v, nbus):

    # (NOTE) This should be the maximum of n connected nodes. Which I must store as variable
    val = np.zeros(10)
    col = np.zeros(10)

    for fr in range(nbus):

        # Self power injection
        row = dev + 2*fr

        col[0] = dev + 2*fr
        col[1] = dev + 2*fr + 1
        val[0] = 0.0
        val[1] = 0.0
        csr_set_row(J_data, J_ptr, J_idx, 2, row, col, val)

        # Self power injection
        row = dev + 2*fr + 1
        col[0] = dev + 2*fr
        col[1] = dev + 2*fr + 1
        val[0] = 0.0
        val[1] = 0.0
        csr_set_row(J_data, J_ptr, J_idx, 2, row, col, val)

        conn = ybus_ptr[fr + 1] - ybus_ptr[fr]

        for i in range(conn):

            to = ybus_idx[ybus_ptr[fr] + i]
            if to == fr:

                gij = ybus_data[ybus_ptr[fr] + i].real
                bij = ybus_data[ybus_ptr[fr] + i].imag

                # Self power injection
                row = dev + 2*fr

                col[0] = dev + 2*fr
                col[1] = dev + 2*fr + 1
                val[0] = -2*v[2*fr]*gij
                val[1] = 0.0
                csr_add_row(J_data, J_ptr, J_idx, 2, row, col, val)

                # Self power injection
                row = dev + 2*fr + 1
                col[0] = dev + 2*fr
                col[1] = dev + 2*fr + 1
                val[0] = 2*v[2*fr]*bij
                val[1] = 0.0
                csr_add_row(J_data, J_ptr, J_idx, 2, row, col, val)

            else:
                angleij = v[2*fr + 1] - v[2*to + 1]

                gij = ybus_data[ybus_ptr[fr] + i].real
                bij = ybus_data[ybus_ptr[fr] + i].imag

                # P
                row = dev + 2*fr
                col[0] = dev + 2*to
                col[1] = dev + 2*to + 1
                val[0] = -v[2*fr]*(gij*np.cos(angleij) + bij*np.sin(angleij))
                val[1] = -v[2*fr]*v[2*to]*(gij*np.sin(angleij) -
                                           bij*np.cos(angleij))
                csr_set_row(J_data, J_ptr, J_idx, 2, row, col, val)

                col[0] = dev + 2*fr
                col[1] = dev + 2*fr + 1
                val[0] = -v[2*to]*(gij*np.cos(angleij) + bij*np.sin(angleij))
                val[1] = -v[2*fr]*v[2*to]*(-gij*np.sin(angleij) +
                                           bij*np.cos(angleij))
                csr_add_row(J_data, J_ptr, J_idx, 2, row, col, val)

                # Q
                row = dev + 2*fr + 1
                col[0] = dev + 2*to
                col[1] = dev + 2*to + 1
                val[0] = -v[2*fr]*(gij*np.sin(angleij) - bij*np.cos(angleij))
                val[1] = -v[2*fr]*v[2*to]*(-gij*np.cos(angleij) -
                                           bij*np.sin(angleij))
                csr_set_row(J_data, J_ptr, J_idx, 2, row, col, val)

                col[0] = dev + 2*fr
                col[1] = dev + 2*fr + 1
                val[0] = -v[2*to]*(gij*np.sin(angleij) - bij*np.cos(angleij))
                val[1] = -v[2*fr]*v[2*to]*(gij*np.cos(angleij) +
                                           bij*np.sin(angleij))
                csr_add_row(J_data, J_ptr, J_idx, 2, row, col, val)


def residual_jacobian(J, z, theta, psys):

    # Lock system vector
    z.flags.writeable = False

    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    pow_size = 2*psys.nbuses  # power balance equations
    sys_size = alg_size + dif_size + 2*psys.nbuses

    # Assign vectors
    x = z[:dif_size]
    y = z[dif_size:dif_size + alg_size]
    v = z[dif_size + alg_size:]

    # NOTE: alpha will NOT pertain to theta in the future.
    alpha = theta[3]

    # Ensure diagonals of diff. eq are 0, else the BEULER
    # routine will add h*I indefinitely.
    # NOTE: should have routine that stores diagonal entries position
    # and perfors this operation quickly.

    val = np.zeros(1)
    col = np.zeros(1)

    for i in range(dif_size):
        col[0] = i
        csr_set_row(J.data, J.indptr, J.indices, 1, i, col, val)
        #J[i, i] = 0.0

    # Power flow jacobian (TODO: move to a separate function. Maybe in pflow)
    dev = alg_size + dif_size

    power_flow_jacobian(psys.ybus_spa.data, psys.ybus_spa.indptr,
                        psys.ybus_spa.indices, J.data, J.indptr, J.indices,
                        dev, v, psys.nbuses)

    # DEVICES
    idxs = np.zeros(5, dtype=np.int64)

    for i in range(psys.num_devices):

        idxs[0] = psys.devices[i].dif_ptr
        idxs[1] = dif_size + psys.devices[i].alg_ptr
        idxs[2] = alg_size + dif_size
        idxs[3] = psys.devices[i].par_ptr
        idxs[4] = psys.devices[i].bus

        ctrl_idx = psys.devices[i].ctrl_idx
        ctrl_var = psys.devices[i].ctrl_var

        psys.devices[i].residual_jac(J, z, v, theta, idxs, ctrl_idx, ctrl_var)

    for load in psys.loads:
        if load.dynamic == 0:
            load.residual_jac(J, z, v, theta, dev)

    for fault in psys.fault_events:
        if fault.active:
            fault.residual_jac(J, z, v, theta, dev)

    # Restore write access to system vector
    z.flags.writeable = True

    return None


@jit(nopython=True, cache=True)
def power_flow_hessian(fr, ybus_data, ybus_ptr, ybus_idx, HP_data, HP_ptr,
                       HP_idx, HQ_data, HQ_ptr, HQ_idx, dev, v, nbus):

    # TODO: See residual jacobian
    val = np.zeros(10)
    col = np.zeros(10)

    pinj_vf_vf = 0.0
    qinj_vf_vf = 0.0

    pinj_af_af = 0.0
    qinj_af_af = 0.0

    pinj_vf_af = 0.0
    qinj_vf_af = 0.0

    conn = ybus_ptr[fr + 1] - ybus_ptr[fr]

    for i in range(conn):

        to = ybus_idx[ybus_ptr[fr] + i]

        if to == fr:
            gij = ybus_data[ybus_ptr[fr] + i].real
            bij = ybus_data[ybus_ptr[fr] + i].imag
            pinj_vf_vf += -2*gij
            qinj_vf_vf += 2*bij

        else:
            # compute quantities
            angleij = v[2*fr + 1] - v[2*to + 1]
            gij = ybus_data[ybus_ptr[fr] + i].real
            bij = ybus_data[ybus_ptr[fr] + i].imag

            gsin = gij*np.sin(angleij)
            bsin = bij*np.sin(angleij)
            gcos = gij*np.cos(angleij)
            bcos = bij*np.cos(angleij)

            # first, accumulate terms into the fr, fr
            pinj_vf_af -= v[2*to]*(-gsin + bcos)
            pinj_af_af -= v[2*fr]*v[2*to]*(-gcos - bsin)

            qinj_vf_af -= v[2*to]*(gcos + bsin)
            qinj_af_af -= v[2*fr]*v[2*to]*(-gsin + bcos)

            # self terms
            pinj_vt_vt = 0.0
            pinj_vt_at = -v[2*fr]*(gsin - bcos)
            pinj_at_at = -v[2*fr]*v[2*to]*(-gcos - bsin)

            qinj_vt_vt = 0.0
            qinj_vt_at = -v[2*fr]*(-gcos - bsin)
            qinj_at_at = -v[2*fr]*v[2*to]*(-gsin + bcos)

            # off terms
            pinj_vt_vf = -(gcos + bsin)
            pinj_vt_af = -v[2*fr]*(-gsin + bcos)
            pinj_vf_at = -v[2*to]*(gsin - bcos)
            pinj_at_af = -v[2*fr]*v[2*to]*(gcos + bsin)

            qinj_vt_vf = -(gsin - bcos)
            qinj_vt_af = -v[2*fr]*(gcos + bsin)
            qinj_vf_at = -v[2*to]*(-gcos - bsin)
            qinj_at_af = -v[2*fr]*v[2*to]*(gsin - bcos)

            # assemble matrices

            # VF, (VT, AT)
            row = dev + 2*fr

            col[0] = dev + 2*to
            col[1] = dev + 2*to + 1

            val[0] = pinj_vt_vf
            val[1] = pinj_vf_at
            csr_set_row(HP_data, HP_ptr, HP_idx, 2, row, col, val)
            val[0] = qinj_vt_vf
            val[1] = qinj_vf_at
            csr_set_row(HQ_data, HQ_ptr, HQ_idx, 2, row, col, val)

            # AF, (VT, AT)
            row = dev + 2*fr + 1
            val[0] = pinj_vt_af
            val[1] = pinj_at_af
            csr_set_row(HP_data, HP_ptr, HP_idx, 2, row, col, val)
            val[0] = qinj_vt_af
            val[1] = qinj_at_af
            csr_set_row(HQ_data, HQ_ptr, HQ_idx, 2, row, col, val)

            # VT, (VF, AF)
            row = dev + 2*to

            col[0] = dev + 2*fr
            col[1] = dev + 2*fr + 1

            val[0] = pinj_vt_vf
            val[1] = pinj_vt_af
            csr_set_row(HP_data, HP_ptr, HP_idx, 2, row, col, val)
            val[0] = qinj_vt_vf
            val[1] = qinj_vt_af
            csr_set_row(HQ_data, HQ_ptr, HQ_idx, 2, row, col, val)

            # AT, (VF, AF)
            row = dev + 2*to + 1
            val[0] = pinj_vf_at
            val[1] = pinj_at_af
            csr_set_row(HP_data, HP_ptr, HP_idx, 2, row, col, val)
            val[0] = qinj_vf_at
            val[1] = qinj_at_af
            csr_set_row(HQ_data, HQ_ptr, HQ_idx, 2, row, col, val)

            # VT, (VT, AT)
            row = dev + 2*to

            col[0] = dev + 2*to
            col[1] = dev + 2*to + 1

            val[0] = pinj_vt_vt
            val[1] = pinj_vt_at
            csr_set_row(HP_data, HP_ptr, HP_idx, 2, row, col, val)
            val[0] = qinj_vt_vt
            val[1] = qinj_vt_at
            csr_set_row(HQ_data, HQ_ptr, HQ_idx, 2, row, col, val)

            # AT, (VT, AT)
            row = dev + 2*to + 1
            val[0] = pinj_vt_at
            val[1] = pinj_at_at
            csr_set_row(HP_data, HP_ptr, HP_idx, 2, row, col, val)
            val[0] = qinj_vt_at
            val[1] = qinj_at_at
            csr_set_row(HQ_data, HQ_ptr, HQ_idx, 2, row, col, val)

    # VF, (VF, AF)
    row = dev + 2*fr

    col[0] = dev + 2*fr
    col[1] = dev + 2*fr + 1

    val[0] = pinj_vf_vf
    val[1] = pinj_vf_af
    csr_set_row(HP_data, HP_ptr, HP_idx, 2, row, col, val)
    val[0] = qinj_vf_vf
    val[1] = qinj_vf_af
    csr_set_row(HQ_data, HQ_ptr, HQ_idx, 2, row, col, val)

    # AF, (VF, AF)
    row = dev + 2*fr + 1
    val[0] = pinj_vf_af
    val[1] = pinj_af_af
    csr_set_row(HP_data, HP_ptr, HP_idx, 2, row, col, val)
    val[0] = qinj_vf_af
    val[1] = qinj_af_af
    csr_set_row(HQ_data, HQ_ptr, HQ_idx, 2, row, col, val)


def residual_hessian(H, z, theta, psys):

    # Lock system vector
    z.flags.writeable = False

    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    pow_size = 2*psys.nbuses  # power balance equations
    sys_size = alg_size + dif_size + 2*psys.nbuses

    # Assign vectors
    x = z[:dif_size]
    y = z[dif_size:dif_size + alg_size]
    v = z[dif_size + alg_size:]

    dev = alg_size + dif_size

    for fr in range(len(psys.graph_list)):

        # retrieve matrices
        Hp = H[dev + 2*fr]
        Hq = H[dev + 2*fr + 1]

        power_flow_hessian(fr, psys.ybus_spa.data, psys.ybus_spa.indptr,
                           psys.ybus_spa.indices, Hp.data, Hp.indptr,
                           Hp.indices, Hq.data, Hq.indptr, Hq.indices, dev, v,
                           psys.nbuses)

    # Load contribution
    for load in psys.loads:
        if load.dynamic == 0:
            load.residual_hes(H, z, v, theta, dev)

    idxs = np.zeros(5, dtype=np.int64)

    for i in range(psys.num_devices):
        idxs[0] = psys.devices[i].dif_ptr
        idxs[1] = dif_size + psys.devices[i].alg_ptr
        idxs[2] = alg_size + dif_size
        idxs[3] = psys.devices[i].par_ptr
        idxs[4] = psys.devices[i].bus

        ctrl_idx = psys.devices[i].ctrl_idx
        ctrl_var = psys.devices[i].ctrl_var

        psys.devices[i].residual_hess(H, z, v, theta, idxs, ctrl_idx, ctrl_var)

    for fault in psys.fault_events:
        if fault.active:
            fault.residual_hes(H, z, v, theta, dev)

    # verify hessian with finite differences

    if VERIFY_HESSIAN == True:

        hes_nd = nd.Hessian(function_hessian_wrapper)
        for eq in range(sys_size):
            H_ND = hes_nd(z, psys, theta, eq)
            #matprint(H_ND)
            if H[eq] is None:
                H_US = np.zeros((sys_size, sys_size))
            else:
                H_US = H[eq].todense()
            #matprint(np.array(H_US))
            is_close = np.allclose(H_US, H_ND)

            if is_close == False:
                matprint(H_ND)
                matprint(np.array(H_US))
                print(H[eq])
                assert False
            else:
                print("True")

    # Restore write access to system vector
    z.flags.writeable = True

    return None


###################################
#### Integration              #####
###################################


def first_sensitivity(psys, z, J, uold, theta, h):
    """Computes first-order sensitivity using backward Euler

    Args:
        psys (psystem): power system object
        z (np.array): power system state
        J (csr_array): power system Jacobian matrix
        uold (np.array): sensitivity vector at step k-1
        theta (np.array): parameter array
        h (float): step size (seconds)
    """

    NDIFFEQ = psys.num_dof_dif

    # Compute jacobian
    csr_to_zeros(J.data, J.indptr, J.indices)
    residual_jacobian(J, z, theta, psys)

    # integration jacobian (NOTE: refactor or move this)
    for i in range(NDIFFEQ):
        csr_mult_row(J.data, J.indptr, J.indices, i, h)
    col = np.array([0])
    data = np.array([-1.0])
    for i in range(NDIFFEQ):
        col[0] = i
        csr_add_row(J.data, J.indptr, J.indices, 1, i, col, data)

    # right hand side. One for each load
    for i in range(psys.nloads):
        b = gradient_p(psys, z, theta, load_idx=i)
        b[:NDIFFEQ] = h*b[:NDIFFEQ]
        b[:NDIFFEQ] += uold[:NDIFFEQ, i]
        b = -b
        uold[:,i] = spsolve(J, b)

def second_sensitivity(psys, x, u, J, HES, vold, theta, h):
    """
    Name: second_sensitivity
    Description: computes second order sensitivities (self sensitivities)
    """

    # Hessian will be just a list of matrices. Since seems to be very sparse, we will
    # just write None for all the 0 matrices.

    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    pow_size = 2*psys.nbuses  # power balance equations
    sys_size = alg_size + dif_size + 2*psys.nbuses

    NDIFFEQ = psys.num_dof_dif
    NEQ = sys_size


    for i in range(psys.nloads):
        GX = gradient_xp(psys, x, theta, load_idx=i)
        g = gradient_pp(psys, x, theta, idx_a=i, idx_b=i)
        ui = u[:,i]

        mu = np.zeros(NEQ)

        for j in range(NEQ):
            if HES[j] is not None:
                mu[j] = ui.dot(HES[j].dot(ui))

        mu += 2.0*np.dot(GX, ui)
        mu += g
        mu[:NDIFFEQ] = h*mu[:NDIFFEQ]
        mu[:NDIFFEQ] += vold[:NDIFFEQ, i]

        mu = -mu

        vold[:,i] = spsolve(J, mu)

def mixed_sensitivity(psys, x, u, J, HES, mold, theta, h):
    """
    Name: mixed_sensitivity
    Description: second order mixed sensitivities
    """

    # Hessian will be just a list of matrices. Since seems to be very sparse, we will
    # just write None for all the 0 matrices.

    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    pow_size = 2*psys.nbuses  # power balance equations
    sys_size = alg_size + dif_size + 2*psys.nbuses

    NDIFFEQ = psys.num_dof_dif
    NEQ = sys_size

    k = 0
    for i in range(psys.nloads):
        for j in range(i + 1, psys.nloads):
            GXi = gradient_xp(psys, x, theta, load_idx=i)
            GXj = gradient_xp(psys, x, theta, load_idx=j)
            g = gradient_pp(psys, x, theta, idx_a=i, idx_b=j)
            ui = u[:,i]
            uj = u[:,j]

            mu = np.zeros(NEQ)

            for eq_idx in range(NEQ):
                if HES[eq_idx] is not None:
                    mu[eq_idx] = ui.dot(HES[eq_idx].dot(uj))

            mu += np.dot(GXi, uj)
            mu += np.dot(GXj, ui)
            mu += g
            mu[:NDIFFEQ] = h*mu[:NDIFFEQ]
            mu[:NDIFFEQ] += mold[:NDIFFEQ, k]

            mu = -mu

            mold[:,k] = spsolve(J, mu)
            k += 1


def jacobian_beuler(J, NDIFFEQ, h):
    #J[:NDIFFEQ,:] = -h*J[:NDIFFEQ,:]
    for i in range(NDIFFEQ):
        csr_mult_row(J.data, J.indptr, J.indices, i, -h)
    #J[:NDIFFEQ, :NDIFFEQ] += sp.sparse.eye(NDIFFEQ)
    col = np.array([0])
    data = np.array([1.0])
    for i in range(NDIFFEQ):
        col[0] = i
        csr_add_row(J.data, J.indptr, J.indices, 1, i, col, data)

def jacobian_implicit(J, NDIFFEQ, a):
    # Converts the Jacobian of the r.h.s into:
    # J = [a*I-df_dx, -df_dy
    #     -dg_dx, -dg_dy]

    for i in range(J.shape[0]):
        csr_mult_row(J.data, J.indptr, J.indices, i, -1)
    
    col = np.array([0])
    data = np.array([a])
    for i in range(NDIFFEQ):
        col[0] = i
        csr_add_row(J.data, J.indptr, J.indices, 1, i, col, data)


def function_beuler_wrapper(z, zold, h, psys, theta):
    NDIFFEQ = psys.num_dof_dif
    F = np.zeros(len(z))

    residual_function(F, z, theta, psys)
    F[:NDIFFEQ] = z[:NDIFFEQ] - zold[:NDIFFEQ] - h*F[:NDIFFEQ]
    return F


def function_beuler_latin_wrapper(z, zold, h, psys, theta):
    NDIFFEQ = psys.num_dof_dif
    F = np.zeros(len(z))

    residual_function(F, z, theta, psys)
    F[:NDIFFEQ] = z[:NDIFFEQ] - zold[:NDIFFEQ] - h*F[:NDIFFEQ]

    bus_idx = psys.busmag_idx_set()

    for bidx in bus_idx:
        if z[bidx] < 0.1:
            F = F/z[bidx]

    return F


def function_hessian_wrapper(z, psys, theta, idx):
    NDIFFEQ = psys.num_dof_dif
    F = np.zeros(len(z))
    residual_function(F, z, theta, psys)
    return F[idx]


def integrate(zold,
              theta,
              h,
              psys,
              F,
              J,
              Hess,
              verbose=False,
              uold=None,
              vold=None,
              mold=None,
              fsolve=False):
    """
    Name: integrate
    Description: implements backward euler for the OMIB,
        returns x_{t + 1} given x_{t}, h and parameters.

    Notes:

        A = [In_x - h f_x, -hf]
    Args:
        xold (numpy array): state vector at (t).
        h (scalar): integration step in seconds
        e_fd (scalar): parameter
        p_m (scalar): parameter

    Output:
        x (numpy array): state vector at (t+1)
    """

    eps = 1e-10  # N-R tolerance
    max_iter = 500
    iteration = 0
    z = zold

    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    pow_size = 2*psys.nbuses  # power balance equations
    sys_size = alg_size + dif_size + 2*psys.nbuses
    NDIFFEQ = dif_size

    residual_function(F, z, theta, psys)
    F[:NDIFFEQ] = z[:NDIFFEQ] - zold[:NDIFFEQ] - h*F[:NDIFFEQ]
    norm_res = np.linalg.norm(F)

    if TEST_JACOBIAN:
        jac = nd.Jacobian(function_beuler_wrapper)

    if fsolve:
        sol, info, ier, msg = optimize.fsolve(function_beuler_latin_wrapper,
                                              zold,
                                              args=(zold, h, psys, theta),
                                              full_output=True,
                                              epsfcn=1e-9)

        if ier == 1:
            if verbose: print("Fsolve converged.")
            z = sol
        else:
            raise NameError('Fsolve did not converge')

    else:

        if verbose:
            print("Iteration %d. Residual norm: %g" % (iteration, norm_res))

        # Iterate until residual norm is below tolerance
        while (norm_res > eps) and (iteration < max_iter):
            iteration = iteration + 1

            # Form sparse jacobian matrix
            residual_jacobian(J, z, theta, psys)
            jacobian_beuler(J, NDIFFEQ, h)

            if TEST_JACOBIAN:
                Jnd = jac(z, zold, h, psys, theta)
                jacobian_nd = np.allclose(J.todense(), Jnd)
                Jdiff = J.todense() - Jnd
                np.savetxt('jac_test.csv', Jdiff, delimiter=',')
                assert jacobian_nd == True

            # step
            zdelta = spsolve(J, F)
            z = z - zdelta

            # calculate new residual
            residual_function(F, z, theta, psys)
            F[:NDIFFEQ] = z[:NDIFFEQ] - zold[:NDIFFEQ] - h*F[:NDIFFEQ]

            # print residual norm
            norm_res = np.linalg.norm(F)

            if verbose:
                print("Iteration %d. Residual norm: %g" %
                      (iteration, norm_res))

        if iteration >= max_iter:
            raise NameError('N-R solver did not converge.')

    if uold is not None:
        # Integrate 1st order sensitivity equations
        first_sensitivity(psys, z, J, uold, theta, h)

    if vold is not None and SECONDORDER:
        # Integrate 2nd order sensitivity equations
        residual_hessian(Hess, z, theta, psys)
        second_sensitivity(psys, z, uold, J, Hess, vold, theta, h)
        mixed_sensitivity(psys, z, uold, J, Hess, mold, theta, h)
    else:
        v = None
        m = None

    return z, uold, vold, mold


def initialize_system(v, p_inj, psys):
    """ Based on system parameters, creates the

        Input: v - voltage vector
               p - power flow vector
               psys - power system data structure.

        Output:
               initialized sytem vector

    """

    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    pow_size = 2*psys.nbuses  # power balance equations
    sys_size = alg_size + dif_size + 2*psys.nbuses

    sysvec = np.zeros(sys_size, dtype=np.float64)
    x = np.zeros(dif_size, dtype=np.float64)  #differential part
    y = np.zeros(alg_size, dtype=np.float64)  #algebraic part

    psys.initialize()

    assert psys.init_flag == True
    assert len(v) == psys.nbuses*2
    assert len(p_inj) == psys.nbuses*2

    p_load = psys.get_loadvec()

    for device in psys.devices:
        vm = v[2*device.bus]
        va = v[2*device.bus + 1]

        if device.model_type == "generator":
            pi = p_inj[2*device.bus] - p_load[2*device.bus]
            qi = p_inj[2*device.bus + 1] - p_load[2*device.bus + 1]
        else:
            pi = p_load[2*device.bus]
            qi = p_load[2*device.bus + 1]

        device.initialize(vm, va, pi, qi, x, y, psys)

    for load in psys.loads:
        load.base_voltage(v[2*load.bus])

    sysvec[:dif_size] = x
    sysvec[dif_size:dif_size + alg_size] = y
    sysvec[dif_size + alg_size:] = v

    # initialize theta
    theta = np.zeros(psys.num_pars)
    for i in range(psys.num_devices):
        psys.devices[i].initialize_theta(theta)

    return sysvec, theta


def initialize_sensitivities(volt, p_inj, psys, z, u, v):

    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    pow_size = 2*psys.nbuses  # power balance equations

    p_load = psys.get_loadvec()

    for device in psys.devices:
        vm = volt[2*device.bus]
        va = volt[2*device.bus + 1]

        if device.model_type == "generator":
            pi = p_inj[2*device.bus] - p_load[2*device.bus]
            qi = p_inj[2*device.bus + 1] - p_load[2*device.bus + 1]
        else:
            pi = p_load[2*device.bus]
            qi = p_load[2*device.bus + 1]

        if device.model_type == "motor":
            device.initialize_sens(vm, va, pi, qi, z, u, v, psys,
                                            dif_size)

    return None


def preallocate_jacobian(psys):

    # checks
    assert psys.init_flag == True
    assert psys.assembled == 1

    # system sizes
    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    pow_size = 2*psys.nbuses  # power balance equations
    sys_size = alg_size + dif_size + 2*psys.nbuses

    # list of lists
    list_coordinates = [[] for i in range(sys_size)]

    # DIAGONAL ENTRIES
    # (NOTE): I am preallocating diagonal entries in the differential equation part.
    # This is because BEULER will need those entries. However, I am dubius I should
    # mix the structure of the Jacobian matrix of the r.h.s and the Jacobian of the
    # BEULER problem. Performance-wise, I will go with mixing for now.

    for i in range(sys_size):
        list_coordinates[i].extend([i])

    # network equations
    ptr = alg_size + dif_size
    for fr_bus in range(len(psys.graph_list)):
        list_coordinates[ptr + 2*fr_bus].extend(
            [ptr + 2*fr_bus, ptr + 2*fr_bus + 1])
        list_coordinates[ptr + 2*fr_bus + 1].extend(
            [ptr + 2*fr_bus, ptr + 2*fr_bus + 1])

        for to_bus in psys.graph_list[fr_bus]:
            list_coordinates[ptr + 2*fr_bus].extend(
                [ptr + 2*to_bus, ptr + 2*to_bus + 1])
            list_coordinates[ptr + 2*fr_bus + 1].extend(
                [ptr + 2*to_bus, ptr + 2*to_bus + 1])

    # each device returns a list of cordinates. we add this isto a list of lists.
    for i in range(psys.num_devices):
        idxs = np.array([
            psys.devices[i].dif_ptr, dif_size + psys.devices[i].alg_ptr,
            alg_size + dif_size
        ],
                        dtype=np.int32)
        coord = psys.devices[i].preallocate_jacobian(idxs, psys)

        for j in range(len(coord)):
            if not list_coordinates[coord[j][0]]:
                list_coordinates[coord[j][0]] = coord[j][1]
            else:
                list_coordinates[coord[j][0]].extend(coord[j][1])
                list_coordinates[coord[j][0]] = list(
                    set(list_coordinates[coord[j][0]]))

    # because the ZIP load depends only on the bus voltage, no need to
    # re-compute this.

    # form coordinate lists (row, col, data)
    row = []
    col = []

    for i in range(len(list_coordinates)):
        if list_coordinates[i]:
            row.extend([i for j in range(len(list_coordinates[i]))])
            col.extend(list_coordinates[i])
    data = np.zeros(len(row))

    Jsparse = csr_matrix((data, (row, col)), shape=(sys_size, sys_size))

    return Jsparse


def coord_to_sparse(rows, cols, sys_size):
    """
    This function returns a sparse matrix given arrays "rows" and "cols"
    that have the following structure:

    rows = [a, b, c]
    cols = [[a, b], [a, b], [c]]

    Thus, rows is a list that contains the indexes of those rows which
    have non-zero entries and cols a list of lists.
    """

    # we convert to coordinate format
    row_coor = []
    col_coor = []

    for i in range(len(rows)):
        row_coor.extend([rows[i] for j in range(len(cols[i]))])
        col_coor.extend(cols[i])
    data = np.zeros(len(row_coor))

    Jsparse = csr_matrix((data, (row_coor, col_coor)),
                         shape=(sys_size, sys_size))

    return Jsparse


def preallocate_hessian(psys):

    # checks
    assert psys.init_flag == True
    assert psys.assembled == 1

    # system sizes
    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    pow_size = 2*psys.nbuses  # power balance equations
    sys_size = alg_size + dif_size + 2*psys.nbuses

    # Hessian base structure
    Hsparse = sys_size*[None]

    # Indexing structure
    h_nnz = [{'rows': [], 'cols': []} for i in range(sys_size)]

    # NETWORK
    ptr = alg_size + dif_size

    for fr_bus, connect in enumerate(psys.graph_list):
        ncon = len(connect)
        # connected buses first
        rows = [0]*2*ncon
        cols = [[]]*2*ncon
        for i in range(ncon):
            rows[2*i] = ptr + 2*connect[i]
            rows[2*i + 1] = ptr + 2*connect[i] + 1
            cols[2*i] = [
                ptr + 2*connect[i], ptr + 2*connect[i] + 1, ptr + 2*fr_bus,
                ptr + 2*fr_bus + 1
            ]
            cols[2*i + 1] = [
                ptr + 2*connect[i], ptr + 2*connect[i] + 1, ptr + 2*fr_bus,
                ptr + 2*fr_bus + 1
            ]
        # add from_bus
        idx_tobus = rows.copy()
        idx_tobus.extend([ptr + 2*fr_bus, ptr + 2*fr_bus + 1])

        rows.extend([ptr + 2*fr_bus, ptr + 2*fr_bus + 1])
        cols.extend([idx_tobus.copy(), idx_tobus.copy()])

        # Both equations have same non-zero derivatives
        h_nnz[ptr + 2*fr_bus]['rows'] = rows
        h_nnz[ptr + 2*fr_bus]['cols'] = cols
        h_nnz[ptr + 2*fr_bus + 1]['rows'] = rows
        h_nnz[ptr + 2*fr_bus + 1]['cols'] = cols

    # DEVICES
    for i in range(psys.num_devices):
        idxs = np.array([
            psys.devices[i].dif_ptr, dif_size + psys.devices[i].alg_ptr,
            alg_size + dif_size
        ],
                        dtype=np.int32)
        psys.devices[i].preallocate_hessian(h_nnz, idxs, psys)

    # assemble sparse structures
    for i in range(sys_size):
        if len(h_nnz[i]['rows']) > 0:
            Hsparse[i] = coord_to_sparse(h_nnz[i]['rows'], h_nnz[i]['cols'],
                                         sys_size)

    return Hsparse

if petsc4py:
    class DAE_petsc(object):
        n = 1
        comm = PETSc.COMM_SELF
        def __init__(self, psys, theta, J):
            self.psys = psys
            self.theta = theta
            self.J = J
        
        def evalFunction(self, ts, t, x, xdot, f):
            start, end = x.getOwnershipRange()
            NDIFFEQ = self.psys.num_dof_dif
            xx = np.array([x[i] for i in range(start, end)])
            ff = np.zeros_like(xx)
            residual_function(ff, xx, self.theta, self.psys)
            f.setArray(-ff)
            f[:NDIFFEQ] += xdot[:NDIFFEQ]
            f.assemble()
        
        def evalJacobian(self, ts, t, x, xdot, a, J, P):
            start, end = x.getOwnershipRange()
            NDIFFEQ = self.psys.num_dof_dif
            xx = np.array([x[i] for i in range(start, end)])
            residual_jacobian(self.J, xx, self.theta, self.psys)
            jacobian_implicit(self.J, NDIFFEQ, a)
            P.setValuesCSR(self.J.indptr, self.J.indices, self.J.data)
            P.assemble()
            if J != P: J.assemble()
            return True # same nonzero pattern
        
        def evalJacobianP(self, ts, t, x, xdot, a, P):
            start, end = x.getOwnershipRange()
            NDIFFEQ = self.psys.num_dof_dif
            xx = np.array([x[i] for i in range(start, end)])
            # Placeholder. Need to work on preallocating and obtaining J_p.
            mat_temp = csr_matrix(np.ones([len(xx), self.psys.nloads]))
            for i in range(self.psys.nloads):
                mat_temp[:, i] = gradient_p(self.psys, xx, self.theta, load_idx=i)
            P.setValuesCSR(mat_temp.indptr, mat_temp.indices, -mat_temp.data)
            P.assemble()
            return True # same nonzero pattern

    class ALG_petsc(object):
        n = 1
        comm = PETSc.COMM_SELF
        def __init__(self, psys, theta, J):
            self.psys = psys
            self.theta = theta
            self.J = J
        
        def evalFunction(self, snes, x, f):
            start, end = x.getOwnershipRange()
            NDIFFEQ = self.psys.num_dof_dif
            xx = np.array([x[i] for i in range(start, end)])
            ff = np.zeros_like(xx)
            residual_function(ff, xx, self.theta, self.psys)
            ff[:NDIFFEQ] = 0.0
            f.setArray(-ff)
            f.assemble()

        def evalJacobian(self, snes, x, J, P):
            start, end = x.getOwnershipRange()
            NDIFFEQ = self.psys.num_dof_dif
            xx = np.array([x[i] for i in range(start, end)])
            residual_jacobian(self.J, xx, self.theta, self.psys)
            # The following has the effect of setting the differential part of the
            # jacobian to 0 and adding 1 to the diagonal hence keeping the differential
            # part constant (projection to manifold)
            jacobian_beuler(self.J, NDIFFEQ, 0.0)
            P.setValuesCSR(self.J.indptr, self.J.indices, -self.J.data)
            P.assemble()

            if J != P: J.assemble()
            return True

    class ADJ_petsc(object):
        n = 1
        comm = PETSc.COMM_SELF
        def __init__(self, psys, theta):
            self.psys = psys
            self.theta = theta
        
        def evalCostIntegrand(self, ts, t, x, r):
            """ We will just compute the integral of the speed deviation for each generator """
            bus_idx = self.psys.genspeed_idx_set()
            cost = 0.0
            for idx in bus_idx:
                cost += x[idx]*x[idx]
            r[0] = cost
            r.assemble()
        
        def evalJacobian(self, ts, t, x, A, B):
            bus_idx = self.psys.genspeed_idx_set()
            for idx in bus_idx:
                A[idx, 0] = 2*x[idx]
            A.assemble()
            return True
        
        def evalJacobianP(self, ts, t, x, A):
            A[:,:] = 0.0
            A.assemble()
            return True
            

def integrate_system(psys,
                     tend=10.0,
                     dt=(1.0/120.0),
                     steps=-1,
                     verbose=False,
                     comp_sens=False,
                     fsolve=False,
                     ton=0.25,
                     toff=0.4,
                     petsc=False,
                     log=None):
    """integrate power system dynamics

    Args:
        psys (powersystem): power system object
        tend (float, optional): integration end (seconds). Defaults to 10.0.
        dt (float, optional): time step. Defaults to (1.0/120.0).
        steps (int, optional): if greater than 0, integration will stop after "steps". Defaults to -1.
        verbose (bool, optional): prints additional information. Defaults to False.
        comp_sens (bool, optional): computes first and second order sensitivities. Defaults to False.
        fsolve (bool, optional): uses fsolve to solve the non-liner equations. Defaults to False.
        ton (float, optional): fault activation time. Defaults to 0.25.
        toff (float, optional): fault deactivation time. Defaults to 0.4.

    Returns:
        [type]: [description]
    """
    # retrieve parameters

    volt, Pinj = runpf(psys, verbose=False)
    z0, theta = initialize_system(volt, Pinj, psys)
    system_size = z0.shape[0]

    J = preallocate_jacobian(psys)
    F = np.zeros(system_size)

    # calculate nsteps
    h = dt
    if steps > 0:
        nsteps = steps
    else:
        nsteps = int(math.floor(tend/dt)) + 1

    # hacky fault event time step calculation
    step_on = int(ton/h)
    step_off = int(toff/h)
    # Integration of D.A.E
    z = z0

    # Sensitivity parameters
    nparam = psys.nloads # For now, we only suport sensitivities of loads
    nmixed = int((nparam**2 - nparam)/2)
    
    # sensitivity variables
    if comp_sens:
        history_u = np.zeros((system_size, nparam, nsteps))
        history_v = np.zeros((system_size, nparam, nsteps))
        history_m = np.zeros((system_size, nmixed, nsteps))
        u = np.zeros((system_size, nparam))
        v = np.zeros((system_size, nparam))
        m = np.zeros((system_size, nmixed))
        Hess = preallocate_hessian(psys)
        #initialize_sensitivities(volt, Pinj, psys, z, u, v)
    else:
        history_u = None
        history_v = None
        history_m = None
        u = None
        v = None
        m = None
        Hess = None

    if petsc4py and petsc:
        if verbose: print("Convert objects to PETSc format")
        nsize = J.shape[0]
        Jp = PETSc.Mat()
        Jp.create(PETSc.COMM_WORLD)
        Jp.setSizes([nsize, nsize])
        Jp.setType('seqaij') # sparse
        csr = [J.indptr, J.indices, J.data]
        Jp.setPreallocationCSR(csr)
        Jp.assemblyBegin()
        Jp.assemblyEnd()

        Jtheta = PETSc.Mat()
        Jtheta.create(PETSc.COMM_WORLD)
        Jtheta.setSizes([nsize, nparam])
        Jtheta.setType('seqaij')
        # as a placeholder, we make this matrix dense
        mat_temp = csr_matrix(np.ones([nsize, nparam]))
        csr = [mat_temp.indptr, mat_temp.indices, mat_temp.data]
        Jtheta.setPreallocationCSR(csr)
        Jtheta.assemblyBegin()
        Jtheta.assemblyEnd()

        z0p = PETSc.Vec()
        z0p.createSeq(nsize)
        z0p.setArray(z0)
        z0p.assemblyBegin()
        z0p.assemblyEnd()

        fp = z0p.duplicate()

        # Create integration object
        dae = DAE_petsc(psys, theta, J)

        ts = PETSc.TS().create(comm=PETSc.COMM_WORLD)
        ts.setProblemType(ts.ProblemType.NONLINEAR)
        ts.setType(ts.Type.THETA)
        ts.setIFunction(dae.evalFunction, fp)
        ts.setIJacobian(dae.evalJacobian, Jp)
        ts.setIJacobianP(dae.evalJacobianP, Jtheta)

        # create adjoint integrator
        if comp_sens:
            DRDX = PETSc.Mat().createDense([nsize, 1], comm=PETSc.COMM_WORLD)
            DRDX.setUp()
            DRDP = PETSc.Mat().createDense([nparam,1], comm=PETSc.COMM_WORLD)
            DRDP.setUp()
            quad = ADJ_petsc(psys, theta)
            quadts = ts.createQuadratureTS(forward=False)
            quadts.setRHSFunction(quad.evalCostIntegrand)
            quadts.setRHSJacobian(quad.evalJacobian, DRDX)
            quadts.setRHSJacobianP(quad.evalJacobianP, DRDP)
            v_lambda = z0p.duplicate()
            v_mu = PETSc.Vec()
            v_mu.createSeq(nparam)
            v_mu.assemblyBegin()
            v_mu.assemblyEnd()
            ts.setCostGradients(v_lambda, v_mu)
            ts.setSaveTrajectory()

        historyp = []
        tvecp = []
        def monitor(ts, i, t, x):
            xx = x[:].tolist()
            historyp.append(xx)
            tvecp.append(t)
        ts.setMonitor(monitor)
        ts.setTime(0.0)
        ts.setTimeStep(dt)
        ts.setMaxTime(ton)
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.INTERPOLATE)
        ts.setFromOptions()
        ts.solve(z0p)
        
        if ton < tend:
            # fault application
            psys.fault_events[0].apply()
            alg = ALG_petsc(psys, theta, J)
            fsp = z0p.duplicate()
            snes = PETSc.SNES()
            snes.create(PETSc.COMM_WORLD)
            snes.setFunction(alg.evalFunction, fsp)
            snes.setJacobian(alg.evalJacobian, Jp)
            snes.setOptionsPrefix("alg_")
            snes.setFromOptions()
            snes.solve(None, z0p)

            # disturbance time
            ts.setTime(ton)
            ts.setMaxTime(toff)
            ts.solve(z0p)

            # fault removal
            psys.fault_events[0].remove()
            snes.solve(None, z0p)

            # post disturbance time
            ts.setTime(toff)
            ts.setMaxTime(tend)
            ts.solve(z0p)

        # adjoint computation
        if comp_sens:
            ts.adjointSolve()
            print("v_mu")
            v_mu.view()
            print("v_lambda")
            v_lambda.view()
            print("cost")
            cst = ts.getCostIntegral()
            cst.view()
            
            if log is not None:
                log["cost"] = np.array(cst[0])



        # Cast history to numpy arrays
        history = np.transpose(np.array(historyp))
        tvec = np.array(tvecp)

    else:
        tvec = np.linspace(0, nsteps*h, nsteps)
        history = np.zeros((system_size, nsteps))
        for i in range(nsteps):
            if verbose: print("Step: %i. Time: %g (sec)" % (i, i*h))
            z, u, v, m = integrate(z,
                                theta,
                                h,
                                psys,
                                F,
                                J,
                                Hess,
                                verbose=verbose,
                                fsolve=fsolve,
                                uold=u,
                                vold=v,
                                mold=m)
            history[:, i] = np.copy(z)

            if i == step_on:
                if verbose: print("Apply fault")
                if len(psys.fault_events) > 0:
                    psys.fault_events[0].apply()
                z, _, _, _ = integrate(z,
                                    theta,
                                    0.0,
                                    psys,
                                    F,
                                    J,
                                    Hess,
                                    verbose=verbose,
                                    fsolve=True,
                                    uold=None,
                                    vold=None,
                                    mold=None)
            if i == step_off:
                if verbose: print("Remove fault")
                if len(psys.fault_events) > 0:
                    psys.fault_events[0].remove()
                z, _, _, _ = integrate(z,
                                    theta,
                                    0.0,
                                    psys,
                                    F,
                                    J,
                                    Hess,
                                    verbose=verbose,
                                    fsolve=True,
                                    uold=None,
                                    vold=None,
                                    mold=None)

            if comp_sens:
                history_u[:, :, i] = np.copy(u)
                history_v[:, :, i] = np.copy(v)
                history_m[:, :, i] = np.copy(m)

        # if tend < toff we remove fault before exiting
        if i < step_off:
            psys.fault_events[0].remove()

    return tvec, history, history_u, history_v, history_m
