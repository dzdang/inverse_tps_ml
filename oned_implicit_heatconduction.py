import numpy as np
from scipy.linalg import solve_banded
from scipy.constants import sigma
import sys

def finite_difference_solve(dx, dt, alpha, T, bc=('isothermal','isothermal',1000.0,500.0)):
    """
    In this formulation, we construct a node-based discretization of the one-dimensional
    heat conduction equation with constant thermal conductivity. The boundary condition
    is embedded directly into the solver and only works for Dirchlet boundaries.

    Args:
        dx (float) :
        dt (float) :
        alpha (float) :
        T (numpy-array) :
        bc (tuple) :

    """
    ncells = np.shape(T)[0]
    R = np.zeros((ncells,),dtype=np.float64)
    A = np.zeros((3,ncells),dtype=np.float64)

    # (1,1) is for one lower and upper diagonal, i.e., a tridiagonal matrix
    form = (1,1)

    # Coefficients
    d = alpha * dt / dx**2
    a = -d
    b = 1.0 + 2.0 * d
    c = -d

    # Store matrix in a diagonal-banded format where the first and last rows are
    # the values above and below the diagonal elements.
    A[0,2:ncells] = c
    A[1,1:ncells-1] = b
    A[2,:ncells-2] = a
    R = T

    if bc[0] == 'isothermal':
        A[0,1] = 0.0
        A[1,0] = 1.0
        R[0] = bc[2]

    if bc[1] == 'isothermal':
        A[1,-1] = 1.0
        A[2,-2] = 0.0
        R[-1] = bc[3]

    return solve_banded(form, A, R)

def finite_volume_solve(dx, dt, alpha, T, bc=('isothermal','isothermal',1000.0,500.0)):
    """
    In this formulation, we construct a cell-centerd discretization of the one-dimensional
    heat conduction equation with constant thermal conductivity. The boundary conditions
    are implemented using ghost cells, which are imbedded into the system. This formulation
    can handle both Dirchlet and von Neumann boundary conditions.

    Args:
        dx (float) :
        dt (float) :
        alpha (float) :
        T (numpy-array) :

    """
    ncells = np.shape(T)[0]
    R = np.zeros((ncells,),dtype=np.float64)
    A = np.zeros((3,ncells),dtype=np.float64)

    # (1,1) is for one lower and upper diagonal, i.e., a tridiagonal matrix
    form = (1,1)

    # Coefficients
    d = alpha * dt / dx**2
    a = -d
    b = 1.0 + 2.0 * d
    c = -d

    # Interior
    for i in range(1,ncells-1):
        R[i] = T[i]

    for i in range(1,ncells):
        A[0,i] = c
    for i in range(1,ncells-1):
        A[1,i] = b
    for i in range(ncells-1):
        A[2,i] = a

    # Left
    if bc[0] == 'isothermal':
        A[0,1] = 1.0
        A[1,0] = 1.0
        R[0] = 2.0 * bc[2]
    elif bc[0] == 'adiabatic':
        A[0,1] = -1.0
        A[1,0] = 1.0
        R[0] = 0.0

    # Right
    if bc[1] == 'isothermal':
        A[1,-1] = 1.0
        A[2,-2] = 1.0
        R[-1] = 2.0 * bc[3]
    elif bc[1] == 'adiabatic':
        A[1,-1] = 1.0
        A[2,-2] = -1.0
        R[-1] = 0.0

    return solve_banded(form, A, R)

def linearized_solve(dx, dt, alpha, T, bc=('isothermal','isothermal',1000.0,500.0),
                     emissivity=1.0):
    """
    In this formulation, we construct a cell-centerd discretization of the one-dimensional
    heat conduction equation with constant thermal conductivity from the linearized form of
    the heat conduction equations.. The boundary conditions are implemented using ghost
    cells, which are imbedded into the system. This formulation can handle both Dirchlet
    and von Neumann boundary conditions.

    Args:
        dx (float) :
        dt (float) :
        alpha (float) :
        T (numpy-array) :

    """
    ncells = np.shape(T)[0]
    R = np.zeros((ncells,),dtype=np.float64)
    A = np.zeros((3,ncells),dtype=np.float64)

    # (1,1) is for one lower and upper diagonal, i.e., a tridiagonal matrix
    form = (1,1)

    # Coefficients
    d = alpha * dt / dx**2
    a = -d
    b = 1.0 + 2.0 * d
    c = -d

    T[0] = 2.0 * bc[2] - T[1]

    R[1:-1] = d * (T[:-2] - 2.0 * T[1:-1] + T[2:])
    A[0,1:] = c
    A[1,1:-1] = b
    A[2,:-1] = a

    # Left
    if bc[0] == 'isothermal':
        A[0,1] = 1.0
        A[1,0] = 1.0
        R[0] = 2.0 * bc[2] - T[1] - T[0]

    elif bc[0] == 'radiative_equilibrium':
        Twall = 0.5 * (T[0] + T[1])
        A[0,1] = alpha - 2.0 * dx * sigma * emissivity * Twall**3
        A[1,0] = -alpha - 2.0 * dx * sigma * emissivity * Twall**3
        R[0] = dx * sigma * emissivity * Twall**4 - alpha * (T[1] - T[0])

    elif bc[0] == 'adiabatic':
        A[0,1] = -1.0
        A[1,0] = 1.0
        R[0] = T[1] - T[0]

    # Right
    if bc[1] == 'isothermal':
        A[1,-1] = 1.0
        A[2,-2] = 1.0
        R[-1] = 2.0 * bc[3] - T[-1] - T[-2]

    elif bc[1] == 'radiative_equilibrium':
        Twall = 0.5 * (T[-2] + T[-1])
        A[1,-1] = alpha - 2.0 * dx * sigma * emissivity * Twall**3
        A[2,-2] = -alpha - 2.0 * dx * sigma * emissivity * Twall**3
        R[-1] = dx * sigma * emissivity * Twall**4 - alpha * (T[-2] - T[-1])

    elif bc[1] == 'adiabatic':
        A[1,-1] = 1.0
        A[2,-2] = -1.0
        R[-1] = T[-2] - T[-1]

    # print(A)
    # print('-')
    # print(R)

    return T + solve_banded(form, A, R)

# def bc_linearized_solve(dx, dt, alpha, T, bc=('isothermal','isothermal',1000.0,500.0),
#                         emissivity=0.5):
#     """
#     In this formulation, we construct a cell-centerd discretization of the one-dimensional
#     heat conduction equation with constant thermal conductivity from the linearized form of
#     the heat conduction equations.. The boundary conditions are implemented using ghost
#     cells, which are imbedded into the system. This formulation can handle both Dirchlet
#     and von Neumann boundary conditions.

#     Args:
#         dx (float) :
#         dt (float) :
#         alpha (float) :
#         T (numpy-array) :

#     """
#     ncells = np.shape(T)[0]

#     # (1,1) is for one lower and upper diagonal, i.e., a tridiagonal matrix
#     form = (1,1)

#     # Coefficients
#     d = alpha * dt / dx**2
#     a = -d
#     b = 1.0 + 2.0 * d
#     c = -d

# #    T[0] = 2 * bc[2] - T[1]

#     dT = np.zeros((ncells,),dtype=np.float64)
#     dTg = np.zeros((4,),dtype=np.float64)

#     max_iterations = 400
#     iteration = 0
#     error = 1.0
#     dT_last = 0.0
#     while (error > 1.0e-2):
# #        print(iteration)
# #        print("--------")

#         R = np.zeros((ncells,),dtype=np.float64)
#         A = np.zeros((3,ncells),dtype=np.float64)

#         R_bc = np.zeros((4,),dtype=np.float64)
#         A_bc = np.zeros((3,4),dtype=np.float64)

#         R[1:-1] = d * (T[:-2] - 2.0 * T[1:-1] + T[2:])
#         A[0,1:] = c
#         A[1,1:-1] = b
#         A[2,:-1] = a

#         A[1,0] = 1.0
#         A[1,-1] = 1.0

#         # Left
#         if bc[0] == 'isothermal':
#             A[0,1] = 0.0 # 1.0
#             R[0] = 0.0 #0.5 * (dTg[1] + dTg[0])
#             R[0] = dT[0]

#         elif bc[0] == 'radiative_equilibrium':
#             #Twall = 0.5 * (T[0] + T[1])
#             #A[0,1] = 1.0 - 2.0 * dx * sigma * emissivity * Twall**3 / alpha
#             #A[1,1] = -1.0 - 2.0 * dx * sigma * emissivity * Twall**3 / alpha
#             #R[0] = A[0,1] * dTg[0] + A[1,1] * dTg[1]
#             #R[0] = A[1,1] * dTg[1]
#             A[0,1] = 0.0
#             R[0] = dT[0]

#         elif bc[0] == 'adiabatic':
#             A[0,1] = 0.0
#             R[0] = dT[0]

#         # Right
#         if bc[1] == 'adiabatic':
#             #A[2,-2] = -1.0
#             #R[-1] = -dTg[3] + dTg[2]

#             A[2,-2] = 0.0
#             R[-1] = dT[-1]

# #        print("A", A)
# #        print("R", R)

#         dT = solve_banded(form, A, R)

# #        print("dT", dT)
# #        print("-")

#         A_bc[1,:] = 1.0
#         # Left
#         if bc[0] == 'isothermal':
#             A_bc[2,0] = 1.0
#             R_bc[0] = dT[1]
#             R_bc[1] = 2.0 * bc[2] - T[0] - T[1]

#         elif bc[0] == 'radiative_equilibrium':
#             Twall = 0.5 * (T[0] + T[1])
#             A_bc[2,0] = -1.0 - 2.0 * dx * sigma * emissivity * Twall**3 / alpha
#             A_bc[1,1] = 1.0 - 2.0 * dx * sigma * emissivity * Twall**3 / alpha
#             R_bc[0] = dT[1]
#             R_bc[1] = dx * sigma * emissivity * Twall**4 / alpha + T[1] - T[0]

#             # print(A_bc[2,0], A_bc[1,1], R_bc[1])

#         elif bc[0] == 'adiabatic':
#             A_bc[2,0] = -1.0
#             R_bc[0] = dT[1]
#             R_bc[1] = T[1] - T[0]

#         else:
#             sys.exit("Bad BC")

#         # Right
#         if bc[1] == 'adiabatic':
#             A_bc[2,-2] = -1.0
#             R_bc[2] = dT[-2]
#             R_bc[3] = T[-2] - T[-1]

#         elif bc[1] == 'isothermal':
#             A_bc[2,-2] = 1.0
#             R_bc[2] = dT[-2]
#             R_bc[3] = 2.0 * bc[3] - T[-1] - T[-2]

# #        print("A_bc", A_bc)
# #        print("R_bc", R_bc)
#         dTg = solve_banded(form, A_bc, R_bc)
# #        print("dTg", dTg)

#         dT[0] = dTg[1]
#         dT[-1] = dTg[3]

#         iteration += 1
#         error = np.abs(dT_last - dT[0])
#         dT_last = dT[0]

#         # print(iteration, dT[0], error)

#         if (iteration > max_iterations):
#             break

#     # print("   ")

#     return T + dT


def bc_relax_solve(dx, dt, alpha, T, bc=('isothermal','isothermal',1000.0,500.0), emissivity=1.0):
    """
    In this formulation, we construct a cell-centerd discretization of the one-dimensional
    heat conduction equation with constant thermal conductivity from the linearized form of
    the heat conduction equations.. The boundary conditions are implemented using ghost
    cells, but are not imbedded into the system and instead are relaxed into the system
    using an iterative approach. This formulation can handle both Dirchlet and von Neumann
    boundary conditions.

    Args:
        dx (float) :
        dt (float) :
        alpha (float) :
        T (numpy-array) :

    """
    nsize = np.shape(T)[0]
    ncells = nsize - 2

    R_in = np.zeros((ncells,),dtype=np.float64)
    A_in = np.zeros((3,ncells),dtype=np.float64)
    dT_in = np.zeros((ncells,),dtype=np.float64)

    R_bc = np.zeros((2,),dtype=np.float64)
    A_bc = np.zeros((3,2),dtype=np.float64)
    dT_bc = np.zeros((2,),dtype=np.float64)

    # (1,1) is for one lower and upper diagonal, i.e., a tridiagonal matrix
    form = (1,1)

    #T[0] = 2.0 * bc[2] - T[1]
    #T[-1] = 2.0 * bc[3] - T[-2]

    # Coefficients
    d = alpha * dt / dx**2
    a = -d
    b = 1.0 + 2.0 * d
    c = -d

    A_in[0,1:] = c
    A_in[1] = b
    A_in[2,:-1] = a
    R_in = d * (T[:-2] - 2 * T[1:-1] + T[2:])

    iteration = 40
    while iteration > 0:

        R_in = d * (T[:-2] - 2 * T[1:-1] + T[2:])

        if bc[0] == 'isothermal':
            #R_in[0] = R_in[0] - a * dT_bc[0]
            R_in[0] = R_in[0] - a * (2.0 * dT_bc[0] - dT_in[0])

        if bc[1] == 'isothermal':
            #R_in[-1] = R_in[-1] - c * dT_bc[1]
            R_in[-1] = R_in[-1] - c * (2.0 * dT_bc[1] - dT_in[-1])

        if bc[0] == 'radiative_equilibrium':
            R_in[0] = R_in[0] - a * dT_bc[0]
            #R_in[0] = R_in[0] - a * (2.0 * dT_bc[0] - dT_in[0])

        if bc[1] == 'radiative_equilibrium':
            R_in[-1] = R_in[-1] - c * (2.0 * dT_bc[1] - dT_in[-1])

        dT_in = solve_banded(form, A_in, R_in)

        if bc[0] == 'isothermal':
            dT_bc[0] = bc[2] - 0.5 * (T[0] + T[1])

        if bc[1] == 'isothermal':
            dT_bc[1] = bc[3] - 0.5 * (T[-1] + T[-2])

        if bc[0] == 'radiative_equilibrium':
            R_bc = alpha * (T[1] - bc[2]) - dx * sigma * emissivity * bc[2]**4
            B_bc = -alpha + 2.0 * dx * sigma * emissivity * bc[2]**3
            A_bc = alpha + 2.0 * dx * sigma * emissivity * bc[2]**3
            dT_bc[0] = (R_bc - B_bc * dT_in[0]) / A_bc

        iteration -= 1

    T[1:-1] = T[1:-1] + dT_in
    T[0] = T[0] + 2.0 * dT_bc[0] - dT_in[0]
    T[-1] = T[-1] + 2.0 * dT_bc[1] -dT_in[-1]
    return T

def bc_linearized_solve(dx, dt, alpha, T, bc=('isothermal','isothermal',1000.0,500.0),
                        emissivity=1.0):
    """
    In this formulation, we construct a cell-centerd discretization of the one-dimensional
    heat conduction equation with constant thermal conductivity from the linearized form of
    the heat conduction equations.. The boundary conditions are implemented using ghost
    cells, which are imbedded into the system. This formulation can handle both Dirchlet
    and von Neumann boundary conditions.

    Args:
        dx (float) :
        dt (float) :
        alpha (float) :
        T (numpy-array) :

    """
    ncells = np.shape(T)[0]

    # (1,1) is for one lower and upper diagonal, i.e., a tridiagonal matrix
    form = (1,1)

    # Coefficients
    d = alpha * dt / dx**2
    a = -d
    b = 1.0 + 2.0 * d
    c = -d

    dT = np.zeros((ncells,),dtype=np.float64)
    dTg = np.zeros((4,),dtype=np.float64)

    iteration = 20
    while iteration > 0:
        # print(iteration)
        # print("--------")

        R = np.zeros((ncells,),dtype=np.float64)
        A = np.zeros((3,ncells),dtype=np.float64)

        R_bc = np.zeros((4,),dtype=np.float64)
        A_bc = np.zeros((3,4),dtype=np.float64)

        #T[0] = 2.0 * bc[2] - T[1]
        #T[-1] = T[-2]

        R[1:-1] = d * (T[:-2] - 2.0 * T[1:-1] + T[2:])
        A[0,1:] = c
        A[1,1:-1] = b
        A[2,:-1] = a

        A[1,0] = 1.0
        A[1,-1] = 1.0

        # Left
        if bc[0] == 'isothermal':
            A[0,1] = 1.0
            R[0] = 2.0 * bc[2] - T[0] - T[1] + dTg[1]

        # Right
        if bc[1] == 'adiabatic':
            A[2,-2] = -1.0
            R[-1] = T[-1] - T[-2] + dTg[3]

        # print("A", A)
        # print("R", R)

        dT = solve_banded(form, A, R)

        # print("dT", dT)
        # print("-")

        A_bc[1,:] = 1.0
        # Left
        if bc[0] == 'isothermal':
            A_bc[2,0] = 1.0
            R_bc[0] = dT[1]
            #R_bc[1] = 2.0 * bc[2] - T[0] - T[1]

        # Right
        if bc[1] == 'adiabatic':
            A_bc[2,-2] = -1.0
            #R_bc[2] = dT[-2]

        # print("A_bc", A_bc)
        # print("R_bc", R_bc)
        dTg = solve_banded(form, A_bc, R_bc)
        # print("dTg", dTg)

        iteration -= 1

    # print("   ")

    return T + dT


def main(num_samples, vary_left_iso_bc = False, vary_diffusivity = False):

    # Main program for a one-dimensional heat conduction solver
    dx = 0.1
    dt = 5.0e-2
    alpha = 1.0

    ncells = 10
    nghosts = 2
    nsize = ncells + nghosts

    num_time_steps = 10

    T_fd = np.zeros((ncells,),dtype=np.float64)
    # T_fv = np.zeros((nsize,),dtype=np.float64)

    temp_left_bc = 1000
    bc=('isothermal', 'isothermal', temp_left_bc, 500.0)

    T_storage = np.zeros((ncells - 2, num_time_steps, num_samples), dtype = np.float64)

    np.random.seed(0)

    labels = np.zeros((num_samples, 2))

    #alpha low and high bounds for uniform random distirution

    alpha_low = 0
    alpha_high = 10;
    if vary_diffusivity == True:
        alpha = np.random.uniform(alpha_low, alpha_high, num_samples)

    labels[:,0] = alpha

    #isothermal temp left low and high bounds for uniform random distirution
    temp_left_bc_low = 500
    temp_left_bc_high = 3000
    if vary_left_iso_bc:
        temp_left_bc = np.random.uniform(temp_left_bc_low, temp_left_bc_high, num_samples)

    labels[:,1] = temp_left_bc

    #generate labeled data. labels are alpha (diffusivity)
    for sample in range(num_samples):
        T_fd = np.zeros((ncells,),dtype=np.float64)
        bc=('isothermal', 'isothermal', labels[sample,1], 500.0)

        for time in range(num_time_steps):
            T_fd = finite_difference_solve(dx, dt, labels[sample,0], T_fd, bc=bc)
            T_storage[:,time,sample] = T_fd[1:-1]   

    return T_storage, labels

#
    # T_fv[:] = 300.0
    # T_fv = finite_volume_solve(dx, dt, alpha, T_fv, bc=bc)
    # print(T_fv)
#
    # bc=('isothermal', 'adiabatic', 1000.0, 300.0)
    # bc=('radiative_equilibrium', 'adiabatic', 1000.0, 300.0)

    # T_fv[:] = 300.0
    # T_fv = bc_linearized_solve(dx, dt, alpha, T_fv, bc=bc)
    # print(T_fv)
    # T_fv = bc_linearized_solve(dx, dt, alpha, T_fv, bc=bc)
    # print(T_fv)
    # T_fv = bc_linearized_solve(dx, dt, alpha, T_fv, bc=bc)
    # print(T_fv)
    # T_fv = bc_linearized_solve(dx, dt, alpha, T_fv, bc=bc)
    # print(T_fv)        
    # T_fv[:] = 300.0
    # T_fv = bc_relax_solve(dx, dt, alpha, T_fv, bc=bc)
    # print(T_fv)

    #T_node = np.zeros_like(T_fd)
    #for i in range(1,nsize-1):
    #    T_node[i-1] = 0.5 * (T_fv[i-1] + T_fv[i])

    #T_wall_1 = 650.0
    #T_wall_2 = 400.0
    #emissivity = 0.0

#    bc=('radiative_equilibrium', 'isothermal', T_wall_1, T_wall_2)
#    bc=('isothermal', 'isothermal', T_wall_1, T_wall_2)

#    T_fv[:] = 300.0
#    T_fv = bc_relax_solve(dx, dt, alpha, T_fv, bc=bc, emissivity=emissivity)
#    print(T_fv)

