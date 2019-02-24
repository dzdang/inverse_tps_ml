import numpy as np
from scipy.linalg import solve_banded

#crank nicholson
def finite_difference_solve(dx, dt, alpha, T, bc=('isothermal','isothermal',1000.0,500.0)):
    """
    In this formulation, we construct a node-based discretization of the one-dimensional
    heat conduction equation with constant thermal conductivity. The boundary condition
    is embedded directly into the solver and only works for Dirchlet boundaries.

    Args:
        dx (float) : The discretization size of the one-dimensional domain.
        dt (float) : The time step.
        alpha (float) : The thermal diffusivity of the material.
        T (numpy-array) : The initial temperature at each point in the simulation domain.
        bc (tuple) : The boundary conditions at the left and right ends of the domain.

    """
    # We can find the analytical solution to the heat conduction equation by solving the
    # linear system, Ax = R, where A is a sparse tri-diagonal matrix and R is the residual.

    # Get the number of points, ncells, by inspecting the size of the initial
    # temperature array. We then create two numPy arrays. R is the residual vector, and
    # A is a sparse banded matrix containing the coefficients for the linear system.
    #
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

def main():
    """Main program for a one-dimensional heat conduction solver"""

    # Define the problem
    ncells = 100
    dx = 0.1
    dt = 5.0
    alpha = 1.0

    # Define the initial temperature in the domain using a numPy array.
    T_init = 300.0
    T = T_init * np.ones((ncells,),dtype=np.float64)

    # Set the boundary conditions
    bc=('isothermal', 'isothermal', 1000.0, 500.0)

    # Solve for the temperature
    T = finite_difference_solve(dx, dt, alpha, T, bc=bc)

    # Find where the temperature is between 600 and 650.
    condition = (T > 600.0) & (T < 650)

    # When then np.where command is only given a condition, it returns a
    # tuple of all the indices that meet the condition. Since we only have
    # one condition, we want the first element in this tuple. We retrieve
    # this before setting the value equal to indices, which is way the [0]
    # is at the end of the np.where command.
    indices = np.where(condition)[0]

    # Let's store our values in a list. We will initialize an empty list
    # and then append values to it.
    T_range = []
    for i in indices:
        T_range.append(T[i])
    print("Here are our values between 600 and 650:")
    print(T_range,"\n")

    # Let's say we wanted to find the values between two ranges, 550-600 and 700-750,
    # then could do the following
    condition = []
    condition.append((T > 550.0) & (T < 600))
    condition.append((T > 700.0) & (T < 750))
    indices = np.where(condition)

    # indices is now a tuple of size 2, since we have two conditionals. Let's store
    # each of these ranges in a list as well, but I'll use a simplier expression for
    # generating the lists. The first array of the indices tuple tells us whether the
    # value the second array of the indices tuple met the first or second conditional,
    # so the first array contains 0 or 1 values and the second array contains our
    # temperatures.
    T_range = list(T[i] for i in indices[1])
    print("Here are our values in the joint regions [550,600] U [700,750]:")
    print(T_range, "\n")

    # Find the temperature closest to 350.
    T_target = 450.0

    # Let's assume we don't know anything the order. So first we need to sort first.
    T_sorted = np.sort(T)

    # Since the array is sorted, we know the first element of T_sorted - T_target
    # greater than zero will be our answer.
    i = np.where((T_sorted - T_target >= 0))[0][0]
    print("This is the value closest to 450 K")
    print(T[i])

if __name__ == "__main__":
    main()


