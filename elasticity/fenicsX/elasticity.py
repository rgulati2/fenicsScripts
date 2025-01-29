#Elasticity works
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, io
from dolfinx.fem import Function, FunctionSpace, dirichletbc
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem.petsc import LinearProblem
from dolfinx import log, default_scalar_type


# Define temporal parameters (useful for time-stepping problems)
T = 1.0  # Final time
num_steps = 10
dt = T / num_steps  # Time step size
dim = 2

# Define mesh
if dim == 2:
        nx, ny = 50, 50
        domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.quadrilateral)
elif dim ==3:
    nx, ny, nz = 50, 50, 10
    domain = mesh.create_box(MPI.COMM_WORLD,
                         [np.array([0, 0, 0]), np.array([1, 1, 0.2])],
                         [nx, ny, nz],
                         mesh.CellType.hexahedron)
#V = fem.functionspace(domain, ("Lagrange", 1), domain.geometry.dim)
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))



# Define initial condition
def initial_condition(x):
    return np.zeros((domain.geometry.dim, x.shape[1]))

u_n = Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

# Define boundary conditions
fdim = domain.topology.dim - 1

def clamped_boundary_1(x):
    return np.isclose(x[0], 0.0)

def clamped_boundary_2(x):
    return np.isclose(x[0], 1.0)

# Locate boundary facets
clamped_boundary1_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary_1)
clamped_boundary2_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary_2)

# Create boundary conditions

if dim == 2:
    u1_D = np.array([0.0, 0.0], dtype=PETSc.ScalarType)
    u2_D = np.array([0.05, 0.0], dtype=PETSc.ScalarType)
elif dim ==3:
    u1_D = np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType)
    u2_D = np.array([0.05, 0.0, 0.0], dtype=PETSc.ScalarType)

bc1 = dirichletbc(u1_D, fem.locate_dofs_topological(V, fdim, clamped_boundary1_facets), V)
bc2 = dirichletbc(u2_D, fem.locate_dofs_topological(V, fdim, clamped_boundary2_facets), V)

# Create XDMF file for output
xdmf = io.XDMFFile(domain.comm, "elasticity_2d_e0.xdmf", "w")
xdmf.write_mesh(domain)

# Material properties
E = 1.0  # Young's modulus
nu = 0.3  # Poisson's ratio
lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))

# Define strain and stress
def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

# Define the variational problem
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
if dim ==2:
        f = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))
elif dim ==3:
    f = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, 0.0)))
    
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx

# Solve the linear elasticity problem
problem = LinearProblem(a, L, bcs=[bc1, bc2], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# Initial solution for visualization
uh = Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
xdmf.write_function(uh, 0.0)

# Time-stepping loop
for i in range(num_steps):
    # Update the displacement for the right boundary (x = 1)
    displacement_x = 0.05 * (i + 1)
    u2_D = np.array([displacement_x, 0.0, 0.0], dtype=PETSc.ScalarType)
    
    # Update the boundary condition for the right boundary
    bc2 = dirichletbc(u2_D, fem.locate_dofs_topological(V, fdim, clamped_boundary2_facets), V)
    
    # Solve the problem with the updated boundary conditions
    problem = LinearProblem(a, L, bcs=[bc1, bc2], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    # Solve the problem
    uh = problem.solve()
    uh.name = "uh"
    
    # Update previous solution
    u_n.x.array[:] = uh.x.array[:]
    
    # Write solution to XDMF file
    xdmf.write_function(uh, (i + 1) * dt)

xdmf.close()


