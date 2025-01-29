import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc, LinearProblem

# Define temporal parameters
t = 0  # Start time
T = 1.0  # Final time
num_steps = 100
dt = T / num_steps  # time step size
dim = 2

# Define mesh
if dim == 2:
        nx, ny = 127, 127
        #domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])],[nx, ny], mesh.CellType.quadrilateral)
        #domain = mesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0], [1.0, 1.0]], [nx, ny],mesh.CellType.quadrilateral)
        domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.quadrilateral)
elif dim == 3:
    nx, ny, nz = 50, 50, 10
    domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([1, 1, 0.2])],[nx, ny, nz],mesh.CellType.hexahedron)

V = fem.functionspace(domain, ("Lagrange", 1))

# Create initial condition
def initial_condition(x, a=5):
    return a*np.exp(-a * ((x[0]-0.5)**2 + (x[1]-0.5)**2))
    #return a*(x[0]+x[1])
    #return np.ones(x.shape[1])*a


u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

# Create boundary condition
fdim = domain.topology.dim - 1
#boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1)),np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))))
#boundary_facets_z = mesh.locate_entities_boundary(domain, fdim, lambda x: np.logical_or(np.logical_or(np.isclose(x[2], 0), np.isclose(x[2], 0.1))))

bc = fem.dirichletbc(PETSc.ScalarType(1.0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)
#bc = fem.dirichletbc(x[0], fem.locate_dofs_topological(V, fdim, boundary_facets), V)
#bc2 = fem.dirichletbc(PETSc.ScalarType(.0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

xdmf = io.XDMFFile(domain.comm, "heat_2d_hc13.xdmf", "w")
xdmf.write_mesh(domain)

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
xdmf.write_function(uh, t)

# Write the initial condition to file
#xdmf.write_function(u_n, t)
x = ufl.SpatialCoordinate(domain)
k = ufl.sin(x[0]) * ufl.cos(x[1])  # Heat conduction coefficient

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
#f = fem.Constant(domain, PETSc.ScalarType(0))
a = u * v * ufl.dx + dt * k * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n) * v * ufl.dx


problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
#solver.convergence_criterion = "incremental"
#solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-7
#bilinear_form = fem.form(a)
#linear_form = fem.form(L)


#A = assemble_matrix(bilinear_form, bcs=[bc])
#A.assemble()
#b = create_vector(linear_form)


#solver = PETSc.KSP().create(domain.comm)
#solver.setOperators(A)
#solver.setType(PETSc.KSP.Type.PREONLY)
#solver.getPC().setType(PETSc.PC.Type.LU)
 
                            
for i in range(num_steps):
    t += dt

    # Update the right hand side reusing the initial vector
    #with b.localForm() as loc_b:
    #    loc_b.set(0)
    #assemble_vector(b, linear_form)

    # Apply Dirichlet boundary condition to the vector
    #apply_lifting(b, [bilinear_form], [[bc]])
    #b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    #set_bc(b, [bc])

    # Solve linear problem
    #solver.solve(b, uh.x.petsc_vec)
    #uh.x.scatter_forward()
    uh = problem.solve()
    uh.name = "uh"
    u_n.x.array[:] = uh.x.array # Update solution at previous time step (u_n)
    xdmf.write_function(uh, t) # Write solution to file
xdmf.close()







