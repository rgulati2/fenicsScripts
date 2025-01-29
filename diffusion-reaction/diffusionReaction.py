from fenics import *
import numpy as np

# Parameters
k = Constant(5e-3)
Du = Constant(1e-3)
Dv = Constant(5e-3)
Nx, Ny = 127, 127
mesh = RectangleMesh(Point(-1, -1), Point(1, 1), Nx, Ny)

# Time parameters
Nt = 101
T = 50.0  # Total simulation time
dt = T / (Nt - 1)  # Time step

# Vector-valued function space for u and v (2D vector)
P1 = FiniteElement('P', mesh.ufl_cell(), 1)
element = VectorElement('P', mesh.ufl_cell(), 1, dim=2)  # 2D vector
V = FunctionSpace(mesh, element)

# Test functions
w = TestFunction(V)

# Solution functions
uh = Function(V)  # Current solution (2D vector)
uh_n = Function(V)  # Previous solution (2D vector)

# Set the name of the function for visualization in ParaView
uh.rename("uh", "uh")
uh_n.rename("uh", "uh")

# Split functions into components
u, v = split(uh)  # u and v are the 2 components of the vector uh
u_n, v_n = split(uh_n)

# Variational form
F = (
    ((u - u_n) / dt) * w[0] * dx
    + Du * inner(grad(u), grad(w[0])) * dx
    - (u - u**3 - k - v) * w[0] * dx
    + ((v - v_n) / dt) * w[1] * dx
    + Dv * inner(grad(v), grad(w[1])) * dx
    - (u - v) * w[1] * dx
)

# Jacobian and solver setup
J = derivative(F, uh)
problem = NonlinearVariationalProblem(F, uh, [], J)
solver = NonlinearVariationalSolver(problem)
prm = solver.parameters['newton_solver']
prm['absolute_tolerance'] = 1e-8
prm['relative_tolerance'] = 1e-8
prm['maximum_iterations'] = 50

# Initial conditions (random noise for u and v)
uh_n.vector()[:] = np.random.normal(0, 1.0, uh_n.vector().size())

# Create a single XDMF file for both u and v
#xdmf_file = XDMFFile("fitzhugh_nagumo.xdmf")
xdmf_file = XDMFFile("elasticity_2d_e0.xdmf")
xdmf_file.parameters["flush_output"] = True
xdmf_file.parameters["functions_share_mesh"] = True  # Ensure u and v share the same mesh

# Save initial condition
xdmf_file.write(uh_n, 0.0)

# Time stepping
t = 0.0
for n in range(Nt - 1):
    t += dt
    print(f"Time step {n+1}/{Nt-1}, t = {t:.2f}")

    # Solve the system
    uh.assign(uh_n)
    solver.solve()
    uh_n.assign(uh)

    # Save uh to the XDMF file
    xdmf_file.write(uh_n, t)

xdmf_file.close()
