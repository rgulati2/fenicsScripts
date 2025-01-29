from dolfin import *
import numpy as np

# Define temporal parameters
t = 0.0
T = 0.05
num_steps = 100
dt = T / num_steps
dim = 2

# Define mesh
if dim == 2:
    nx, ny = 127, 127
    mesh = UnitSquareMesh(nx, ny)
elif dim == 3:
    nx, ny, nz = 50, 50, 10
    mesh = BoxMesh(Point(0, 0, 0), Point(1, 1, 0.2), nx, ny, nz)

V = FunctionSpace(mesh, "CG", 1)

# Initial condition using UserExpression
class InitialCondition(UserExpression):
    def __init__(self, a=5, **kwargs):
        super().__init__(**kwargs)
        self.a = a
    def eval(self, value, x):
        value[0] = self.a * np.exp(-self.a * ((x[0]-0.5)**2 + (x[1]-0.5)**2))
    def value_shape(self):
        return ()

# Create solution function and interpolate initial condition
uh = Function(V, name="uh")  # <-- Explicitly name the function
uh.interpolate(InitialCondition(degree=2))

# Boundary condition (Dirichlet)
bc = DirichletBC(V, Constant(1.0), "on_boundary")

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
x = SpatialCoordinate(mesh)
k = sin(x[0]) * cos(x[1])

a = u * v * dx + dt * k * inner(grad(u), grad(v)) * dx
L = uh * v * dx  # <-- Use uh as previous time step

# Assemble the matrix once and apply BCs
A = assemble(a)
bc.apply(A)

# Setup XDMF file for output
xdmf = XDMFFile("heat_2d_hc0.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.write(uh, t)  # Write initial condition

# Time-stepping loop
u_n = Function(V)  # Helper function to store previous solution
u_n.assign(uh)

for i in range(num_steps):
    t += dt

    # Assemble the RHS vector
    b = assemble(L)

    # Apply BCs to the RHS vector
    bc.apply(b)

    # Solve the linear system using LU decomposition
    solve(A, uh.vector(), b, "lu")

    # Write solution to file (same Function object every time)
    xdmf.write(uh, t)

    # Update previous solution
    u_n.assign(uh)

xdmf.close()
