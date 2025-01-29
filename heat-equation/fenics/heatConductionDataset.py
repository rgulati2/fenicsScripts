from dolfin import *
import numpy as np

# Temporal parameters
T = 0.05
num_steps = 100
dt = T / num_steps

# Create mesh and function space once (2D)
mesh = UnitSquareMesh(127, 127)
V = FunctionSpace(mesh, "CG", 1)

class CustomInitialCondition(UserExpression):
    def __init__(self, a, expr_type, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.expr_type = expr_type
        
    def eval(self, value, x):
        if self.expr_type == 0:
            value[0] = self.a * np.exp(-self.a * ((x[0]-0.5)**2 + (x[1]-0.5)**2))
        elif self.expr_type == 1:
            value[0] = self.a * (x[0] + x[1])
        elif self.expr_type == 2:
            value[0] = self.a
        elif self.expr_type == 3:
            value[0] = self.a * x[0] * x[1]
        elif self.expr_type == 4:
            value[0] = self.a * x[0]
    
    def value_shape(self):
        return ()

# Main simulation loop
for A_idx in range(10):
    # Determine initial condition parameters
    expr_type = A_idx // 2
    a_val = 5 if (A_idx % 2 == 0) else 1
    
    for B_idx in range(10):
        # Create heat conductivity expression
        x = SpatialCoordinate(mesh)
        if B_idx == 0:
            k = sin(x[0]) * cos(x[1])
        elif B_idx == 1:
            k = sin(x[0]) * sin(x[1])
        elif B_idx == 2:
            k = cos(x[0]) * cos(x[1])
        elif B_idx == 3:
            k = cos(x[0]) * sin(x[1])
        elif B_idx == 4:
            k = x[0] + x[1]
        elif B_idx == 5:
            k = x[0] * x[1]
        elif B_idx == 6:
            k = Constant(1.0)
        elif B_idx == 7:
            k = Constant(3.0)
        elif B_idx == 8:
            k = Constant(5.0)
        elif B_idx == 9:
            k = Constant(7.0)

        for C_idx in range(10):
            # Calculate simulation number (1-1000)
            sim_number = A_idx * 100 + B_idx * 10 + C_idx + 1
            filename = f"heat_2d_hc{sim_number}.xdmf"
            
            # Set up initial condition
            uh = Function(V, name="uh")
            ic = CustomInitialCondition(a=a_val, expr_type=expr_type, degree=2)
            uh.interpolate(ic)

            # Set up boundary condition
            if C_idx < 4:
                if C_idx == 0:
                    bc_expr = Expression("x[0]*x[1]", degree=2)
                elif C_idx == 1:
                    bc_expr = Expression("x[0]+x[1]", degree=2)
                elif C_idx == 2:
                    bc_expr = Expression("x[0]", degree=2)
                elif C_idx == 3:
                    bc_expr = Expression("x[1]", degree=2)
                bc_value = Function(V)
                bc_value.interpolate(bc_expr)
                bc = DirichletBC(V, bc_value, "on_boundary")
            elif 4 <= C_idx <= 8:
                bc = DirichletBC(V, Constant(C_idx - 3), "on_boundary")
            else:  # C_idx == 9
                bc_expr = Expression("sin(x[0])*cos(x[1])", degree=2)
                bc_value = Function(V)
                bc_value.interpolate(bc_expr)
                bc = DirichletBC(V, bc_value, "on_boundary")

            # Define variational problem
            u = TrialFunction(V)
            v = TestFunction(V)
            a = u * v * dx + dt * inner(k * grad(u), grad(v)) * dx
            L = uh * v * dx

            # Assemble system
            A = assemble(a)
            bc.apply(A)

            # Setup output file
            xdmf = XDMFFile(filename)
            xdmf.parameters["flush_output"] = True
            xdmf.parameters["functions_share_mesh"] = True
            xdmf.write(uh, 0.0)

            # Time-stepping
            u_n = Function(V)
            u_n.assign(uh)
            t = 0.0

            for _ in range(num_steps):
                t += dt
                b = assemble(L)
                bc.apply(b)
                solve(A, uh.vector(), b, "lu")
                xdmf.write(uh, t)
                u_n.assign(uh)

            xdmf.close()
