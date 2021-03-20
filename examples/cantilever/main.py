from dolfin import *
set_log_level(0) #https://fenicsproject.org/qa/810/how-to-disable-message-solving-linear-variational-problem/

mul = 5
L = 25.
H = 1.
Nx = 250 * mul
Ny = 10 * mul
mesh = RectangleMesh(Point(0., 0.), Point(L, H), Nx, Ny, "crossed")

def eps(v):
    return sym(grad(v))

E = Constant(1e5)
nu = Constant(0.3)
model = "plane_stress"

mu = E/2/(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)
if model == "plane_stress":
    lmbda = 2*mu*lmbda/(lmbda+2*mu)

def sigma(v):
    return lmbda*tr(eps(v))*Identity(2) + 2.0*mu*eps(v)

rho_g = 1e-3
f = Constant((0, -rho_g))

V = VectorFunctionSpace(mesh, 'Lagrange', degree=2)
u_sol = Function(V, name="Displacement")


u = TrialFunction(V)
v = TestFunction(V)
E = inner(sigma(u_sol), eps(u_sol))*dx - dot(f,u_sol)*ds()

equation = derivative(E, u_sol, v)
left = CompiledSubDomain("on_boundary && near(x[0], 0, tol)", tol=1e-14)

bc = DirichletBC(V, Constant((0.,0.)), left)

grad_equation = derivative(equation, u_sol, u)

problem_disp = NonlinearVariationalProblem(
                    equation, u_sol, bc, grad_equation)

solver_disp = NonlinearVariationalSolver(problem_disp)
prm_disp = solver_disp.parameters

solvers = (
    "bicgstab",
    "cg",
    "default",
    "gmres",
    "minres",
    "mumps",
    "petsc",
    "richardson",
    "superlu",
    "tfqmr",
    "umfpack",
)

solvers_direct = (
    "mumps",
    "superlu",
    "umfpack",
)

solvers_iterative = (
    "bicgstab",
    "cg",
    "gmres",
    "minres",
    "richardson",
)

preconditioners = (
    "amg",
    "default",
    "hypre_amg",
    "hypre_euclid",
    "hypre_parasails",
    "icc",
    "ilu",
    "jacobi",
    "none",
    "petsc_amg",
    "sor",
)
linesearch = ("basic", "bt", "cp", "l2", "nleqerr")
prm_disp["nonlinear_solver"] = "newton"
prm_disp["newton_solver"]["maximum_iterations"] = 100000
prm_disp["newton_solver"]["report"] = False
prm_disp["newton_solver"]["absolute_tolerance"] = 1e-5
prm_disp["newton_solver"]["relative_tolerance"] = 1e-7
prm_disp["newton_solver"]["linear_solver"] = "cg"
prm_disp["newton_solver"]["preconditioner"] = "default"

#prm_disp["newton_solver"]["lu_solver"]["report"] = True

prm_disp["newton_solver"]["krylov_solver"]["report"] = True
prm_disp["newton_solver"]["krylov_solver"]["error_on_nonconvergence"] = True
prm_disp["newton_solver"]["krylov_solver"]["absolute_tolerance"] = 1e-7
prm_disp["newton_solver"]["krylov_solver"]["relative_tolerance"] = 1e-5
prm_disp["newton_solver"]["krylov_solver"]["maximum_iterations"] = 100
prm_disp["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = True


solver_disp.solve()



print("Maximal deflection:", -u_sol(L,H/2.)[1])

