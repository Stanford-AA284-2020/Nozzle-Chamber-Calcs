#function var = solveLOCP(var_k)
n = 6 # get the state dimension
m = 3 # get the control dimension

#var_k = state
x_k = np.zeros([N+1,1])
y_k = np.zeros([N+1,1])
z_k = np.zeros([N+1,1])
V_k = np.zeros([N+1,1])
gamma_k = np.zeros([N+1,1])
psi_k = np.zeros([N+1,1])
# controls
u_k1 = np.zeros([N,1]) #recovering x_k,y_k, and u_k from var_k
u_k2 = np.zeros([N,1])
u_k3 = np.zeros([N,1])

for i in range(1,N+1):
    x_k[i] = var_k[9*i-9]
    y_k[i] = var_k[9*i-8]
    z_k[i] = var_k[9*i-7]
    V_k[i] = var_k[9*i-6]
    gamma[i] = var_k[9*i-5]
    psi[i] = var_k[9*i-4]

    if i <= N:
        u_k1[i] = var_k[9*i-3]
        u_k2[i] = var_k[9*i-2]
        u_k3[i] = var_k[9*i-1]

        
# SET UP INDEXING FUNCTIONS
def x_start(i):
global n, m
     return (i-1)*(n+m) + 1 # get the first entry of x_i in the vector z
def x_end(i):
global n, m
     return i*(n+m) # get the last entry of x_i in the vector z

n_constraints = n*N
C = np.zeros([n_constraints, (n+m)*n_constraints]) #% Cvx accepts equality constraints in the form of C*var=d
                         # and bounds of type lb <= var <= ub.
                         # C will contain all the linearized dynamical
                         # constraints
                         
d = np.zeros([n_constraints,1])
lb = np.zeros([(n+m)*n_constraints,1])
ub = np.zeros([(n+m)*n_constraints,1])
"""
C[n*N+1,1] = 1.
C[n*N+2,2] = 1. # Initial conditions on the state 
C[n*N+3,3] = 1.
C[n*N+4,4] = 1.
C[n*N+5,5] = 1.
C[n*N+6,6] = 1.
d[n*N+1] = 0.
d[n*N+2] = 0.
d[n*N+3] = 0.
d[n*N+4] = 0.
d[n*N+5] = 0.
d[n*N+6] = 0.

C[n*N+7,m*N+1] = 1. 
C[n*N+8,m*N+2] = 1. # Final conditions on the state 
d[n*N+3] = M
d[n*N+4] = l
"""
h = (1.0*T/(1.0*N)
     
##
for i in range(1,N): # Dynamic constraints
    x_ref = z_old(x_start(i): x_end(i));
    u_ref = z_old(u_start(i):u_end(i));
    [A, B, ~] = linearize_dynamics(x_ref, u_ref, dt);
    C[x_start(i):x_end(i), x_start(i + 1):x_end(i + 1)] = np.eye(n);
    C[x_start(i):x_end(i), x_start(i):x_end(i)] = - A;
    C[x_start(i):x_end(i), u_start(i):u_end(i)] = - B;
    f_old = f(x_ref, u_ref, dt) - x_ref;
    d(x_start(i):x_end(i)) = f_old - (A - eye(n)) * x_ref - B * u_ref;
    
     
for i in range(1,N): # Dynamic constraints
    
    [noteTerm_k, DxyfDyn_k, DufDyn_k] = fDynLin((i-1)*h, x_k, y_k, u_k);
    j_iter = 0
    for j in range(x_start(i):x_end(i)+1):
        
        d[j] = h * noteTerm_k[j_iter] 
        
        C[j,j] = -1. - h*DxyfDyn_k[j_iter,0]
        C[j,j+1] = - h*DxyfDyn_k[j_iter,1]
        C[j,j+2] = - h*DxyfDyn_k[j_iter,2]
        C[j,j+3] = - h*DxyfDyn_k[j_iter,3]
        C[j,j+4] = - h*DxyfDyn_k[j_iter,4] # Filling the matric constraint C with linearized dynamical constraints
        C[j,j+5] = - h*DxyfDyn_k[j_iter,5]

        C[j,j+7] = -h*DufDyn_k[0]
        C[j,j+8] = -h*DufDyn_k[1]
        C[j,j+9] = -h*DufDyn_k[2] 
        j_iter = j_iter + 1

for i in range(1,N+1):
    ub[x_end(i)-5] = x_k[i] + epsTrust # Lower and upper bounds will contain constant
                                # "trust region" constraints. There are:
                                # |x(t)-x_k(t)| <= epsTrust
    ub[x_end(i)-4] = y_k[i] + epsTrust
    ub[x_end(i)-3] = z_k[i] + epsTrust 
    ub[x_end(i)-2] = V_k[i] + epsTrust
    ub[x_end(i)-1] = gamma_k[i] + epsTrust
    ub[x_end(i)] = psi_k[i] + epsTrust
    if i <= N
        ub[x_end(i)+1] = 10
        ub[x_end(i)+2] = 10
        ub[x_end(i)+3] = 10

    lb[x_end(i)-5] = x_k[i] - epsTrust 
    lb[x_end(i)-4] = y_k[i] - epsTrust
    lb[x_end(i)-3] = z_k[i] - epsTrust 
    lb[x_end(i)-2] = V_k[i] - epsTrust
    lb[x_end(i)-1] = gamma_k[i] - epsTrust
    lb[x_end(i)] = psi_k[i] - epsTrust
    if i <= N
        lb[x_end(i)+1] = 10
        lb[x_end(i)+2] = 10
        lb[x_end(i)+3] = 10
    
'''
cvx_begin # Beginning of Cvx environment
# cvx_begin quiet; % This removes Cvx output

variable z(3*N+2) % Defining the main varibable

cost = 0.;

for i = 1:N % To increase the chance to find feasible solutions at each SCP iteration, we erather penalize cost/state-constraints
cost = cost + z(3*i)*z(3*i) + max(z(3*i)-uMax,0.) + max(-z(3*i-2),0) + max(-z(3*i-1),0) + max(z(3*i-2)-M,0.) + max(z(3*i-1)-l,0);
end

minimize( cost ) % defining and solving the problem
subject to 
C*z == d
lb <= z <= ub

cvx_end

var = z
'''