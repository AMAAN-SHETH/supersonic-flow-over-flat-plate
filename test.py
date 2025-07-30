import numpy as np
import matplotlib.pyplot as plt

T_freestrem = 288.16
P_freestrem = 101325
R = 287
M_freestream = 4.0
gamma = 1.4
a_freestrem = np.sqrt(gamma * R * T_freestrem)
Pr_number = 0.71
mu_freestrem = 1.7894 * 10**-5
rho_freestrem = P_freestrem / (R * T_freestrem)
cv = R / (gamma - 1)
cp = gamma * cv
e_freestream = cv * T_freestrem
T_wall = T_freestrem

def DYN_VIS(T):
    mu = mu_freestrem * ((T / T_freestrem)**(3/2)) * ((T_freestrem + 110) / (T + 110))
    return mu
    
def THEMC(mu):
    k = mu * cp / Pr_number
    return k

def boundary_conditions(u_field, v_field, p_field, T_field, rho, e, T_case):
    
            # leading edge
            u_field[0, 0] = 0
            v_field[0, 0] = 0
            p_field[0,0] = P_freestrem
            T_field[0,0] = T_freestrem
            rho[0,0] = P_freestrem / (R * T_freestrem)
            e[0,0] = cv * T_freestrem

            #inflow boundary 
            u_field[1:, 0] =  a_freestrem * M_freestream
            v_field[1:, 0] =  0
            p_field[1:, 0] =  P_freestrem
            T_field[1:, 0] =  T_freestrem
            rho[1:, 0] = p_field[1:, 0] / (R * T_field[1:, 0])
            e[1:, 0] = cv * T_field[1:, 0]

            # upper boundary
            u_field[-1, :] = a_freestrem * M_freestream
            v_field[-1, :] = 0
            p_field[-1, :] = P_freestrem
            T_field[-1, :] = T_freestrem
            rho[-1, :] = p_field[-1, :] / (R * T_field[-1, :])
            e[-1, :] = cv * T_field[-1, :]

            # surface
            u_field[0, 1:] = 0
            v_field[0, 1:] = 0
            p_field[0, 1:] = 2 * p_field[1, 1:] - p_field[2, 1:]
            if T_case == "const":
                T_field[0, 1:] = T_wall
            elif T_case == "adia":
                T_field[0, 1:] = T_field[1, 1:]
            rho[0, 1:] = p_field[0, 1:] / (R * T_field[0, 1:])
            e[0, 1:] = cv * T_field[0, 1:]

            # outflow
            u_field[1:-1, -1] = 2 * u_field[1:-1, -2] - u_field[1:-1, -3]
            v_field[1:-1, -1] = 2 * v_field[1:-1, -2] - v_field[1:-1, -3]
            p_field[1:-1, -1] = 2 * p_field[1:-1, -2] - p_field[1:-1, -3]
            T_field[1:-1, -1] = 2 * T_field[1:-1, -2] - T_field[1:-1, -3]
            rho[1:-1, -1] = p_field[1:-1, -1] / (R * T_field[1:-1, -1])
            e[1:-1, -1] = cv * T_field[1:-1, -1]

            return u_field, v_field, p_field, T_field, rho, e

def inititialize(T_case):
    u = np.zeros((Nx, Ny))
    v = np.zeros_like(u)
    T = np.zeros_like(u)
    p = np.zeros_like(u)

    u[:, :] = a_freestrem * M_freestream
    T[:, :] = T_freestrem
    p = np.full((np.shape(u)), P_freestrem)
    e = cv * T
    rho = p / (R * T)

    u, v, p, T, rho, e = boundary_conditions(u_field=u, v_field=v, p_field=p, T_field=T, T_case=T_case, rho=rho, e=e)

    mu = DYN_VIS(T)
    k = THEMC(mu)
    rho = p / (R * T)
    e = cv * T

    return u, v, p, T, rho, e

Nx = 70
Ny = 70
Lx = 0.00001
Re_L = rho_freestrem * M_freestream * a_freestrem * Lx / mu_freestrem
Ly = 25 * Lx / np.sqrt(Re_L)
K = 0.6
Nt = 7000

dx = Lx / (Nx-1)
dy = Ly / (Ny-1)
print(dx, dy)

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)


def TAU_XY(case, u, v, T):
    if case == 1: # for dE_dx in predictor
        du_dy = np.zeros_like(u) # central difference
            
        du_dy[0, :] = (u[1, :] - u[0, :]) / dy # lower boundary 
        du_dy[-1, :] = (u[-1, :] - u[-2, :]) / dy # upper boundary
        du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)

        dv_dx = np.zeros_like(u) #reareward difference
        dv_dx[:, 0] = (v[:, 1] - v[:, 0]) / dx # left boundary
        dv_dx[:, 1:] = (v[:, 1:] - v[:, :-1]) / dx
        
    elif case == 2: # for dE_dx in corrector
        du_dy = np.zeros_like(u) #central difference
            
        du_dy[0, :] = (u[1, :] - u[0, :]) / dy # lower boundary
        du_dy[-1, :] = (u[-1, :] - u[-2, :]) / dy # upper boundary
        du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy) 

        dv_dx = np.zeros_like(u) #forward difference
        dv_dx[:, -1] = (v[:, -1] - v[:, -2]) / dx # right boundary
        dv_dx[:, :-1] = (v[:, 1:] - v[:, :-1]) / dx

    elif case == 3: #for dF_dy in predictor
        du_dy = np.zeros_like(u) #reareward difference

        du_dy[0, :] = (u[1, :] - u[0, :]) / dy # lower boundary 
        du_dy[1:, :] = (u[1:, :] - u[:-1, :]) / dy

        dv_dx = np.zeros_like(u) #central difference
        dv_dx[:, 0] = (v[:, 1] - v[:, 0]) / dx # left boundary
        dv_dx[:, -1] = (v[:, -1] - v[:, -2]) / dx # right boundary
        dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx)

    elif case == 4: # for dF_dy in corrector
        du_dy = np.zeros_like(u) #forward difference

        du_dy[-1, :] = (u[-1, :] - u[-2, :]) / dy # upper boundary
        du_dy[:-1, :] = (u[1:, :] - u[:-1, :]) / dy

        dv_dx = np.zeros_like(u) #central difference
        dv_dx[:, 0] = (v[:, 1] - v[:, 0]) / dx # left boundary
        dv_dx[:, -1] = (v[:, -1] - v[:, -2]) / dx # right boundary
        dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx)
        
    tau_xy = DYN_VIS(T) * (du_dy + dv_dx)
    return tau_xy

def TAU_XX(case, u, v, T):
    if case == 1: # for dE_dx in predictor
        du_dx = np.zeros_like(u) # reareward difference

        du_dx[:, 0] = (u[:, 1] - u[:, 0]) / dx # left boundary
        du_dx[:, 1:] = (u[:, 1:] - u[:, :-1]) / dx

        dv_dy = np.zeros_like(u) # central difference

        dv_dy[0, :] = (v[1, :] - v[0, :]) / dy # lower boundary
        dv_dy[-1, :] = (v[-1, :] - v[-2, :]) / dy # upper boundary
        dv_dy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dy)

    elif case == 2: # for dE_dx in corrector
        du_dx = np.zeros_like(u) # forward difference

        du_dx[:, -1] = (u[:, -1] - u[:, -2]) / dx # right boundary
        du_dx[:, :-1] = (u[:, 1:] - u[:, :-1]) / dx

        dv_dy = np.zeros_like(u) # central difference

        dv_dy[0, :] = (v[1, :] - v[0, :]) / dy # lower boundary
        dv_dy[-1, :] = (v[-1, :] - v[-2, :]) / dy # upper boundary
        dv_dy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dy)

    tau_xx = DYN_VIS(T) * ((4 * du_dx / 3) - (2 * dv_dy / 3))
    return tau_xx
    
def TAU_YY(case, u, v, T):
    if case == 1: # for dF_dy in predictor
        dv_dy = np.zeros_like(u)# rearward difference
        dv_dy[0, :] = (v[1, :] - v[0, :]) / dy # lower boundary
        dv_dy[1:, :] = (v[1:, :] - v[:-1, :]) / dy
        
        du_dx = np.zeros_like(u) # central difference
        du_dx[:, 0] = (u[:, 1] - u[:, 0]) / dx # left boundary
        du_dx[:, -1] = (u[:, -1] - u[:, -2]) / dx # right boundary
        du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    
    elif case == 2: # for dF_dy in corrector
        dv_dy = np.zeros_like(u) # forward difference       
        dv_dy[-1, :] = (v[-1, :] - v[-2, :]) / dy # upper boundary
        dv_dy[:-1, :] = (v[1:, :] - v[:-1, :]) / dy

        du_dx = np.zeros_like(u) # central difference
        du_dx[:, 0] = (u[:, 1] - u[:, 0]) / dx # left boundary
        du_dx[:, -1] = (u[:, -1] - u[:, -2]) / dx # right boundary
        du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)  

    tau_yy = DYN_VIS(T) * ((4 * dv_dy / 3) - (2 * du_dx / 3))
    return tau_yy
    

def Q_X(case, T):
    if case == 1: # for dE_dx in predictor 
        dT_dx = np.zeros_like(T) # rearward difference
        dT_dx[:, 0] = (T[:, 1] - T[:, 0]) / dx # left boundary
        dT_dx[:, 1:] = (T[:, 1:] - T[:, :-1]) / dx

    elif case == 2: #for dE_dx in corrector
        dT_dx = np.zeros_like(T) #forward difference
        dT_dx[:, -1] = (T[:, -1] - T[:, -2]) / dx #right boundary
        dT_dx[:, :-1] = (T[:, 1:] - T[:, :-1]) / dx
        
    mu = DYN_VIS(T)
    q_x = -THEMC(mu) * (dT_dx)
    return q_x

def Q_Y(case, T):
    if case == 1: #for dF_dy in predictor
        dT_dy = np.zeros_like(T)#rearward difference
        dT_dy[0, :] = (T[1, :] - T[0, :]) / dy
        dT_dy[1:, :] = (T[1:, :] - T[:-1, :]) / dy
        
    elif case == 2: #for dF_dx in corrector
        dT_dy = np.zeros_like(T) #forward difference
        dT_dy[-1, :] = (T[-1, :] - T[-2, :]) / dy
        dT_dy[:-1, :] = (T[1:, :] - T[:-1, :]) / dy
        
    mu = DYN_VIS(T)
    q_y = -THEMC(mu) * (dT_dy)
    return q_y

    
def MAC(rho, u, v, e, T, p, dt, T_case):
    e = cv * T
    E_t = rho * (e + (((u**2 + v**2))/2))

    U1 = rho
    U2 = rho * u
    U3 = rho * v
    U5 = E_t

    E1 = rho * u
    E2 = (rho * u**2) + p - TAU_XX(case=1, u=u, v=v, T=T)
    E3 = (rho * u * v) - TAU_XY(case=1, u=u, v=v, T=T)

    print("tau_xx min/max:", np.min(TAU_XX(1, u, v, T)), np.max(TAU_XX(1, u, v, T)))
    print("tau_xy min/max:", np.min(TAU_XY(1, u, v, T)), np.max(TAU_XY(1, u, v, T)))
    print("q_x min/max:", np.min(Q_X(1, T)), np.max(Q_X(1, T)))


    E5 = ((E_t + p) * u) - ((u * TAU_XX(case=1, u=u, v=v, T=T)) - (v * TAU_XY(case=1, u=u, v=v, T=T)) + Q_X(case=1, T=T))

    F1 = rho * v
    F2 = (rho * u * v) - TAU_XY(case=3, u=u, v=v, T=T)
    F3 = (rho * v**2) + p - TAU_YY(case=1, u=u, v=v, T=T)
    F5 = ((E_t + p) * v) - (u * TAU_XY(case=3, u=u, v=v, T=T) - (v * TAU_YY(case=1, u=u, v=v, T=T)) + Q_Y(case=1, T=T)) 

    U1_bar = np.copy(U1)
    U2_bar = np.copy(U2)
    U3_bar = np.copy(U3)
    U5_bar = np.copy(U5)

    U1_bar[1:-1, 1:-1] = U1[1:-1, 1:-1] - (dt * (E1[1:-1, 2:] - E1[1:-1, 1:-1]) / dx) - (dt * (F1[2:, 1:-1] - F1[1:-1, 1:-1]) / dy)
    U2_bar[1:-1, 1:-1] = U2[1:-1, 1:-1] - (dt * (E2[1:-1, 2:] - E2[1:-1, 1:-1]) / dx) - (dt * (F2[2:, 1:-1] - F2[1:-1, 1:-1]) / dy)
    U3_bar[1:-1, 1:-1] = U3[1:-1, 1:-1] - (dt * (E3[1:-1, 2:] - E3[1:-1, 1:-1]) / dx) - (dt * (F3[2:, 1:-1] - F3[1:-1, 1:-1]) / dy)
    U5_bar[1:-1, 1:-1] = U5[1:-1, 1:-1] - (dt * (E5[1:-1, 2:] - E5[1:-1, 1:-1]) / dx) - (dt * (F5[2:, 1:-1] - F5[1:-1, 1:-1]) / dy)

    rho_bar = U1_bar
    u_bar = U2_bar / rho_bar
    v_bar = U3_bar / rho_bar
    e_bar = (U5_bar / rho_bar) - (((U2_bar / rho_bar)**2 + (U3_bar / rho_bar)**2) / 2)
    T_bar = e_bar / cv
    
    p_bar = rho_bar * R * T_bar
    print(T_bar)
    print(e_bar)
    u_bar, v_bar, p_bar, T_bar, rho_bar, e_bar = boundary_conditions(u_field=u_bar, v_field=v_bar, p_field=p_bar, T_field=T_bar, T_case=T_case, rho=rho_bar, e=e_bar)

    mu_bar = DYN_VIS(T_bar)
    k_bar = THEMC(mu_bar)

    a_bar = np.sqrt(gamma * R * T_bar)

    E_t_bar = rho_bar * (e_bar + (((u_bar**2 + v_bar**2))/2))

    E1_bar = rho_bar * u_bar
    E2_bar = (rho_bar * u_bar**2) + p_bar - TAU_XX(case=2, u=u_bar, v=v_bar, T=T_bar)
    E3_bar = (rho_bar * u_bar * v_bar) - TAU_XY(case=2, u=u_bar, v=v_bar, T=T_bar)
    E5_bar = ((E_t_bar + p_bar) * u_bar) - ((u_bar * TAU_XX(case=2, u=u_bar, v=v_bar, T=T_bar)) - (v_bar * TAU_XY(case=2, u=u_bar, v=v_bar, T=T_bar)) + Q_X(case=2, T=T_bar))

    F1_bar = rho_bar * v_bar
    F2_bar = (rho_bar * u_bar * v_bar) - TAU_XY(case=4, u=u_bar, v=v_bar, T=T_bar)
    F3_bar = (rho_bar * v_bar**2) + p_bar - TAU_YY(case=2, u=u_bar, v=v_bar, T=T_bar)
    F5_bar = ((E_t_bar + p_bar) * v_bar) - ((u_bar * TAU_XY(case=4, u=u_bar, v=v_bar, T=T_bar)) - (v_bar * TAU_YY(case=2, u=u_bar, v=v_bar, T=T_bar)) + Q_Y(case=2, T=T_bar))

    U1[1:-1, 1:-1] = 0.5 * (U1[1:-1, 1:-1] + U1_bar[1:-1, 1:-1] - (dt * (E1_bar[1:-1, 1:-1] - E1_bar[1:-1, :-2]) / dx) - (dt * (F1_bar[1:-1, 1:-1] - F1_bar[:-2, 1:-1]) / dy))
    U2[1:-1, 1:-1] = 0.5 * (U2[1:-1, 1:-1] + U2_bar[1:-1, 1:-1] - (dt * (E2_bar[1:-1, 1:-1] - E2_bar[1:-1, :-2]) / dx) - (dt * (F2_bar[1:-1, 1:-1] - F2_bar[:-2, 1:-1]) / dy))
    U3[1:-1, 1:-1] = 0.5 * (U3[1:-1, 1:-1] + U3_bar[1:-1, 1:-1] - (dt * (E3_bar[1:-1, 1:-1] - E3_bar[1:-1, :-2]) / dx) - (dt * (F3_bar[1:-1, 1:-1] - F3_bar[:-2, 1:-1]) / dy))
    U5[1:-1, 1:-1] = 0.5 * (U5[1:-1, 1:-1] + U5_bar[1:-1, 1:-1] - (dt * (E5_bar[1:-1, 1:-1] - E5_bar[1:-1, :-2]) / dx) - (dt * (F5_bar[1:-1, 1:-1] - F5_bar[:-2, 1:-1]) / dy))

    rho = U1
    u = U2 / rho
    v = U3 / rho
    e = (U5 / rho) - (((U2 / rho)**2 + (U3 / rho)**2) / 2)
    T = e / cv
    p = rho * R * T

    u, v, p, T, rho, e = boundary_conditions(u_field=u, v_field=v, p_field=p, T_field=T, T_case=T_case, rho=rho, e=e)
    mu = DYN_VIS(T)
    k = THEMC(mu)
    rho = p / (R * T)
    a = np.sqrt(gamma * R * T)
    e = cv * T

    return u, v, p, T, rho, mu, k, a, e


def time_step(u, v, T, rho):
    nu = -float('inf')
    nu_factor = max(4 / 3, gamma / Pr_number)
    for i in range(1, Ny - 1):
        for j in range(1, Nx - 1):
            mu = DYN_VIS(T[i, j])
            temp_nu = nu_factor * mu / rho[i, j]
            nu = max(nu, temp_nu)

    # Determine 'dt_cfl' that will be used in the final equation for 'dt'
    dt = float('inf')
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            a = (gamma * R * T[i, j]) ** 0.5
            dxdy = 1 / dx ** 2 + 1 / dy ** 2
            term_0 = abs(u[i, j]) / dx
            term_1 = abs(v[i, j]) / dy
            term_2 = a * dxdy ** 0.5
            term_3 = 2 * nu * dxdy
            dt_cfl = K * (term_0 + term_1 + term_2 + term_3) ** -1
            dt = min(dt, dt_cfl)

    return dt

def convergence_check(u_old, u_new, iteration, tol=1e-10):
    error = np.max(np.abs(u_new - u_old))
    if error < tol:
        return True, iteration, error
    else:
        return False, None, error
    
def plotting(u, v, p, T, rho, mu, a):
    plt.figure(1, figsize=(10, 6))
    plt.axhline(y=0, color='black', linewidth=4, label='Flat Plate')  # horizontal line at y = 0
    plt.contourf(X, Y, u, 100, cmap='jet')
    plt.title("U-Velocity Field")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.colorbar(label="u [m/s]")

    plt.figure(2, figsize=(10, 6))
    plt.axhline(y=0, color='black', linewidth=4, label='Flat Plate')  # horizontal line at y = 0
    plt.contourf(X, Y, v, 100, cmap='jet')
    plt.title("V-Velocity Field")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.colorbar(label="v [m/s]")

    plt.figure(3, figsize=(10, 6))
    plt.axhline(y=0, color='black', linewidth=4, label='Flat Plate')  # horizontal line at y = 0
    plt.contourf(X, Y, p, 100, cmap='jet')
    plt.title("Pressure Field")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.colorbar(label="Pressure [Pa]")

    plt.figure(4, figsize=(10, 6))
    plt.axhline(y=0, color='black', linewidth=4, label='Flat Plate')  # horizontal line at y = 0
    plt.contourf(X, Y, T, 100, cmap='jet')
    plt.contour(X, Y, T, levels=20, colors='k', linewidths=0.5)
    plt.title("Temperature Field")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.colorbar(label="Temperature [K]")

    plt.figure(5, figsize=(10, 6))
    plt.axhline(y=0, color='black', linewidth=4, label='Flat Plate')  # horizontal line at y = 0
    plt.contourf(X, Y, rho, 100, cmap='jet')
    plt.title("Density Field")
    plt.xlabel("x [m]") 
    plt.ylabel("y [m]")
    plt.colorbar(label="Density [kg/m^3]")

    plt.figure(6, figsize=(10, 6))
    plt.axhline(y=0, color='black', linewidth=4, label='Flat Plate')  # horizontal line at y = 0
    plt.contourf(X, Y, mu, 100, cmap='jet')
    plt.title("Dynamic Viscosity Field")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.colorbar(label="Dynamic Viscosity [Pa.s]")

    plt.figure(7, figsize=(10, 6))
    plt.axhline(y=0, color='black', linewidth=4, label='Flat Plate')  # flat plate
    plt.contourf(X, Y, (np.sqrt(u**2 + v**2) / a), 100, cmap='jet')   # Mach contour
    plt.streamplot(X, Y, u/a, v/a, color='white', linewidth=1)            # streamlines in white
    plt.title("Mach Field with Streamlines")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.colorbar(label="Mach Number")
    plt.suptitle("Supersonic Flow Over a Flat Plate", fontsize=16)

    plt.figure(8, figsize=(10, 6))
    plt.plot(T[:, -1] / T_freestrem, y, label="normalised surface Temperature")
    plt.plot(u[:, -1] / (M_freestream * a_freestrem), y, label="normalised surface velocity")
    plt.plot(p[:, -1] / P_freestrem, y, label="normalised Surface pressure")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    T_case = "adia"

    u, v, p, T, rho, e = inititialize(T_case=T_case)
        
    for n in range(Nt):

        if np.any(np.isnan(T)) or np.any(T < 0):
            print(f"NaN or negative T at iteration {n}")
            break
        u_old = u.copy()
        v_old = v.copy()
        p_old = p.copy()
        T_old = T.copy()
        mu = DYN_VIS(T)
        k = THEMC(mu)
        a = np.sqrt(gamma * R * T)
        e = cv * T

        dt = time_step(u, v, T, rho)

        u, v, p, T, rho, mu, k, a, e = MAC(rho, u, v, e, T, p, dt=dt, T_case=T_case)

        converged, conv_iter, err = convergence_check(u_old=u_old, u_new=u, iteration=n)
        if converged:
            print(f"Converged at iteration {conv_iter} with error {err:.2e}")
            plotting(u, v, p, T, rho, mu, a)
            # break
        elif n % 100 == 0:
            print(f"Iteration {n}, Max u change: error {err:.2e}")
            if n == 6900:
                plotting(u, v, p, T, rho, mu, a)
    return

if __name__ == "__main__":
    main()