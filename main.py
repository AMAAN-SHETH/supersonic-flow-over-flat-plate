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
Nx = 70
Ny = 70
Lx = 0.00001
Re_L = rho_freestrem * M_freestream * a_freestrem * Lx / mu_freestrem
Ly = 25 * Lx / np.sqrt(Re_L)
K = 0.6
Nt = 7000
dx = Lx / (Nx-1)
dy = Ly / (Ny-1)    


def primitive_variables(U1, U2,U3, U5):
    rho = U1
    u = U2 / rho
    v = U3 / rho
    e = (U5 / rho) - (((U2 / rho)**2 + (U3 / rho)**2) / 2)
    T = e / cv
    p = rho * R * T

    return u, v, p, T, rho, e

def conservative_variables(rho, u, v, e):
    U1 = rho
    U2 = rho * u
    U3 = rho * v
    U5 = rho * (e + 0.5 * (u**2 + v**2))

    return U1, U2, U3, U5

def compute_E(u, v, T, rho, E_t, p, Q_X, TAU_XX, TAU_XY):
    E1 = rho * u
    E2 = (rho * u**2) + p - TAU_XX
    E3 = (rho * u * v) - TAU_XY
    E5 = ((E_t + p) * u) - (u * TAU_XX) - (v * TAU_XY) + Q_X
    return E1, E2, E3, E5

def compute_F(u, v, T, rho, E_t, p, Q_Y, TAU_YX, TAU_YY):
    F1 = rho * v
    F2 = (rho * u * v) - TAU_YX
    F3 = (rho * v**2) + p - TAU_YY
    F5 = ((E_t + p) * v) - (u * TAU_YX) - (v * TAU_YY) + Q_Y
    return F1, F2, F3, F5   

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

def DYN_VIS(T):
    mu = mu_freestrem * ((T / T_freestrem)**(3/2)) * ((T_freestrem + 110) / (T + 110))
    return mu
 
def THEMC(mu):
    k = mu * cp / Pr_number
    return k

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

def MAC(u, v, T, rho, p, e, T_case):

    U1, U2, U3, U5 = conservative_variables(rho, u, v, e)
    E_t = U5.copy()

    dt = time_step(u, v, T, rho)

    tau_xx_E = TAU_XX(1, u, v, T)
    tau_xy_E = TAU_XY(1, u, v, T)
    tau_xy_F = TAU_XY(3, u, v, T)
    tau_yy_F = TAU_YY(1, u, v, T)
    qx_E = Q_X(1, T)
    qy_F = Q_Y(1, T)

    E1, E2, E3, E5 = compute_E(u, v, T, rho, E_t, p, qx_E, tau_xx_E, tau_xy_E)
    F1, F2, F3, F5 = compute_F(u, v, T, rho, E_t, p, qy_F, tau_xy_F, tau_yy_F)

    U1_bar = np.copy(U1)
    U2_bar = np.copy(U2)
    U3_bar = np.copy(U3)
    U5_bar = np.copy(U5)

    U1_bar[1:-1, 1:-1] = U1[1:-1, 1:-1] - (dt * (E1[1:-1, 2:] - E1[1:-1, 1:-1]) / dx) - (dt * (F1[2:, 1:-1] - F1[1:-1, 1:-1]) / dy)
    U2_bar[1:-1, 1:-1] = U2[1:-1, 1:-1] - (dt * (E2[1:-1, 2:] - E2[1:-1, 1:-1]) / dx) - (dt * (F2[2:, 1:-1] - F2[1:-1, 1:-1]) / dy)
    U3_bar[1:-1, 1:-1] = U3[1:-1, 1:-1] - (dt * (E3[1:-1, 2:] - E3[1:-1, 1:-1]) / dx) - (dt * (F3[2:, 1:-1] - F3[1:-1, 1:-1]) / dy)
    U5_bar[1:-1, 1:-1] = U5[1:-1, 1:-1] - (dt * (E5[1:-1, 2:] - E5[1:-1, 1:-1]) / dx) - (dt * (F5[2:, 1:-1] - F5[1:-1, 1:-1]) / dy)

    u_bar, v_bar, p_bar, T_bar, rho_bar, e_bar = primitive_variables(U1_bar, U2_bar, U3_bar, U5_bar)
    E_t_bar = U5_bar.copy()

    u_bar, v_bar, p_bar, T_bar, rho_bar, e_bar = boundary_conditions(u_field=u_bar, v_field=v_bar, p_field=p_bar, T_field=T_bar, rho=rho_bar, e=e_bar, T_case=T_case)

    tau_xx_E = TAU_XX(2, u_bar, v_bar, T_bar)
    tau_xy_E = TAU_XY(2, u_bar, v_bar, T_bar)
    tau_xy_F = TAU_XY(4, u_bar, v_bar, T_bar)
    tau_yy_F = TAU_YY(2, u_bar, v_bar, T_bar)
    qx_E = Q_X(2, T_bar)    
    qy_F = Q_Y(2, T_bar)

    E1_bar, E2_bar, E3_bar, E5_bar = compute_E(u_bar, v_bar, T_bar, rho_bar, E_t_bar, p_bar, qx_E, tau_xx_E, tau_xy_E)
    F1_bar, F2_bar, F3_bar, F5_bar = compute_F(u_bar, v_bar, T_bar, rho_bar, E_t_bar, p_bar, qy_F, tau_xy_F, tau_yy_F)

    U1[1:-1, 1:-1] = 0.5 * (U1[1:-1, 1:-1] + U1_bar[1:-1, 1:-1] - (dt * (E1_bar[1:-1, 1:-1] - E1_bar[1:-1, :-2]) / dx) - (dt * (F1_bar[1:-1, 1:-1] - F1_bar[:-2, 1:-1]) / dy))
    U2[1:-1, 1:-1] = 0.5 * (U2[1:-1, 1:-1] + U2_bar[1:-1, 1:-1] - (dt * (E2_bar[1:-1, 1:-1] - E2_bar[1:-1, :-2]) / dx) - (dt * (F2_bar[1:-1, 1:-1] - F2_bar[:-2, 1:-1]) / dy))
    U3[1:-1, 1:-1] = 0.5 * (U3[1:-1, 1:-1] + U3_bar[1:-1, 1:-1] - (dt * (E3_bar[1:-1, 1:-1] - E3_bar[1:-1, :-2]) / dx) - (dt * (F3_bar[1:-1, 1:-1] - F3_bar[:-2, 1:-1]) / dy))
    U5[1:-1, 1:-1] = 0.5 * (U5[1:-1, 1:-1] + U5_bar[1:-1, 1:-1] - (dt * (E5_bar[1:-1, 1:-1] - E5_bar[1:-1, :-2]) / dx) - (dt * (F5_bar[1:-1, 1:-1] - F5_bar[:-2, 1:-1]) / dy))

    u, v, p, T, rho, e = primitive_variables(U1, U2, U3, U5)
    u, v, p, T, rho, e = boundary_conditions(u_field=u, v_field=v, p_field=p, T_field=T, rho=rho, e=e, T_case=T_case)

    return u, v, p, T, rho, e, dt

    
def plotting(u, v, p, T, rho, mu, a):
    plt.figure(1, figsize=(10, 6))
    plt.axhline(y=0, color='black', linewidth=4, label='Flat Plate')  # horizontal line at y = 0
    plt.contourf(X, Y, u, 40, cmap='jet')
    plt.title("U-Velocity Field")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.colorbar(label="u [m/s]")

    plt.figure(2, figsize=(10, 6))
    plt.axhline(y=0, color='black', linewidth=4, label='Flat Plate')  # horizontal line at y = 0
    plt.contourf(X, Y, v, 40, cmap='jet')
    plt.title("V-Velocity Field")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.colorbar(label="v [m/s]")

    plt.figure(3, figsize=(10, 6))
    plt.axhline(y=0, color='black', linewidth=4, label='Flat Plate')  # horizontal line at y = 0
    plt.contourf(X, Y, p, 40, cmap='jet')
    plt.title("Pressure Field")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.colorbar(label="Pressure [Pa]")

    plt.figure(4, figsize=(10, 6))
    plt.axhline(y=0, color='black', linewidth=4, label='Flat Plate')  # horizontal line at y = 0
    plt.contourf(X, Y, T, 40, cmap='jet')
    # plt.contour(X, Y, T, levels=20, colors='k', linewidths=0.5)
    plt.title("Temperature Field")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.colorbar(label="Temperature [K]")

    plt.figure(5, figsize=(10, 6))
    plt.axhline(y=0, color='black', linewidth=4, label='Flat Plate')  # horizontal line at y = 0
    plt.contourf(X, Y, rho, 40, cmap='jet')
    plt.title("Density Field")
    plt.xlabel("x [m]") 
    plt.ylabel("y [m]")
    plt.colorbar(label="Density [kg/m^3]")

    plt.figure(6, figsize=(10, 6))
    plt.axhline(y=0, color='black', linewidth=4, label='Flat Plate')  # horizontal line at y = 0
    plt.contourf(X, Y, mu, 40, cmap='jet')
    plt.title("Dynamic Viscosity Field")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.colorbar(label="Dynamic Viscosity [Pa.s]")

    plt.figure(7, figsize=(10, 6))
    plt.axhline(y=0, color='black', linewidth=4, label='Flat Plate')  # flat plate
    plt.contourf(X, Y, (np.sqrt(u**2 + v**2) / a), 40, cmap='jet')   # Mach contour
    plt.streamplot(X, Y, u/a, v/a, color='white', linewidth=1)            # streamlines in white
    plt.title("Mach Field with Streamlines")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.colorbar(label="Mach Number")
    plt.suptitle("Supersonic Flow Over a Flat Plate", fontsize=16)

    plt.tight_layout()
    plt.show()

def cross_plots(T_const, T_adia, u_const, u_adia, p_const, p_adia):
    plt.figure(1, figsize=(10, 6))
    plt.title("Temperature at trailing edge")
    plt.plot(T_const[:, -1] / T_freestrem, y, label="constant")
    plt.plot(T_adia[:, -1] / T_freestrem, y, label="adiabatic")
    plt.legend()

    plt.figure(2, figsize=(10,6))
    plt.title("U at trailing edge")
    plt.plot(u_const[:, -1] / (M_freestream * a_freestrem), y, label="const")
    plt.plot(u_adia[:, -1] / (M_freestream * a_freestrem), y, label="adiabatic")
    plt.legend()

    plt.figure(3, figsize=(10,6))
    plt.title("Pressure at trailing edge")
    plt.plot(p_adia[:, -1] / P_freestrem, y, label="adiabatic")
    plt.plot(p_const[:, -1] / P_freestrem, y, label="constant")
    plt.legend()

    plt.figure(4, figsize=(10,6))
    plt.title("Pressure at Surface")
    plt.plot(x, p_adia[0, :] / P_freestrem, label="adiabatic")
    plt.plot(x, p_const[0, :] / P_freestrem, label="constatnt")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
def main():
    
    u_adia, v_adia, p_adia, T_adia, rho_adia, e_adia = inititialize(T_case="adia")
    u_const, v_const, p_const, T_const, rho_const, e_const = inititialize(T_case="const")

    for i in range(Nt):
        u_adia, v_adia, p_adia, T_adia, rho_adia, e_adia, dt = MAC(u=u_adia, v=v_adia, T=T_adia, rho=rho_adia, p=p_adia, e=e_adia, T_case="adia")
        u_const, v_const, p_const, T_const, rho_const, e_const, dt = MAC(u=u_const, v=v_const, T=T_const, rho=rho_const, p=p_const, e=e_const, T_case="const")   
        if i % 100 == 0:
            print(f"Iteration {i}")
        
    plotting(u_adia, v_adia, p_adia, T_adia, rho_adia, DYN_VIS(T_adia), np.sqrt(gamma * R * T_adia))
    # plotting(u_const, v_const, p_const, T_const, rho_const, DYN_VIS(T_const), np.sqrt(gamma * R * T_const))
    # cross_plots(u_adia=u_adia, u_const=u_const, T_adia=T_adia, T_const=T_const, p_adia=p_adia, p_const=p_const)


main()






