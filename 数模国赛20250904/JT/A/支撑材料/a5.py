import numpy as np
import numba
from numba import jit, prange
import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

M_P_GGG = np.array([
    [20000.0, 0.0, 2000.0],
    [19000.0, 600.0, 2100.0],
    [18000.0, -600.0, 1900.0]
])

U_P_GGG = np.array([
    [17800.0, 0.0, 1800.0],
    [12000.0, 1400.0, 1400.0],
    [6000.0, -3000.0, 700.0],
    [11000.0, 2000.0, 1800.0],
    [13000.0, -2000.0, 1300.0]
])

T_C_GGG = np.array([0.0, 200.0, 0.0])
T_R_GGG = 7.0
T_H_GGG = 10.0

M_S_GGG = 300.0
U_N_GGG = 70.0
U_X_GGG = 140.0
S_K_GGG = 3.0
S_R_GGG = 10.0
S_T_GGG = 20.0
G_G_GGG = 9.8

S_A_GGG = 0.0
S_E_GGG = 100.0
T_E_GGG = 0.01

P_S_GGG = 800
N_G_GGG = 300
C_R_GGG = 0.85
M_R_GGG = 0.25
E_C_GGG = 2

N_S_GGG = 100

def g_t_ggg():
    theta = np.linspace(0, 2 * np.pi, N_S_GGG)
    b_c_ggg = np.column_stack([
        T_C_GGG[0] + T_R_GGG * np.cos(theta),
        T_C_GGG[1] + T_R_GGG * np.sin(theta),
        np.full(N_S_GGG, T_C_GGG[2] - T_H_GGG / 2)
    ])
    t_c_ggg = np.column_stack([
        T_C_GGG[0] + T_R_GGG * np.cos(theta),
        T_C_GGG[1] + T_R_GGG * np.sin(theta),
        np.full(N_S_GGG, T_C_GGG[2] + T_H_GGG / 2)
    ])
    return np.vstack([b_c_ggg, t_c_ggg])

T_A_GGG = g_t_ggg()

@jit(nopython=True)
def c_m_ggg():
    t_s_ggg = int((S_E_GGG - S_A_GGG) / T_E_GGG) + 1
    t_r_ggg = np.zeros((3, t_s_ggg, 3))
    for i in range(3):
        i_p_ggg = M_P_GGG[i]
        n_v_ggg = np.sqrt(i_p_ggg[0] ** 2 + i_p_ggg[1] ** 2 + i_p_ggg[2] ** 2)
        direction = -i_p_ggg / n_v_ggg
        for t_idx in range(t_s_ggg):
            t = S_A_GGG + t_idx * T_E_GGG
            t_r_ggg[i, t_idx] = i_p_ggg + direction * M_S_GGG * t
    return t_r_ggg

M_T_GGG = c_m_ggg()

@jit(nopython=True)
def p_d_ggg(point, l_s_ggg, l_e_ggg):
    l_v_ggg = l_e_ggg - l_s_ggg
    p_v_ggg = point - l_s_ggg
    l_l_ggg = np.sqrt(l_v_ggg[0] ** 2 + l_v_ggg[1] ** 2 + l_v_ggg[2] ** 2)
    l_u_ggg = l_v_ggg / l_l_ggg
    p_s_ggg = p_v_ggg / l_l_ggg
    t = l_u_ggg[0] * p_s_ggg[0] + l_u_ggg[1] * p_s_ggg[1] + l_u_ggg[2] * p_s_ggg[2]
    t = max(0.0, min(1.0, t))
    n_p_ggg = l_s_ggg + t * l_v_ggg
    dx = n_p_ggg[0] - point[0]
    dy = n_p_ggg[1] - point[1]
    dz = n_p_ggg[2] - point[2]
    return np.sqrt(dx * dx + dy * dy + dz * dz)

@jit(nopython=True)
def i_o_ggg(m_p_ggg, t_p_ggg, s_c_ggg, s_r_ggg):
    return p_d_ggg(s_c_ggg, m_p_ggg, t_p_ggg) <= s_r_ggg

@jit(nopython=True)
def i_t_ggg(m_p_ggg, s_c_ggg, s_r_ggg):
    for i in range(T_A_GGG.shape[0]):
        t_p_ggg = T_A_GGG[i]
        obscured = False
        for j in range(s_c_ggg.shape[0]):
            if i_o_ggg(m_p_ggg, t_p_ggg, s_c_ggg[j], s_r_ggg):
                obscured = True
                break
        if not obscured:
            return False
    return True

@jit(nopython=True)
def c_s_ggg(u_p_ggg, uav_idx, smoke_idx):
    p_o_ggg = (uav_idx * 3 + smoke_idx) * 4
    speed = u_p_ggg[p_o_ggg]
    d_d_ggg = u_p_ggg[p_o_ggg + 1]
    d_t_ggg = u_p_ggg[p_o_ggg + 2]
    d_e_ggg = u_p_ggg[p_o_ggg + 3]
    d_r_ggg = np.deg2rad(d_d_ggg)
    u_i_ggg = U_P_GGG[uav_idx]
    u_d_ggg = np.array([np.cos(d_r_ggg), np.sin(d_r_ggg), 0.0])
    d_p_ggg = u_i_ggg + u_d_ggg * speed * d_t_ggg
    e_t_ggg = d_t_ggg + d_e_ggg
    e_p_ggg = d_p_ggg + u_d_ggg * speed * d_e_ggg
    e_p_ggg[2] -= 0.5 * G_G_GGG * d_e_ggg ** 2
    e_s_ggg = e_t_ggg
    e_e_ggg = e_t_ggg + S_T_GGG
    return d_p_ggg, e_p_ggg, e_s_ggg, e_e_ggg

@jit(nopython=True)
def c_i_ggg(individual):
    t_i_ggg = 0.0
    num_uavs = 5
    n_s_ggg = 3
    t_s_ggg = num_uavs * n_s_ggg
    s_p_ggg = np.zeros((t_s_ggg, 5))
    for uav_idx in range(num_uavs):
        for smoke_idx in range(n_s_ggg):
            d_p_ggg, e_p_ggg, e_s_ggg, e_e_ggg = c_s_ggg(
                individual, uav_idx, smoke_idx
            )
            param_idx = uav_idx * n_s_ggg + smoke_idx
            s_p_ggg[param_idx, 0:3] = e_p_ggg
            s_p_ggg[param_idx, 3] = e_s_ggg
            s_p_ggg[param_idx, 4] = e_e_ggg
    a_s_ggg = np.zeros((t_s_ggg, 3))
    for t_idx in range(M_T_GGG.shape[1]):
        t = S_A_GGG + t_idx * T_E_GGG
        n_a_ggg = 0
        for i in range(t_s_ggg):
            e_p_ggg = s_p_ggg[i, 0:3]
            e_s_ggg = s_p_ggg[i, 3]
            e_e_ggg = s_p_ggg[i, 4]
            if e_s_ggg <= t <= e_e_ggg:
                s_d_ggg = S_K_GGG * (t - e_s_ggg)
                s_c_ggg = e_p_ggg.copy()
                s_c_ggg[2] -= s_d_ggg
                a_s_ggg[n_a_ggg] = s_c_ggg
                n_a_ggg += 1
        if n_a_ggg == 0:
            continue
        v_s_ggg = a_s_ggg[:n_a_ggg]
        for missile_idx in range(3):
            m_p_ggg = M_T_GGG[missile_idx, t_idx]
            if i_t_ggg(m_p_ggg, v_s_ggg, S_R_GGG):
                t_i_ggg += T_E_GGG
    return t_i_ggg

def i_p_ggg():
    population = np.zeros((P_S_GGG, 60))
    for i in range(P_S_GGG):
        for j in range(60):
            if j % 4 == 0:
                population[i, j] = np.random.uniform(U_N_GGG, U_X_GGG)
            elif j % 4 == 1:
                population[i, j] = np.random.uniform(0, 360)
            elif j % 4 == 2:
                population[i, j] = np.random.uniform(0, 10)
            else:
                population[i, j] = np.random.uniform(0, 8)
    return population

def crossover(parent1, parent2):
    if np.random.rand() < C_R_GGG:
        c_p_ggg = np.random.randint(1, len(parent1) - 1)
        child1 = np.concatenate([parent1[:c_p_ggg], parent2[c_p_ggg:]])
        child2 = np.concatenate([parent2[:c_p_ggg], parent1[c_p_ggg:]])
        return child1, child2
    return parent1.copy(), parent2.copy()

def mutate(individual):
    for i in range(len(individual)):
        if np.random.rand() < M_R_GGG:
            if i % 4 == 0:
                individual[i] = np.clip(individual[i] + np.random.normal(0, 5), U_N_GGG, U_X_GGG)
            elif i % 4 == 1:
                individual[i] = np.clip(individual[i] + np.random.normal(0, 10), 0, 360)
            elif i % 4 == 2:
                individual[i] = np.clip(individual[i] + np.random.normal(0, 1), 0, 10)
            else:
                individual[i] = np.clip(individual[i] + np.random.normal(0, 0.5), 0, 8)
    return individual

@jit(nopython=True, parallel=True)
def c_f_ggg(population):
    fitness = np.zeros(population.shape[0])
    for i in prange(population.shape[0]):
        fitness[i] = c_i_ggg(population[i])
    return fitness

def g_a_ggg():
    population = i_p_ggg()
    b_h_ggg = []
    b_i_ggg = None
    b_f_ggg = -1
    for generation in range(N_G_GGG):
        s_t_ggg = time.time()
        fitness = c_f_ggg(population)
        m_f_ggg = np.argmax(fitness)
        if fitness[m_f_ggg] > b_f_ggg:
            b_f_ggg = fitness[m_f_ggg]
            b_i_ggg = population[m_f_ggg].copy()
        b_h_ggg.append(b_f_ggg)
        e_i_ggg = np.argsort(fitness)[-E_C_GGG:]
        n_p_ggg = population[e_i_ggg].copy()
        while len(n_p_ggg) < P_S_GGG:
            t_s_ggg = 3
            t_i_ggg = np.random.choice(P_S_GGG, t_s_ggg, replace=False)
            t_f_ggg = fitness[t_i_ggg]
            p1_idx = t_i_ggg[np.argmax(t_f_ggg)]
            t_i_ggg = np.random.choice(P_S_GGG, t_s_ggg, replace=False)
            t_f_ggg = fitness[t_i_ggg]
            p2_idx = t_i_ggg[np.argmax(t_f_ggg)]
            child1, child2 = crossover(population[p1_idx], population[p2_idx])
            child1 = mutate(child1)
            child2 = mutate(child2)
            n_p_ggg = np.vstack([n_p_ggg, child1])
            if len(n_p_ggg) < P_S_GGG:
                n_p_ggg = np.vstack([n_p_ggg, child2])
        population = n_p_ggg
        e_t_ggg = time.time()
        print(
            f"Generation {generation + 1}/{N_G_GGG}, Best Fitness: {b_f_ggg:.2f}, Time: {e_t_ggg - s_t_ggg:.2f}s")
    return b_i_ggg, b_f_ggg, b_h_ggg

def a_s_ggg(b_i_ggg):
    print("最优解分析:")
    print("=" * 50)
    s_d_ggg = []
    for uav_idx in range(5):
        for smoke_idx in range(3):
            p_o_ggg = (uav_idx * 3 + smoke_idx) * 4
            speed = b_i_ggg[p_o_ggg]
            d_d_ggg = b_i_ggg[p_o_ggg + 1]
            d_t_ggg = b_i_ggg[p_o_ggg + 2]
            d_e_ggg = b_i_ggg[p_o_ggg + 3]
            d_r_ggg = np.deg2rad(d_d_ggg)
            u_i_ggg = U_P_GGG[uav_idx]
            u_d_ggg = np.array([np.cos(d_r_ggg), np.sin(d_r_ggg), 0.0])
            d_p_ggg = u_i_ggg + u_d_ggg * speed * d_t_ggg
            e_p_ggg = d_p_ggg + u_d_ggg * speed * d_e_ggg
            e_p_ggg[2] -= 0.5 * G_G_GGG * d_e_ggg ** 2
            e_s_ggg = d_t_ggg + d_e_ggg
            e_e_ggg = e_s_ggg + S_T_GGG
            o_t_ggg = np.zeros(3)
            for missile_idx in range(3):
                o_d_ggg = 0
                for t_idx in range(M_T_GGG.shape[1]):
                    t = S_A_GGG + t_idx * T_E_GGG
                    if e_s_ggg <= t <= e_e_ggg:
                        s_k_ggg = S_K_GGG * (t - e_s_ggg)
                        s_c_ggg = e_p_ggg.copy()
                        s_c_ggg[2] -= s_k_ggg
                        m_p_ggg = M_T_GGG[missile_idx, t_idx]
                        s_c_array_ggg = np.array([s_c_ggg])
                        if i_t_ggg(m_p_ggg, s_c_array_ggg, S_R_GGG):
                            o_d_ggg += T_E_GGG
                o_t_ggg[missile_idx] = o_d_ggg
            s_d_ggg.append({
                'uav': uav_idx + 1,
                'smoke': smoke_idx + 1,
                'speed': speed,
                'direction': d_d_ggg,
                'deploy_time': d_t_ggg,
                'detonation_delay': d_e_ggg,
                'deploy_point': d_p_ggg,
                'explosion_point': e_p_ggg,
                'obscured_times': o_t_ggg
            })
            print(f"UAV {uav_idx + 1} 烟雾弹 {smoke_idx + 1}:")
            print(f"  速度: {speed:.2f} m/s, 方向: {d_d_ggg:.2f}°")
            print(f"  投放时间: {d_t_ggg:.2f} s, 引爆延迟: {d_e_ggg:.2f} s")
            print(f"  投放点: ({d_p_ggg[0]:.2f}, {d_p_ggg[1]:.2f}, {d_p_ggg[2]:.2f})")
            print(f"  爆炸点: ({e_p_ggg[0]:.2f}, {e_p_ggg[1]:.2f}, {e_p_ggg[2]:.2f})")
            print(f"  对M1遮蔽时间: {o_t_ggg[0]:.2f} s")
            print(f"  对M2遮蔽时间: {o_t_ggg[1]:.2f} s")
            print(f"  对M3遮蔽时间: {o_t_ggg[2]:.2f} s")
            print()
    t_o_ggg = c_i_ggg(b_i_ggg)
    print(f"\n最终适应度得分 (各导弹遮蔽时间总和): {t_o_ggg:.2f} s")
    return s_d_ggg, t_o_ggg

def main():
    print("开始优化...")
    s_m_ggg = time.time()
    b_i_ggg, b_f_ggg, f_h_ggg = g_a_ggg()
    e_m_ggg = time.time()
    print(f"优化完成，耗时: {e_m_ggg - s_m_ggg:.2f} 秒")
    print(f"最佳适应度: {b_f_ggg:.2f}")
    s_d_ggg, t_o_ggg = a_s_ggg(b_i_ggg)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, N_G_GGG + 1), f_h_ggg)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (Obscured Time)')
    plt.title('Genetic Algorithm Evolution')
    plt.grid(True)
    plt.savefig('evolution.png')
    plt.show()
    return b_i_ggg, s_d_ggg, t_o_ggg

if __name__ == "__main__":
    b_i_ggg, s_d_ggg, t_o_ggg = main()