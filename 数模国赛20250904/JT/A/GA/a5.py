import numpy as np
import numba
from numba import jit, prange
import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# 常量定义 - 使用浮点数
MISSILE_POSITIONS = np.array([
    [20000.0, 0.0, 2000.0],  # M1
    [19000.0, 600.0, 2100.0],  # M2
    [18000.0, -600.0, 1900.0]  # M3
])

UAV_INIT_POSITIONS = np.array([
    [17800.0, 0.0, 1800.0],  # FY1
    [12000.0, 1400.0, 1400.0],  # FY2
    [6000.0, -3000.0, 700.0],  # FY3
    [11000.0, 2000.0, 1800.0],  # FY4
    [13000.0, -2000.0, 1300.0]  # FY5
])

TARGET_CENTER = np.array([0.0, 200.0, 0.0])
TARGET_RADIUS = 7.0
TARGET_HEIGHT = 10.0

MISSILE_SPEED = 300.0
UAV_SPEED_MIN = 70.0
UAV_SPEED_MAX = 140.0
SMOKE_SINK_SPEED = 3.0
SMOKE_EFFECTIVE_RADIUS = 10.0
SMOKE_EFFECTIVE_TIME = 20.0
GRAVITY = 9.8

SIMULATION_START = 0.0
SIMULATION_END = 100.0
TIME_STEP = 0.01

# 遗传算法参数
POPULATION_SIZE = 800  # 减小种群大小以提高性能
NUM_GENERATIONS = 300  # 减少代数
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.25
ELITISM_COUNT = 2

# 目标圆柱体采样点
NUM_TARGET_SAMPLES = 100  # 减少采样点数量


def generate_target_samples():
    theta = np.linspace(0, 2 * np.pi, NUM_TARGET_SAMPLES)
    bottom_circle = np.column_stack([
        TARGET_CENTER[0] + TARGET_RADIUS * np.cos(theta),
        TARGET_CENTER[1] + TARGET_RADIUS * np.sin(theta),
        np.full(NUM_TARGET_SAMPLES, TARGET_CENTER[2] - TARGET_HEIGHT / 2)
    ])

    top_circle = np.column_stack([
        TARGET_CENTER[0] + TARGET_RADIUS * np.cos(theta),
        TARGET_CENTER[1] + TARGET_RADIUS * np.sin(theta),
        np.full(NUM_TARGET_SAMPLES, TARGET_CENTER[2] + TARGET_HEIGHT / 2)
    ])

    return np.vstack([bottom_circle, top_circle])


TARGET_SAMPLES = generate_target_samples()


# 预计算导弹轨迹
@jit(nopython=True)
def calculate_missile_trajectories():
    time_steps = int((SIMULATION_END - SIMULATION_START) / TIME_STEP) + 1
    trajectories = np.zeros((3, time_steps, 3))  # 3 missiles, time steps, 3 coordinates

    for i in range(3):
        initial_pos = MISSILE_POSITIONS[i]
        # 手动计算范数，避免使用np.linalg.norm
        norm_val = np.sqrt(initial_pos[0] ** 2 + initial_pos[1] ** 2 + initial_pos[2] ** 2)
        direction = -initial_pos / norm_val

        for t_idx in range(time_steps):
            t = SIMULATION_START + t_idx * TIME_STEP
            trajectories[i, t_idx] = initial_pos + direction * MISSILE_SPEED * t

    return trajectories


MISSILE_TRAJECTORIES = calculate_missile_trajectories()


# 计算线段到点的最短距离
@jit(nopython=True)
def point_to_line_distance(point, line_start, line_end):
    line_vec = line_end - line_start
    point_vec = point - line_start
    # 手动计算范数
    line_len = np.sqrt(line_vec[0] ** 2 + line_vec[1] ** 2 + line_vec[2] ** 2)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len

    t = line_unitvec[0] * point_vec_scaled[0] + line_unitvec[1] * point_vec_scaled[1] + line_unitvec[2] * \
        point_vec_scaled[2]
    t = max(0.0, min(1.0, t))

    nearest = line_start + t * line_vec
    # 手动计算距离
    dx = nearest[0] - point[0]
    dy = nearest[1] - point[1]
    dz = nearest[2] - point[2]
    return np.sqrt(dx * dx + dy * dy + dz * dz)


# 检查烟雾是否遮蔽目标点
@jit(nopython=True)
def is_point_obscured(missile_pos, target_point, smoke_center, smoke_radius):
    return point_to_line_distance(smoke_center, missile_pos, target_point) <= smoke_radius


# 检查目标是否完全被遮蔽
@jit(nopython=True)
def is_target_obscured(missile_pos, smoke_centers, smoke_radius):
    for i in range(TARGET_SAMPLES.shape[0]):
        target_point = TARGET_SAMPLES[i]
        obscured = False
        for j in range(smoke_centers.shape[0]):
            if is_point_obscured(missile_pos, target_point, smoke_centers[j], smoke_radius):
                obscured = True
                break
        if not obscured:
            return False
    return True


# 计算烟雾弹轨迹和有效时间
@jit(nopython=True)
def calculate_smoke_parameters(uav_params, uav_idx, smoke_idx):
    # 解码参数
    param_offset = (uav_idx * 3 + smoke_idx) * 4
    speed = uav_params[param_offset]
    direction_deg = uav_params[param_offset + 1]
    deploy_time = uav_params[param_offset + 2]
    detonation_delay = uav_params[param_offset + 3]

    # 转换为弧度
    direction_rad = np.deg2rad(direction_deg)

    # 计算投放点
    uav_init_pos = UAV_INIT_POSITIONS[uav_idx]
    uav_direction = np.array([np.cos(direction_rad), np.sin(direction_rad), 0.0])
    deploy_point = uav_init_pos + uav_direction * speed * deploy_time

    # 计算爆炸点
    explosion_time = deploy_time + detonation_delay
    explosion_point = deploy_point + uav_direction * speed * detonation_delay
    explosion_point[2] -= 0.5 * GRAVITY * detonation_delay ** 2

    # 计算烟雾有效时间窗口
    effective_start = explosion_time
    effective_end = explosion_time + SMOKE_EFFECTIVE_TIME

    return deploy_point, explosion_point, effective_start, effective_end


# 计算适应度（每个导弹被遮蔽时间的总和）
@jit(nopython=True)
def calculate_fitness_individual(individual):
    # 这个变量现在代表“各个导弹遮蔽时间的总和”，作为适应度分数
    total_individual_obscured_time = 0.0
    num_uavs = 5
    num_smokes_per_uav = 3
    total_smokes = num_uavs * num_smokes_per_uav

    # 这部分预计算烟雾参数的代码保持不变
    smoke_params_arr = np.zeros((total_smokes, 5))
    for uav_idx in range(num_uavs):
        for smoke_idx in range(num_smokes_per_uav):
            deploy_point, explosion_point, effective_start, effective_end = calculate_smoke_parameters(
                individual, uav_idx, smoke_idx
            )
            param_idx = uav_idx * num_smokes_per_uav + smoke_idx
            smoke_params_arr[param_idx, 0:3] = explosion_point
            smoke_params_arr[param_idx, 3] = effective_start
            smoke_params_arr[param_idx, 4] = effective_end

    active_smoke_centers_arr = np.zeros((total_smokes, 3))

    # 模拟时间步
    for t_idx in range(MISSILE_TRAJECTORIES.shape[1]):
        t = SIMULATION_START + t_idx * TIME_STEP

        num_active_smokes = 0
        for i in range(total_smokes):
            explosion_point = smoke_params_arr[i, 0:3]
            effective_start = smoke_params_arr[i, 3]
            effective_end = smoke_params_arr[i, 4]

            if effective_start <= t <= effective_end:
                sink_distance = SMOKE_SINK_SPEED * (t - effective_start)
                smoke_center = explosion_point.copy()
                smoke_center[2] -= sink_distance
                active_smoke_centers_arr[num_active_smokes] = smoke_center
                num_active_smokes += 1

        if num_active_smokes == 0:
            continue

        valid_smoke_centers = active_smoke_centers_arr[:num_active_smokes]

        # ==================== MODIFIED LOGIC START ====================
        # 原来的逻辑是检查是否“所有”导弹都被遮蔽，才增加一次时间。
        # 新逻辑是检查“每一个”导弹，只要被遮蔽，就为其增加时间。

        # 循环检查每个导弹
        for missile_idx in range(3):
            missile_pos = MISSILE_TRAJECTORIES[missile_idx, t_idx]
            # 如果当前导弹的视线被任何一个烟雾遮蔽
            if is_target_obscured(missile_pos, valid_smoke_centers, SMOKE_EFFECTIVE_RADIUS):
                # 将这个时间步长累加到总分中
                total_individual_obscured_time += TIME_STEP
        # ===================== MODIFIED LOGIC END =====================

    return total_individual_obscured_time


# 遗传算法操作
def initialize_population():
    # 每个个体有 5 UAVs * 3 smokes * 4 parameters = 60 个基因
    population = np.zeros((POPULATION_SIZE, 60))

    for i in range(POPULATION_SIZE):
        for j in range(60):
            if j % 4 == 0:  # 速度参数
                population[i, j] = np.random.uniform(UAV_SPEED_MIN, UAV_SPEED_MAX)
            elif j % 4 == 1:  # 方向参数 (角度)
                population[i, j] = np.random.uniform(0, 360)
            elif j % 4 == 2:  # 投放时间
                population[i, j] = np.random.uniform(0, 10)
            else:  # 引爆延迟
                population[i, j] = np.random.uniform(0, 8)

    return population


def crossover(parent1, parent2):
    if np.random.rand() < CROSSOVER_RATE:
        crossover_point = np.random.randint(1, len(parent1) - 1)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
    return parent1.copy(), parent2.copy()


def mutate(individual):
    for i in range(len(individual)):
        if np.random.rand() < MUTATION_RATE:
            if i % 4 == 0:  # 速度参数
                individual[i] = np.clip(individual[i] + np.random.normal(0, 5), UAV_SPEED_MIN, UAV_SPEED_MAX)
            elif i % 4 == 1:  # 方向参数
                individual[i] = np.clip(individual[i] + np.random.normal(0, 10), 0, 360)
            elif i % 4 == 2:  # 投放时间
                individual[i] = np.clip(individual[i] + np.random.normal(0, 1), 0, 10)
            else:  # 引爆延迟
                individual[i] = np.clip(individual[i] + np.random.normal(0, 0.5), 0, 8)
    return individual


# Add this new function
@jit(nopython=True, parallel=True)
def calculate_fitness(population):
    fitness = np.zeros(population.shape[0])
    # Use prange for parallel execution
    for i in prange(population.shape[0]):
        fitness[i] = calculate_fitness_individual(population[i])
    return fitness

def genetic_algorithm():
    population = initialize_population()
    best_fitness_history = []
    best_individual = None
    best_fitness = -1

    for generation in range(NUM_GENERATIONS):
        start_time = time.time()

        # 计算适应度
        fitness = calculate_fitness(population)

        # 更新最佳个体
        max_fitness_idx = np.argmax(fitness)
        if fitness[max_fitness_idx] > best_fitness:
            best_fitness = fitness[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()

        best_fitness_history.append(best_fitness)

        # 选择精英
        elite_indices = np.argsort(fitness)[-ELITISM_COUNT:]
        new_population = population[elite_indices].copy()

        # 锦标赛选择
        while len(new_population) < POPULATION_SIZE:
            # 选择父代
            tournament_size = 3
            tournament_indices = np.random.choice(POPULATION_SIZE, tournament_size, replace=False)
            tournament_fitness = fitness[tournament_indices]
            parent1_idx = tournament_indices[np.argmax(tournament_fitness)]

            tournament_indices = np.random.choice(POPULATION_SIZE, tournament_size, replace=False)
            tournament_fitness = fitness[tournament_indices]
            parent2_idx = tournament_indices[np.argmax(tournament_fitness)]

            # 交叉
            child1, child2 = crossover(population[parent1_idx], population[parent2_idx])

            # 变异
            child1 = mutate(child1)
            child2 = mutate(child2)

            new_population = np.vstack([new_population, child1])
            if len(new_population) < POPULATION_SIZE:
                new_population = np.vstack([new_population, child2])

        population = new_population

        end_time = time.time()
        print(
            f"Generation {generation + 1}/{NUM_GENERATIONS}, Best Fitness: {best_fitness:.2f}, Time: {end_time - start_time:.2f}s")

    return best_individual, best_fitness, best_fitness_history


# 分析最优解
def analyze_solution(best_individual):
    print("最优解分析:")
    print("=" * 50)

    smoke_details = []

    for uav_idx in range(5):
        for smoke_idx in range(3):
            param_offset = (uav_idx * 3 + smoke_idx) * 4
            speed = best_individual[param_offset]
            direction_deg = best_individual[param_offset + 1]
            deploy_time = best_individual[param_offset + 2]
            detonation_delay = best_individual[param_offset + 3]

            direction_rad = np.deg2rad(direction_deg)
            uav_init_pos = UAV_INIT_POSITIONS[uav_idx]
            uav_direction = np.array([np.cos(direction_rad), np.sin(direction_rad), 0.0])

            deploy_point = uav_init_pos + uav_direction * speed * deploy_time
            explosion_point = deploy_point + uav_direction * speed * detonation_delay
            explosion_point[2] -= 0.5 * GRAVITY * detonation_delay ** 2

            effective_start = deploy_time + detonation_delay
            effective_end = effective_start + SMOKE_EFFECTIVE_TIME

            # 计算这个烟雾弹对每个导弹的遮蔽时间
            obscured_times = np.zeros(3)
            for missile_idx in range(3):
                obscured_time = 0
                for t_idx in range(MISSILE_TRAJECTORIES.shape[1]):
                    t = SIMULATION_START + t_idx * TIME_STEP
                    if effective_start <= t <= effective_end:
                        sink_distance = SMOKE_SINK_SPEED * (t - effective_start)
                        smoke_center = explosion_point.copy()
                        smoke_center[2] -= sink_distance

                        missile_pos = MISSILE_TRAJECTORIES[missile_idx, t_idx]
                        # 创建一个包含单个烟雾中心的数组
                        single_smoke_center = np.array([smoke_center])
                        if is_target_obscured(missile_pos, single_smoke_center, SMOKE_EFFECTIVE_RADIUS):
                            obscured_time += TIME_STEP

                obscured_times[missile_idx] = obscured_time

            smoke_details.append({
                'uav': uav_idx + 1,
                'smoke': smoke_idx + 1,
                'speed': speed,
                'direction': direction_deg,
                'deploy_time': deploy_time,
                'detonation_delay': detonation_delay,
                'deploy_point': deploy_point,
                'explosion_point': explosion_point,
                'obscured_times': obscured_times
            })

            print(f"UAV {uav_idx + 1} 烟雾弹 {smoke_idx + 1}:")
            print(f"  速度: {speed:.2f} m/s, 方向: {direction_deg:.2f}°")
            print(f"  投放时间: {deploy_time:.2f} s, 引爆延迟: {detonation_delay:.2f} s")
            print(f"  投放点: ({deploy_point[0]:.2f}, {deploy_point[1]:.2f}, {deploy_point[2]:.2f})")
            print(f"  爆炸点: ({explosion_point[0]:.2f}, {explosion_point[1]:.2f}, {explosion_point[2]:.2f})")
            print(f"  对M1遮蔽时间: {obscured_times[0]:.2f} s")
            print(f"  对M2遮蔽时间: {obscured_times[1]:.2f} s")
            print(f"  对M3遮蔽时间: {obscured_times[2]:.2f} s")
            print()

    # 计算总遮蔽时间
    total_obscured_time = calculate_fitness_individual(best_individual)
    print(f"\n最终适应度得分 (各导弹遮蔽时间总和): {total_obscured_time:.2f} s")

    return smoke_details, total_obscured_time


# 主函数
def main():
    print("开始优化...")
    start_time = time.time()

    best_individual, best_fitness, fitness_history = genetic_algorithm()

    end_time = time.time()
    print(f"优化完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"最佳适应度: {best_fitness:.2f}")

    # 分析最优解
    smoke_details, total_obscured_time = analyze_solution(best_individual)

    # 绘制进化过程
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_GENERATIONS + 1), fitness_history)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (Obscured Time)')
    plt.title('Genetic Algorithm Evolution')
    plt.grid(True)
    plt.savefig('evolution.png')
    plt.show()

    return best_individual, smoke_details, total_obscured_time


if __name__ == "__main__":
    best_individual, smoke_details, total_obscured_time = main()