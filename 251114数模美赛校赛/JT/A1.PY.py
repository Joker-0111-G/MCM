import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import permutations

# ================= 1. 参数配置 =================
ROOM_WIDTH = 6.0
ROOM_DEPTH = 5.0
HALLWAY_WIDTH = 3.0
ROOM_AREA = ROOM_WIDTH * ROOM_DEPTH
TARGET_COVERAGE = 0.95 * ROOM_AREA

BASE_SPEED = 1.0  # 正常无烟移动速度
R_MAX = math.sqrt(ROOM_WIDTH ** 2 + ROOM_DEPTH ** 2)
R_MIN = 0.5
T_CRIT = 6 * 60  # 360s 完蛋时间
LAMBDA = 3 / T_CRIT

# 【核心修改1】大幅降低搜索效率
# 在看不清的情况下，为了确认安全，会有极大的重复搜索和确认动作
SEARCH_EFFICIENCY = 0.3


# ================= 2. 核心计算逻辑修正 =================

def get_visibility_radius(t):
    """计算 t时刻的有效视场半径 R(t)"""
    val = (R_MAX - R_MIN) * np.exp(-LAMBDA * t) + R_MIN
    return max(val, R_MIN)


def calculate_sweep_duration_and_path_stats(arrival_time):
    """
    计算清扫时长，并返回此时的平均能见度用于画图。
    引入【速度惩罚】：能见度越低，走得越慢。
    """
    covered_area = 0
    t_elapsed = 0
    dt = 1.0
    current_t = arrival_time

    # 记录这一段时间的平均R，用于决定画图的密度
    r_samples = []

    while covered_area < TARGET_COVERAGE:
        r_curr = get_visibility_radius(current_t)
        r_samples.append(r_curr)

        # 【核心修改2】速度随能见度衰减模型
        # 如果 R > 2m，速度为 1.0 m/s
        # 如果 R < 0.5m，速度降至 0.2 m/s (摸索)
        # 使用简单的线性插值模拟减速
        if r_curr >= 2.0:
            current_speed = BASE_SPEED
        else:
            # 线性下降到 0.2
            ratio = (r_curr - 0.5) / (2.0 - 0.5)  # 0 to 1
            current_speed = 0.2 + 0.8 * max(0, ratio)

        # 计算这一秒扫过的面积
        # 有效宽度受限于 R 和 房间物理尺寸
        effective_width = min(2 * r_curr, min(ROOM_WIDTH, ROOM_DEPTH))

        # 面积增量 = 速度(受烟雾影响) * 宽度(受烟雾影响) * 效率
        area_step = current_speed * effective_width * dt * SEARCH_EFFICIENCY

        covered_area += area_step
        t_elapsed += dt
        current_t += dt

        if t_elapsed > 1200: break  # 强制防死循环

    # 另外，计算纯物理遍历四角所需的时间（作为底线）
    # 同样，物理遍历的速度也受此时的平均烟雾影响
    avg_r = np.mean(r_samples)
    if avg_r >= 2.0:
        walk_speed = BASE_SPEED
    else:
        ratio = (avg_r - 0.5) / 1.5
        walk_speed = 0.2 + 0.8 * max(0, ratio)

    perimeter_dist = (ROOM_WIDTH + ROOM_DEPTH) * 2
    t_physical = perimeter_dist / walk_speed

    final_time = max(t_elapsed, t_physical)

    return final_time, avg_r


# ================= 3. 路径生成算法 (可视化) =================

def generate_search_path_coords(room_id, avg_r):
    """
    根据房间ID和当时的能见度 R，生成具体的搜寻路径坐标。
    R 越小，路径越密集 (Zigzag)。
    R 很大，路径为沿墙走 (Perimeter)。
    """
    info = rooms_info[room_id]
    rx, ry, w, h = info['rect']
    door_pos = info['door']

    path = []
    path.append(door_pos)  # 从门口开始

    # 稍微向内缩一点，避免画在墙线上
    margin = 0.5
    ix, iy = rx + margin, ry + margin
    iw, ih = w - 2 * margin, h - 2 * margin

    # 四个角落坐标
    corners = [
        (ix, iy), (ix + iw, iy), (ix + iw, iy + ih), (ix, iy + ih)
    ]

    # 策略选择
    if avg_r > 2.5:
        # 【模式A：清澈】沿墙走一圈 + 稍微看一眼中间
        # 顺序：最近的角 -> ... -> 门
        path.extend(corners)
        path.append(corners[0])  # 闭环
    else:
        # 【模式B：烟雾】弓字形扫描 (Lawnmower pattern)
        # 扫描线的间距取决于 R (假设视场会有一定重叠)
        step_size = max(1.0, avg_r * 1.5)
        num_steps = int(ih / step_size) + 1

        # 生成 Zigzag
        for i in range(num_steps):
            y_level = iy + (i * (ih / max(1, num_steps)))
            if i % 2 == 0:
                # 左 -> 右
                path.append((ix, y_level))
                path.append((ix + iw, y_level))
            else:
                # 右 -> 左
                path.append((ix + iw, y_level))
                path.append((ix, y_level))

    path.append(door_pos)  # 回到门口
    return path


# ================= 4. 地图定义 =================

rooms_info = {
    1: {'rect': (0, 0, 6, 5), 'door': (5.5, 5.0), 'label': 'R1'},
    2: {'rect': (6, 0, 6, 5), 'door': (9.0, 5.0), 'label': 'R2'},
    3: {'rect': (12, 0, 6, 5), 'door': (12.5, 5.0), 'label': 'R3'},
    4: {'rect': (0, 8, 6, 5), 'door': (5.5, 8.0), 'label': 'R4'},
    5: {'rect': (6, 8, 6, 5), 'door': (9.0, 8.0), 'label': 'R5'},
    6: {'rect': (12, 8, 6, 5), 'door': (12.5, 8.0), 'label': 'R6'}
}

exits = {
    'Left': (-1, 6.5),
    'Right': (19, 6.5)
}


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# ================= 5. 仿真逻辑 =================

def simulate_sequence(room_list, start_node_name):
    curr_pos = exits[start_node_name]
    curr_time = 0
    full_path = [curr_pos]
    logs = []

    for rid in room_list:
        door_pos = rooms_info[rid]['door']

        # 1. 走廊移动
        d = dist(curr_pos, door_pos)
        curr_time += d / BASE_SPEED  # 走廊里假设还能跑，或者也可以减速，这里暂设为正常
        curr_pos = door_pos
        full_path.append(curr_pos)

        # 2. 房间内搜寻计算
        # 这里不仅算时间，还算当时的平均R，用于画图
        sweep_time, avg_r_during_sweep = calculate_sweep_duration_and_path_stats(curr_time)

        # 3. 生成房间内的具体路径
        room_path = generate_search_path_coords(rid, avg_r_during_sweep)
        full_path.extend(room_path)

        # 更新时间
        curr_time += sweep_time
        logs.append(
            f"R{rid}: Start@{curr_time - sweep_time:.1f}s, Dur:{sweep_time:.1f}s (Avg Vis:{avg_r_during_sweep:.1f}m)")

    return curr_time, full_path, logs


def solve_best_strategy():
    best_time = float('inf')
    best_res = None

    # 演示用：左边 Res1，右边 Res2
    perms_a = list(permutations([1, 4, 2]))
    perms_b = list(permutations([3, 6, 5]))

    for pa in perms_a:
        for pb in perms_b:
            t1, p1, l1 = simulate_sequence(pa, 'Left')
            t2, p2, l2 = simulate_sequence(pb, 'Right')
            total = max(t1, t2)
            if total < best_time:
                best_time = total
                best_res = (t1, p1, l1, t2, p2, l2, pa, pb)
    return best_res


# ================= 6. 绘图 =================

def plot_results(res_data):
    t1, path1, log1, t2, path2, log2, _, _ = res_data
    fig, ax = plt.subplots(figsize=(14, 10))  # 画布大一点

    # 画房间
    for rid, info in rooms_info.items():
        rx, ry, w, h = info['rect']
        rect = patches.Rectangle((rx, ry), w, h, linewidth=2, edgecolor='black', facecolor='#e6e6e6')
        ax.add_patch(rect)
        ax.text(rx + 0.5, ry + h - 0.5, info['label'], fontsize=12, fontweight='bold')
        dx, dy = info['door']
        ax.add_patch(patches.Circle((dx, dy), 0.2, color='brown'))

    # 画走廊
    ax.add_patch(patches.Rectangle((0, 5), 18, 3, linewidth=2, edgecolor='black', facecolor='#f0f0f0', zorder=-1))
    ax.text(9, 6.5, "HALLWAY", ha='center', fontsize=15, color='gray')
    ax.text(-1.5, 6.5, "EXIT L", ha='center', color='red', fontweight='bold')
    ax.text(19.5, 6.5, "EXIT R", ha='center', color='red', fontweight='bold')

    # 画路径 Res1
    px1, py1 = zip(*path1)
    ax.plot(px1, py1, color='blue', marker='.', markersize=2, linestyle='-', linewidth=1.5,
            label=f'Res1 (Left) - {t1:.1f}s')

    # 画路径 Res2
    px2, py2 = zip(*path2)
    ax.plot(px2, py2, color='green', marker='.', markersize=2, linestyle='-', linewidth=1.5,
            label=f'Res2 (Right) - {t2:.1f}s')

    ax.set_xlim(-2, 20)
    ax.set_ylim(-1, 14)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title(f"Optimized Sweep with Smoke-Adaptive Paths (Total: {max(t1, t2):.1f}s)", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    print("-" * 50)
    print(f"FINAL TOTAL TIME: {max(t1, t2):.2f} s")
    print("-" * 50)
    print("Responder 1 (Blue - Left):")
    for l in log1: print(l)
    print("-" * 50)
    print("Responder 2 (Green - Right):")
    for l in log2: print(l)

    # DEBUG Check
    print("-" * 50)
    print("DEBUG MODEL CHECK:")
    t0, _ = calculate_sweep_duration_and_path_stats(0)
    t300, _ = calculate_sweep_duration_and_path_stats(300)
    print(f"Time to sweep at t=0s (Clear): {t0:.2f}s")
    print(f"Time to sweep at t=300s (Smoke): {t300:.2f}s")


best_data = solve_best_strategy()
plot_results(best_data)