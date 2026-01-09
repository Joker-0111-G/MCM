import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import heapq
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ==========================================
# 全局参数 (对应 MATLAB PARAMS)
# ==========================================
class Params:
    V_RESP_STRAIGHT = 1.2
    V_RESP_TURN = 1.0
    V_RESP_STAIRS = 1.1
    V_MIN_SMOKE = 0.2
    R_MIN = 0.5
    R_THRESH_SWEEP = 2.5
    SWEEP_ETA = 0.3
    SWEEP_KAPPA = 1.4
    SWEEP_DELTA_MIN = 0.5
    FLOOR_HEIGHT = 2.5


# ==========================================
# 数据结构定义
# ==========================================
@dataclass
class Zone:
    id: int
    name: str
    rect: List[float]  # [x, y, w, h]
    type: str
    T_crit: float
    R_max: float
    graph_node: str
    z_level: float


@dataclass
class Node:
    name: str
    pos: np.ndarray  # [x, y, z]
    zone_id: int
    type: str


@dataclass
class Edge:
    n1: str
    n2: str
    dist: float
    type: str


class BuildingGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []


@dataclass
class WarehouseGrid:
    grid: np.ndarray
    origin: List[float]
    res: float


@dataclass
class Target:
    name: str
    zone_id: int
    pos_m: np.ndarray
    type: str
    status: str
    zone_idx_for_sweep: float
    pos_grid: Optional[Tuple[int, int]] = None


@dataclass
class Responder:
    id: int
    start_node: str
    time_free: float


@dataclass
class PathSegment:
    # 可以是节点列表(高层)或坐标点列表(低层)
    is_high_level: bool
    data: List[np.ndarray]  # list of [x,y,z] or [x,y]


@dataclass
class FullPath:
    resp_id: int
    segments: List[PathSegment]
    target_name: str
    zone_id: int
    total_time: float


# ==========================================
# 1. 环境构建 (Build Environment)
# ==========================================
def build_environment_v11(res):
    params = Params()
    Z_F1 = 0.0
    Z_F2 = params.FLOOR_HEIGHT

    zones = []

    # 几何尺寸
    W_wh, H_wh = 28, 22
    X_offset = 28
    shop_w, shop_h = 9, 16 / 3
    room_w, room_h = 6, 4
    hall_w = 3

    # Z-1: Warehouse
    zones.append(
        Zone(1, 'Warehouse', [0, 0, W_wh, H_wh], 'Warehouse', 30 * 60, np.linalg.norm([W_wh, H_wh]), 'Warehouse', Z_F1))

    # Z-2,3,4: Shops
    for i in range(3):
        y_s = 3 + i * shop_h
        zones.append(Zone(2 + i, f'Shop {i + 1}', [X_offset, y_s, shop_w, shop_h], 'Shop', 6 * 60,
                          np.linalg.norm([shop_w, shop_h]), f'Shop{i + 1}', Z_F1))

    # Z-5,6,7,8: Apartments
    for i in range(4):
        y_a = 3 + i * room_h
        zones.append(Zone(5 + i, f'Apt {i + 1}', [X_offset, y_a, room_w, room_h], 'Apt', 6 * 60,
                          np.linalg.norm([room_w, room_h]), f'Apt{i + 1}', Z_F2))

    # Z-9: Hallway
    zones.append(Zone(9, 'Hallway', [X_offset + room_w, 3, hall_w, 16], 'Hallway', 6 * 60, np.linalg.norm([hall_w, 16]),
                      'Hallway', Z_F2))

    # Z-10, 11: Stairs
    zones.append(Zone(10, 'Stairs 1', [X_offset, 0, 9, 3], 'Stairs', 6 * 60, np.linalg.norm([9, 3]), 'Stairs1', Z_F1))
    zones.append(Zone(11, 'Stairs 2', [X_offset, 19, 9, 3], 'Stairs', 6 * 60, np.linalg.norm([9, 3]), 'Stairs2', Z_F1))

    # --- 构建 Grid ---
    grid_w = math.ceil(W_wh / res)
    grid_h = math.ceil(H_wh / res)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    def draw_grid_rect(g, x, y, w, h, val, res):
        x1 = max(0, math.floor(x / res))
        y1 = max(0, math.floor(y / res))
        x2 = min(g.shape[1], math.ceil((x + w) / res))
        y2 = min(g.shape[0], math.ceil((y + h) / res))
        g[y1:y2, x1:x2] = val
        return g

    # 墙壁 (设为1)
    grid = draw_grid_rect(grid, 0, 0, W_wh, res, 1, res)  # Bottom
    grid = draw_grid_rect(grid, 0, H_wh - res, W_wh, res, 1, res)  # Top
    grid = draw_grid_rect(grid, 0, 0, res, H_wh, 1, res)  # Left
    grid = draw_grid_rect(grid, W_wh - res, 0, res, H_wh, 1, res)  # Right

    # 障碍物 (设为2)
    wh_obstacles = [
        [3, 16, 8, 2], [3, 12, 2, 4], [6, 13, 2, 2], [3, 4, 8, 3], [3, 8, 3, 2],
        [13, 14, 2, 4], [13, 4, 2, 4], [17, 10, 2, 6], [17, 17, 2, 2], [17, 2, 2, 2],
        [22, 5, 2, 12], [13, 19, 4, 1], [25, 18, 1, 3]
    ]
    for obs in wh_obstacles:
        grid = draw_grid_rect(grid, obs[0], obs[1], obs[2], obs[3], 2, res)

    # 门 (清除障碍，设为0)
    grid = draw_grid_rect(grid, 0, 7, res, 8, 0, res)  # Left Door
    grid = draw_grid_rect(grid, 13, H_wh - res, 2, res, 0, res)  # Top Door
    grid = draw_grid_rect(grid, 13, 0, 2, res, 0, res)  # Bottom Door
    grid = draw_grid_rect(grid, W_wh - res, 0.5, res, 2, 0, res)  # Stairs 1 Door
    grid = draw_grid_rect(grid, W_wh - res, 19.5, res, 2, 0, res)  # Stairs 2 Door

    warehouse_grid = WarehouseGrid(grid, [0, 0], res)

    # --- 构建 Building Graph ---
    G = BuildingGraph()

    def add_node(name, pos, z_id, type_):
        G.nodes[name] = Node(name, np.array(pos), z_id, type_)

    def add_edge(n1, n2, type_):
        node1 = G.nodes[n1]
        node2 = G.nodes[n2]
        if type_ == 'Stairs':
            dist = np.linalg.norm(node1.pos[:2] - node2.pos[:2]) + abs(node1.pos[2] - node2.pos[2])
        else:
            dist = np.linalg.norm(node1.pos - node2.pos)
        G.edges.append(Edge(n1, n2, dist, type_))

    # 节点定义 (完全复刻 MATLAB)
    # Entries
    add_node('Entry_WH_Left', [-1, 11, Z_F1], 0, 'Entry')
    add_node('Entry_WH_Top', [14, 23, Z_F1], 0, 'Entry')
    add_node('Entry_WH_Bottom', [14, -1, Z_F1], 0, 'Entry')
    add_node('Entry_Shop1', [X_offset + shop_w + 1, 3 + shop_h / 2, Z_F1], 0, 'Entry')
    add_node('Entry_Shop2', [X_offset + shop_w + 1, 3 + shop_h * 1.5, Z_F1], 0, 'Entry')
    add_node('Entry_Shop3', [X_offset + shop_w + 1, 3 + shop_h * 2.5, Z_F1], 0, 'Entry')
    add_node('Entry_Stairs1', [X_offset + shop_w + 1, 1.5, Z_F1], 0, 'Entry')
    add_node('Entry_Stairs2', [X_offset + shop_w + 1, 20.5, Z_F1], 0, 'Entry')

    # Rooms
    add_node('Warehouse', [14, 11, Z_F1], 1, 'Room')
    add_node('Shop1', [X_offset + shop_w / 2, 3 + shop_h / 2, Z_F1], 2, 'Room')
    add_node('Shop2', [X_offset + shop_w / 2, 3 + shop_h * 1.5, Z_F1], 3, 'Room')
    add_node('Shop3', [X_offset + shop_w / 2, 3 + shop_h * 2.5, Z_F1], 4, 'Room')
    add_node('Apt1', [X_offset + room_w / 2, 3 + room_h * 0.5, Z_F2], 5, 'Room')
    add_node('Apt2', [X_offset + room_w / 2, 3 + room_h * 1.5, Z_F2], 6, 'Room')
    add_node('Apt3', [X_offset + room_w / 2, 3 + room_h * 2.5, Z_F2], 7, 'Room')
    add_node('Apt4', [X_offset + room_w / 2, 3 + room_h * 3.5, Z_F2], 8, 'Room')
    add_node('Hallway', [X_offset + room_w + hall_w / 2, 11, Z_F2], 9, 'Transition')
    add_node('Stairs1', [X_offset + shop_w / 2, 1.5, Z_F1], 10, 'Transition')
    add_node('Stairs2', [X_offset + shop_w / 2, 20.5, Z_F1], 11, 'Transition')

    # Doors
    add_node('WH_Door_Left', [0, 11, Z_F1], 1, 'Door')
    add_node('WH_Door_Top', [14, 22, Z_F1], 1, 'Door')
    add_node('WH_Door_Bottom', [14, 0, Z_F1], 1, 'Door')
    add_node('WH_Door_Stairs1', [28, 1.5, Z_F1], 1, 'Door')
    add_node('WH_Door_Stairs2', [28, 20.5, Z_F1], 1, 'Door')
    add_node('Shop1_Door', [X_offset + shop_w, 3 + shop_h / 2, Z_F1], 2, 'Door')
    add_node('Shop2_Door', [X_offset + shop_w, 3 + shop_h * 1.5, Z_F1], 3, 'Door')
    add_node('Shop3_Door', [X_offset + shop_w, 3 + shop_h * 2.5, Z_F1], 4, 'Door')
    add_node('Apt1_Door', [X_offset + room_w, 3 + room_h * 0.5, Z_F2], 5, 'Door')
    add_node('Apt2_Door', [X_offset + room_w, 3 + room_h * 1.5, Z_F2], 6, 'Door')
    add_node('Apt3_Door', [X_offset + room_w, 3 + room_h * 2.5, Z_F2], 7, 'Door')
    add_node('Apt4_Door', [X_offset + room_w, 3 + room_h * 3.5, Z_F2], 8, 'Door')

    entry_nodes = [n for n, node in G.nodes.items() if node.type == 'Entry']

    # Edges
    for i in range(len(entry_nodes)):
        for j in range(i + 1, len(entry_nodes)):
            add_edge(entry_nodes[i], entry_nodes[j], 'Hall')

    # Manual Edge Connections
    pairs = [
        ('Entry_WH_Left', 'WH_Door_Left'), ('Entry_WH_Top', 'WH_Door_Top'), ('Entry_WH_Bottom', 'WH_Door_Bottom'),
        ('Entry_Stairs1', 'Stairs1'), ('Entry_Stairs2', 'Stairs2'),
        ('Entry_Shop1', 'Shop1_Door'), ('Entry_Shop2', 'Shop2_Door'), ('Entry_Shop3', 'Shop3_Door'),
        ('Shop1_Door', 'Shop1'), ('Shop2_Door', 'Shop2'), ('Shop3_Door', 'Shop3'),
        ('Warehouse', 'WH_Door_Left'), ('Warehouse', 'WH_Door_Top'), ('Warehouse', 'WH_Door_Bottom'),
        ('Warehouse', 'WH_Door_Stairs1'), ('Warehouse', 'WH_Door_Stairs2'),
        ('WH_Door_Stairs1', 'Stairs1'), ('WH_Door_Stairs2', 'Stairs2'),
        ('Hallway', 'Apt1_Door'), ('Hallway', 'Apt2_Door'), ('Hallway', 'Apt3_Door'), ('Hallway', 'Apt4_Door'),
        ('Apt1_Door', 'Apt1'), ('Apt2_Door', 'Apt2'), ('Apt3_Door', 'Apt3'), ('Apt4_Door', 'Apt4')
    ]
    for u, v in pairs:
        add_edge(u, v, 'Hall')

    add_edge('Stairs1', 'Hallway', 'Stairs')
    add_edge('Stairs2', 'Hallway', 'Stairs')

    return zones, G, warehouse_grid, wh_obstacles, entry_nodes


# ==========================================
# 2. 路径规划算法 (Pathfinding)
# ==========================================

def get_dynamic_speed(zone: Optional[Zone], t_current: float, v_base: float) -> float:
    if zone is None:
        return v_base

    params = Params()
    T_crit = zone.T_crit
    R_max = zone.R_max
    R_min = params.R_MIN
    lambd = 3.0 / T_crit

    R_curr = (R_max - R_min) * math.exp(-lambd * t_current) + R_min

    if R_curr >= 2.0:
        return v_base
    else:
        v_min_smoke = params.V_MIN_SMOKE
        v_dynamic = v_min_smoke + (v_base - v_min_smoke) * (R_curr - R_min) / (2.0 - R_min)
        return max(v_min_smoke, v_dynamic)


def get_edge_time_3d(G: BuildingGraph, zones: List[Zone], u: str, v: str, t_arrival_u: float) -> float:
    params = Params()
    edge = next((e for e in G.edges if (e.n1 == u and e.n2 == v) or (e.n1 == v and e.n2 == u)), None)

    if not edge:
        return float('inf')

    v_base = params.V_RESP_STAIRS if edge.type == 'Stairs' else params.V_RESP_STRAIGHT

    z1 = G.nodes[u].zone_id
    z2 = G.nodes[v].zone_id

    zone_to_check = None
    # Index correction: zone_id 1 maps to zones[0]
    if z1 > 0 and z2 > 0:
        z1_obj = zones[z1 - 1]
        z2_obj = zones[z2 - 1]
        zone_to_check = z1_obj if z1_obj.T_crit < z2_obj.T_crit else z2_obj
    elif z1 > 0:
        zone_to_check = zones[z1 - 1]
    elif z2 > 0:
        zone_to_check = zones[z2 - 1]

    v_dynamic = get_dynamic_speed(zone_to_check, t_arrival_u, v_base)
    return edge.dist / v_dynamic


def find_path_high_level_3d(start_node: str, end_node: str, G: BuildingGraph, zones: List[Zone], t_start: float):
    # Dijkstra
    pq = [(0, start_node)]
    dists = {node: float('inf') for node in G.nodes}
    dists[start_node] = 0
    prev = {node: None for node in G.nodes}

    while pq:
        d, u = heapq.heappop(pq)

        if d > dists[u]:
            continue
        if u == end_node:
            break

        # Neighbors
        neighbors = []
        for e in G.edges:
            if e.n1 == u:
                neighbors.append(e.n2)
            elif e.n2 == u:
                neighbors.append(e.n1)
        neighbors = list(set(neighbors))

        for v in neighbors:
            edge_time = get_edge_time_3d(G, zones, u, v, t_start + d)
            new_dist = d + edge_time
            if new_dist < dists[v]:
                dists[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    if dists[end_node] == float('inf'):
        return [], float('inf')

    # Reconstruct
    path_nodes = []
    curr = end_node
    while curr:
        path_nodes.insert(0, G.nodes[curr])
        curr = prev[curr]

    return path_nodes, dists[end_node]


def find_closest_walkable(pos_m, grid, res):
    # pos_m is [x, y], grid is numpy array
    c = int(round(pos_m[0] / res))
    r = int(round(pos_m[1] / res))

    h, w = grid.shape
    c = max(0, min(w - 1, c))
    r = max(0, min(h - 1, r))

    if grid[r, c] == 0:
        return (c, r)

    # Spiral search
    for sz in range(1, 20):
        for dx in range(-sz, sz + 1):
            for dy in range(-sz, sz + 1):
                if abs(dx) != sz and abs(dy) != sz: continue
                nx, ny = c + dx, r + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if grid[ny, nx] == 0:
                        return (nx, ny)
    return (c, r)


def find_path_astar_warehouse(start_node, end_node, wh_grid: WarehouseGrid, zones: List[Zone], t_start: float,
                              res: float):
    # start_node, end_node are (x, y) indices
    params = Params()
    grid = wh_grid.grid
    h_grid, w_grid = grid.shape

    start_node = tuple(start_node)
    end_node = tuple(end_node)

    def heuristic(a, b):
        dist = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) * res
        return dist / params.V_RESP_STRAIGHT

    open_set = []
    heapq.heappush(open_set, (0, start_node))

    g_score = {start_node: 0}
    came_from = {}

    found = False
    final_time = float('inf')

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    costs = [1, 1, 1, 1, 1.414, 1.414, 1.414, 1.414]

    while open_set:
        current_f, current = heapq.heappop(open_set)

        if current == end_node:
            final_time = g_score[current]
            found = True
            break

        for i, move in enumerate(moves):
            nx, ny = current[0] + move[0], current[1] + move[1]

            if 0 <= nx < w_grid and 0 <= ny < h_grid:
                if grid[ny, nx] != 0: continue

                segment_dist_m = costs[i] * res
                t_curr = t_start + g_score[current]
                v_dyn = get_dynamic_speed(zones[0], t_curr, params.V_RESP_STRAIGHT)

                tentative_g = g_score[current] + (segment_dist_m / v_dyn)

                neighbor = (nx, ny)
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, end_node)
                    heapq.heappush(open_set, (f, neighbor))

    path = []
    if found:
        curr = end_node
        while curr in came_from:
            path.insert(0, list(curr))
            curr = came_from[curr]
        path.insert(0, list(start_node))
        return np.array(path), final_time
    else:
        return np.array([]), float('inf')


# ==========================================
# 3. 扫描逻辑 (Sweep Logic)
# ==========================================

def calculate_sweep_time_v11(zone: Zone, arrival_time: float):
    params = Params()
    T_crit = zone.T_crit
    R_max = zone.R_max
    R_min = params.R_MIN
    lambd = 3 / T_crit

    R_arr = (R_max - R_min) * math.exp(-lambd * arrival_time) + R_min
    v_base = params.V_RESP_STRAIGHT
    v_arr = get_dynamic_speed(zone, arrival_time, v_base)

    W, H = zone.rect[2], zone.rect[3]

    if zone.type == 'Apt' or zone.type == 'Shop':
        door_pos = [W, H / 2]
    else:
        door_pos = [W / 2, 0]

    padding = 0.1
    path_nodes = []

    if R_arr > params.R_THRESH_SWEEP:
        # Perimeter
        c1 = [padding, padding]
        c2 = [W - padding, padding]
        c3 = [W - padding, H - padding]
        c4 = [padding, H - padding]
        path_nodes = [door_pos, c1, c2, c3, c4, door_pos]

        L_peri = sum(
            np.linalg.norm(np.array(path_nodes[i + 1]) - np.array(path_nodes[i])) for i in range(len(path_nodes) - 1))
        T_sweep = L_peri / v_arr

    else:
        # Zigzag
        delta = max(params.SWEEP_DELTA_MIN, params.SWEEP_KAPPA * R_arr)
        y_levels = np.arange(0 + delta / 2, H - delta / 2 + 0.001, delta)  # +0.001 to mimic MATLAB inclusion behavior
        if len(y_levels) == 0: y_levels = [H / 2]
        if y_levels[-1] < (H - delta):
            y_levels = np.append(y_levels, H - delta / 2)

        L_zigzag = 0
        prev_pt = np.array(door_pos)
        path_nodes = [list(door_pos)]

        for k, y in enumerate(y_levels):
            x_left, x_right = padding, W - padding
            if (k + 1) % 2 == 1:
                pts = [[x_left, y], [x_right, y]]
            else:
                pts = [[x_right, y], [x_left, y]]

            for pt in pts:
                curr_pt = np.array(pt)
                L_zigzag += np.linalg.norm(curr_pt - prev_pt)
                path_nodes.append(pt)
                prev_pt = curr_pt

        L_zigzag += np.linalg.norm(prev_pt - np.array(door_pos))
        path_nodes.append(list(door_pos))

        T_sweep = L_zigzag / (v_arr * params.SWEEP_ETA)

    return T_sweep, np.array(path_nodes)


def calculate_warehouse_sweep_v11(start_grid, wh_grid: WarehouseGrid, zones: List[Zone], t_start: float, res: float,
                                  zone_idx, total_zones):
    params = Params()
    grid = wh_grid.grid
    h_grid, w_grid = grid.shape

    # Zone bounds
    zone_width = math.floor(w_grid / total_zones)
    # Python 0-based index logic
    x_start = (int(zone_idx) - 1) * zone_width
    x_end = int(zone_idx) * zone_width if int(zone_idx) < total_zones else w_grid

    sweep_res_m = 1.0
    step_size = max(1, int(round(sweep_res_m / res)))

    path_nodes = [list(start_grid)]
    total_time = 0
    t_curr = t_start
    last_node = start_grid

    direction = 1

    # Iterate Y from bottom (high index) to top (low index) similar to MATLAB
    # But wait, MATLAB loop: (grid_h-1 : -step : 1). Coordinate Y=0 is index 0.
    # Let's follow MATLAB logic: Y index decreases.
    y_range = range(h_grid - 1, 0, -step_size)

    for y in y_range:
        x_range = range(x_start, x_end, 1) if direction == 1 else range(x_end - 1, x_start - 1, -1)

        last_node_in_row = None

        for x in x_range:
            if grid[y, x] == 0:
                curr_node = (x, y)

                if last_node_in_row is None:
                    # Jump
                    p_seg, t_seg = find_path_astar_warehouse(last_node, curr_node, wh_grid, zones, t_curr, res)
                    if len(p_seg) > 0:
                        for p in p_seg[1:]: path_nodes.append(list(p))
                        total_time += t_seg
                        t_curr += t_seg
                else:
                    # Continuous
                    dist_m = np.linalg.norm(np.array(curr_node) - np.array(last_node_in_row)) * res
                    v_dyn = get_dynamic_speed(zones[0], t_curr, params.V_RESP_STRAIGHT)
                    t_seg = dist_m / v_dyn

                    total_time += t_seg
                    t_curr += t_seg
                    path_nodes.append(list(curr_node))

                last_node_in_row = curr_node
                last_node = curr_node
            else:
                last_node_in_row = None

        direction *= -1

    return np.array(path_nodes), total_time


# ==========================================
# 4. 主仿真逻辑 (Main Loop)
# ==========================================
def main():
    # --- Setup ---
    N_RESPONDERS = 5
    N_WAREHOUSE_RESPONDERS = 2
    START_DELAY = 60.0
    RESOLUTION = 0.25

    print(f'--- 启动混合救援仿真 (Python 版) ---')
    print(f'Responders: {N_RESPONDERS}, WH_Parallel: {N_WAREHOUSE_RESPONDERS}, Delay: {START_DELAY}s')

    X_offset = 28
    shop_h = 16 / 3
    room_h = 4

    zones, building_graph, wh_grid, wh_obstacles, all_entry_nodes = build_environment_v11(RESOLUTION)
    print(f'Env built: {len(zones)} zones, {len(all_entry_nodes)} entries')

    # --- Tasks ---
    # Name, ZoneName, [x, y], Type, ZoneIdxForSweep
    raw_tasks = [
        ('Infant', 'Apt 4', [29, 19], 'Injured', math.nan),
        ('Pregnant', 'Apt 1', [29, 4], 'Injured', math.nan),
        ('Shop Person', 'Shop 2', [30, 11.33], 'Normal', math.nan),
        ('WH Search 1 (P1)', 'Warehouse', [0.5, 0.5], 'Search', math.nan),
        ('WH Search 2 (P2)', 'Warehouse', [16, 18], 'Search', math.nan),
        ('Search Apt 2', 'Apt 2', [X_offset + 3, 3 + room_h * 1.5], 'Search', math.nan),
        ('Search Apt 3', 'Apt 3', [X_offset + 3, 3 + room_h * 2.5], 'Search', math.nan),
        ('Search Shop 1', 'Shop 1', [X_offset + 4.5, 3 + shop_h / 2], 'Search', math.nan),
        ('Search Shop 3', 'Shop 3', [X_offset + 4.5, 3 + shop_h * 2.5], 'Search', math.nan),
    ]

    wh_tasks = []
    for z in range(1, N_WAREHOUSE_RESPONDERS + 1):
        wh_tasks.append((f'WH Sweep Zone {z}', 'Warehouse', [14, 11], 'Sweep', z))

    all_task_data = raw_tasks + wh_tasks

    targets = []
    for idx, t_data in enumerate(all_task_data):
        z_id = next((z.id for z in zones if z.name == t_data[1]), 0)
        t_obj = Target(t_data[0], z_id, np.array(t_data[2]), t_data[3], 'Pending', t_data[4])

        if zones[z_id - 1].type == 'Warehouse':
            t_obj.pos_grid = find_closest_walkable(t_obj.pos_m, wh_grid.grid, RESOLUTION)

        targets.append(t_obj)

    # --- Responders ---
    responders = []
    num_entries = len(all_entry_nodes)
    for i in range(N_RESPONDERS):
        entry_idx = i % num_entries
        start_node = all_entry_nodes[entry_idx]
        responders.append(Responder(i + 1, start_node, START_DELAY))
        print(f'  R{i + 1} ready at {start_node}')

    task_queue = list(range(len(targets)))
    all_paths: List[FullPath] = []

    while task_queue:
        current_task_idx = task_queue.pop(0)
        target = targets[current_task_idx]
        target_zone = zones[target.zone_id - 1]

        # Find best responder (min time_free)
        resp = min(responders, key=lambda r: r.time_free)
        t_start = resp.time_free

        print(f'  -> [T={t_start:.0f}s] R{resp.id} assigned: {target.name} ({target_zone.name})')

        # 1. High Level Target Node
        if target_zone.type == 'Warehouse':
            hl_target = target_zone.graph_node
        else:
            hl_target = target_zone.graph_node + '_Door'

        path_high, time_high = find_path_high_level_3d(resp.start_node, hl_target, building_graph, zones, t_start)

        if time_high == float('inf'):
            print(f'     !! ERROR: No path to {hl_target}')
            continue

        path_segments = []
        task_total_time = 0.0
        t_arr_zone = 0.0

        # --- Warehouse Logic ---
        if target_zone.type == 'Warehouse':
            start_node_name = resp.start_node

            if len(path_high) == 1:
                # Already inside logic
                target_pos_m = (np.array(target.pos_grid) - 0.5) * RESOLUTION
                wh_doors = ['WH_Door_Left', 'WH_Door_Top', 'WH_Door_Bottom', 'WH_Door_Stairs1', 'WH_Door_Stairs2']
                best_door = min(wh_doors,
                                key=lambda d: np.linalg.norm(building_graph.nodes[d].pos[:2] - target_pos_m[:2]))

                path_to_door, time_to_door = find_path_high_level_3d(start_node_name, best_door, building_graph, zones,
                                                                     t_start)

                t_arr_zone = t_start + time_to_door
                task_total_time = time_to_door
                path_segments.append(PathSegment(True, [n.pos for n in path_to_door]))
                entry_door_name = best_door
            else:
                # From outside
                entry_door_name = path_high[-2].name  # Penultimate node is the door
                t_arr_zone = t_start + time_high
                task_total_time = time_high
                path_segments.append(PathSegment(True, [n.pos for n in path_high]))

            wh_entry_pos_m = building_graph.nodes[entry_door_name].pos
            wh_entry_grid = find_closest_walkable(wh_entry_pos_m[:2], wh_grid.grid, RESOLUTION)

            sweep_time = 0

            if target.type == 'Search':
                # A* to point
                path_low, time_low = find_path_astar_warehouse(wh_entry_grid, target.pos_grid, wh_grid, zones,
                                                               t_arr_zone, RESOLUTION)
                if len(path_low) == 0:
                    print('     !! Warn: A* failed')
                    continue

                task_total_time += time_low
                path_segments.append(PathSegment(False, path_low))  # Grid coords
                sweep_time = 5.0

            elif target.type == 'Sweep':
                # Zigzag
                zone_idx = target.zone_idx_for_sweep
                h, w = wh_grid.grid.shape
                zone_w_grid = w // N_WAREHOUSE_RESPONDERS
                x_start_g = int((zone_idx - 1) * zone_w_grid + 3)
                sweep_start_g = (x_start_g, h - 3)

                path_to_start, time_to_start = find_path_astar_warehouse(wh_entry_grid, sweep_start_g, wh_grid, zones,
                                                                         t_arr_zone, RESOLUTION)
                if len(path_to_start) == 0: continue

                t_arr_sweep = t_arr_zone + time_to_start
                task_total_time += time_to_start
                path_segments.append(PathSegment(False, path_to_start))

                sweep_path, s_time = calculate_warehouse_sweep_v11(sweep_start_g, wh_grid, zones, t_arr_sweep,
                                                                   RESOLUTION, zone_idx, N_WAREHOUSE_RESPONDERS)

                task_total_time += s_time
                path_segments.append(PathSegment(False, sweep_path))
                sweep_time = 0

            task_total_time += sweep_time

        # --- Apt/Shop Logic ---
        else:
            t_arr_zone = t_start + time_high
            task_total_time = time_high
            path_segments.append(PathSegment(True, [n.pos for n in path_high]))

            sweep_time, sweep_path_loc = calculate_sweep_time_v11(target_zone, t_arr_zone)

            # Convert local sweep path to global
            sweep_path_glob = sweep_path_loc.copy()
            sweep_path_glob[:, 0] += target_zone.rect[0]
            sweep_path_glob[:, 1] += target_zone.rect[1]

            path_segments.append(PathSegment(False, sweep_path_glob))  # Meter coords
            task_total_time += sweep_time

        # Update Responder
        resp.time_free = t_start + task_total_time
        if target_zone.type == 'Warehouse':
            resp.start_node = target_zone.graph_node
        else:
            resp.start_node = hl_target

        target.status = 'Completed'
        all_paths.append(FullPath(resp.id, path_segments, target.name, target.zone_id, task_total_time))
        print(f'     Task done. R{resp.id} free at {resp.time_free:.0f}s')

    # --- Exit Logic ---
    print('--- Calculating Exit Paths ---')
    for resp in responders:
        t_start = resp.time_free
        best_exit = ''
        min_time = float('inf')

        for exit_node in all_entry_nodes:
            _, t_exit = find_path_high_level_3d(resp.start_node, exit_node, building_graph, zones, t_start)
            if t_exit < min_time:
                min_time = t_exit
                best_exit = exit_node

        path_exit, time_exit = find_path_high_level_3d(resp.start_node, best_exit, building_graph, zones, t_start)
        resp.time_free += time_exit

        exit_seg = PathSegment(True, [n.pos for n in path_exit])
        all_paths.append(FullPath(resp.id, [exit_seg], 'Exit', 1, time_exit))
        print(f'  R{resp.id} exits to {best_exit}. Final T={resp.time_free:.1f}s')

    # --- Plotting ---
    plot_simulation_3D(zones, building_graph, wh_obstacles, targets, all_paths, RESOLUTION, N_RESPONDERS)
    plot_2d_floor_plans(zones, building_graph, wh_obstacles, targets, all_paths, RESOLUTION, N_RESPONDERS)
    plt.show()


# ==========================================
# 5. 绘图函数 (Plotting)
# ==========================================

def plot_simulation_3D(zones, graph, wh_obs, targets, paths, res, n_responders):
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('3D Hybrid Path Simulation (Python V11)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    # Colors
    col_floor = (0.9, 0.9, 0.9, 0.8)
    col_wall = (0.1, 0.1, 0.1, 0.05)
    col_obs = (0.6, 0.6, 0.6, 0.6)

    # 1. Draw Environment
    def plot_rect_3d(r, z, h, col, ax):
        x, y, w, height = r
        # Floor
        xx = [x, x + w, x + w, x]
        yy = [y, y, y + height, y + height]
        zz = [z, z, z, z]
        verts = [list(zip(xx, yy, zz))]
        ax.add_collection3d(Poly3DCollection(verts, facecolors=col, edgecolors='k', linewidths=0.1))

        # If walls needed (simplified here as blocks)
        if h > 0:
            # Top
            zz_top = [z + h] * 4
            verts_top = [list(zip(xx, yy, zz_top))]
            ax.add_collection3d(Poly3DCollection(verts_top, facecolors=col, alpha=0.05))

    for z in zones:
        c = col_floor
        if z.type == 'Stairs': c = (0.95, 0.95, 0.8, 0.8)
        plot_rect_3d(z.rect, z.z_level, 0, c, ax)

    # Walls (Abstracted)
    plot_rect_3d([0, 0, 28, 22], 0, 2.5, col_wall, ax)
    plot_rect_3d([28, 0, 9, 22], 0, 2.5, col_wall, ax)
    plot_rect_3d([28, 3, 6, 16], 2.5, 2.5, col_wall, ax)

    for obs in wh_obs:
        plot_rect_3d(obs, 0, 2.5, col_obs, ax)

    # 2. Targets
    for t in targets:
        z_lvl = zones[t.zone_id - 1].z_level
        c = 'r' if t.type == 'Injured' else 'b'
        if t.type in ['Search', 'Sweep']: c = 'g'
        marker = 'o' if t.type in ['Injured', 'Normal'] else 'x'
        ax.scatter(t.pos_m[0], t.pos_m[1], z_lvl, c=c, marker=marker, s=50)
        ax.text(t.pos_m[0], t.pos_m[1], z_lvl + 0.5, t.name, fontsize=8)

    # 3. Paths
    cmap = plt.get_cmap('tab10')

    for p in paths:
        color = cmap(p.resp_id % 10)
        z_base = 0
        if 0 < p.zone_id <= len(zones):
            z_base = zones[p.zone_id - 1].z_level

        curr_zone_type = zones[p.zone_id - 1].type if 0 < p.zone_id <= len(zones) else ''

        for seg in p.segments:
            if seg.is_high_level:
                pts = np.array(seg.data)
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], c=color, linewidth=2, marker='o', markersize=2)
            else:
                pts = np.array(seg.data)
                if curr_zone_type == 'Warehouse':
                    pts = (pts - 0.5) * res  # Convert grid to meter

                zs = np.full(len(pts), z_base)
                ax.plot(pts[:, 0], pts[:, 1], zs, c=color, linewidth=1.5)

    ax.set_xlim(-2, 40)
    ax.set_ylim(-2, 24)
    ax.set_zlim(0, 6)


def plot_2d_floor_plans(zones, graph, wh_obs, targets, paths, res, n_responders):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('2D Floor Plans')

    cmap = plt.get_cmap('tab10')

    # --- Floor 1 ---
    ax1.set_title('Floor 1: Warehouse & Shops')
    ax1.set_aspect('equal')

    for z in zones:
        if z.z_level == 0:
            fc = 'none'
            if z.type == 'Stairs': fc = '#f2f2cc'
            rect = plt.Rectangle((z.rect[0], z.rect[1]), z.rect[2], z.rect[3],
                                 edgecolor='k', facecolor=fc, linewidth=1.5)
            ax1.add_patch(rect)
            if z.type != 'Warehouse':
                ax1.text(z.rect[0] + z.rect[2] / 2, z.rect[1] + z.rect[3] / 2, z.name, ha='center')

    for obs in wh_obs:
        rect = plt.Rectangle((obs[0], obs[1]), obs[2], obs[3], facecolor='gray', alpha=0.5)
        ax1.add_patch(rect)

    # Targets F1
    for t in targets:
        z_lvl = zones[t.zone_id - 1].z_level
        if z_lvl == 0:
            c = 'r' if t.type == 'Injured' else ('b' if t.type == 'Normal' else 'g')
            marker = 'o' if t.type in ['Injured', 'Normal'] else 'x'
            ax1.scatter(t.pos_m[0], t.pos_m[1], c=c, marker=marker)

    # Paths F1
    for p in paths:
        z_lvl = 0
        if 0 < p.zone_id <= len(zones): z_lvl = zones[p.zone_id - 1].z_level
        if p.target_name == 'Exit':
            start_name = p.segments[0].data[0]  # Actually in python data is pos array not name
            # Simplified: Check z of first point
            if p.segments[0].data[0][2] > 0: z_lvl = 2.5

        if z_lvl > 0: continue

        col = cmap(p.resp_id % 10)
        curr_zone_type = zones[p.zone_id - 1].type if 0 < p.zone_id <= len(zones) else ''

        for seg in p.segments:
            pts = np.array(seg.data)
            if not seg.is_high_level and curr_zone_type == 'Warehouse':
                pts = (pts - 0.5) * res

            ax1.plot(pts[:, 0], pts[:, 1], c=col, linewidth=1.5)

    # --- Floor 2 ---
    ax2.set_title('Floor 2: Apartments')
    ax2.set_aspect('equal')
    ax2.set_xlim(26, 40)
    ax2.set_ylim(-2, 24)

    for z in zones:
        if z.z_level > 0:
            fc = 'none'
            if z.type == 'Hallway': fc = '#f2f2f2'
            rect = plt.Rectangle((z.rect[0], z.rect[1]), z.rect[2], z.rect[3],
                                 edgecolor='k', facecolor=fc, linewidth=1.5)
            ax2.add_patch(rect)
            ax2.text(z.rect[0] + z.rect[2] / 2, z.rect[1] + z.rect[3] / 2, z.name, ha='center')

    # Targets F2
    for t in targets:
        z_lvl = zones[t.zone_id - 1].z_level
        if z_lvl > 0:
            c = 'r' if t.type == 'Injured' else 'b'
            marker = 'o' if t.type in ['Injured', 'Normal'] else 'x'
            ax2.scatter(t.pos_m[0], t.pos_m[1], c=c, marker=marker)

    # Paths F2
    for p in paths:
        z_lvl = 0
        if 0 < p.zone_id <= len(zones): z_lvl = zones[p.zone_id - 1].z_level
        if p.target_name == 'Exit':
            if p.segments[0].data[0][2] > 0: z_lvl = 2.5

        if z_lvl == 0: continue

        col = cmap(p.resp_id % 10)
        for seg in p.segments:
            pts = np.array(seg.data)
            ax2.plot(pts[:, 0], pts[:, 1], c=col, linewidth=1.5)


if __name__ == '__main__':
    main()