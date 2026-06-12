import numpy as np
import networkx as nx
from itertools import combinations
from typing import List
from scipy.spatial import KDTree
from .graph_extration import build_graph
import os

def build_graph_from_path_list(path_list: List[np.ndarray]) -> nx.Graph:
    G = nx.Graph()
    for path in path_list:
        for i in range(len(path[0]) - 1):
            p1 = tuple(path[0][i])
            p2 = tuple(path[0][i + 1])
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            G.add_edge(p1, p2, weight=dist)
    return G

def align_points_to_graph(points: List[tuple], target_graph: nx.Graph, radius: float = 5.0):
    if len(target_graph.nodes) == 0:
        return []

    tree = KDTree(list(target_graph.nodes))
    aligned = []
    for pt in points:
        dist, idx = tree.query(pt)
        if dist <= radius:
            nearest = tree.data[idx]
            aligned.append(tuple(nearest))
    return aligned

def compute_apls_one_direction(source_paths: List[np.ndarray], source_graph: nx.Graph,
                               target_graph: nx.Graph, radius: float = 8.0) -> float:
    total_sim = 0.0
    count = 0

    for path in source_paths:
        if len(path[0]) < 2:
            continue
        aligned_points = align_points_to_graph([tuple(p) for p in path[0]], target_graph, radius=radius)
        if len(aligned_points) < 2:
            continue
        for s, t in combinations(aligned_points, 2):
            try:
                s_src = tuple(path[0][0])
                t_src = tuple(path[0][1])
                L_src = nx.shortest_path_length(source_graph, s_src, t_src, weight='weight')
            except nx.NetworkXNoPath:
                continue
            try:
                L_tgt = nx.shortest_path_length(target_graph, s, t, weight='weight')
                diff = abs(L_src - L_tgt) / L_src
                sim = 1.0 - min(1.0, diff)
            except nx.NetworkXNoPath:
                sim = 0.0
            total_sim += sim
            count += 1

    return total_sim / count if count > 0 else 0.0

# def compute_bidirectional_apls(pred_mask, gt_mask, radius: float = 5.0) -> float:
#     pred_mask = pred_mask.squeeze(0).squeeze(0).cpu().numpy()
#     gt_mask = gt_mask.squeeze(0).squeeze(0).cpu().numpy()
#
#     pred_edges = build_graph(pred_mask)
#     gt_edges = build_graph(gt_mask)
#
#     gt_graph = build_graph_from_path_list(gt_edges)
#     pred_graph = build_graph_from_path_list(pred_edges)
#
#     apls_gt_to_pred = compute_apls_one_direction(gt_edges, gt_graph, pred_graph, radius)
#     apls_pred_to_gt = compute_apls_one_direction(pred_edges, pred_graph, gt_graph, radius)
#
#     return (apls_gt_to_pred + apls_pred_to_gt) / 2

def compute_bidirectional_apls(gt_edges, pred_edges, radius: float = 5.0) -> float:
    # 删除这两行：原代码错误地将列表当作张量处理
    # pred_mask = pred_mask.squeeze(0).squeeze(0).cpu().numpy()
    # gt_mask = gt_mask.squeeze(0).squeeze(0).cpu().numpy()

    # 直接使用传入的边列表构建图（无需再调用build_graph，因为已经提前处理过）
    gt_graph = build_graph_from_path_list(gt_edges)
    pred_graph = build_graph_from_path_list(pred_edges)

    # 计算双向APLS
    apls_gt_to_pred = compute_apls_one_direction(gt_edges, gt_graph, pred_graph, radius)
    apls_pred_to_gt = compute_apls_one_direction(pred_edges, pred_graph, gt_graph, radius)

    return (apls_gt_to_pred + apls_pred_to_gt) / 2


# if __name__ == "__main__":
#     pred_mask_path = "./apls_test/pred_masks/"
#     gt_mask_path = "./apls_test/masks/"
#
#     name_list = os.listdir(pred_mask_path)
#     apls = 0
#     for name in name_list:
#         pred_k, pred_v = build_graph(pred_mask_path, name)
#         gt_k, gt_v = build_graph(gt_mask_path, name)
#         apls_score = compute_bidirectional_apls(gt_v, pred_v, radius=8.0)
#         print(f"Bidirectional APLS Score: {apls_score:.4f}")
#         apls = apls + apls_score
#     print(apls / len(name_list))

if __name__ == "__main__":
    import cv2  # 新增：导入cv2用于读取PNG图像

    # 1. 配置数据路径（可根据你的实际路径修改）
    pred_mask_path = "./apls_test/pred_masks/"
    gt_mask_path = "./apls_test/masks/"

    # 2. 获取文件名列表（确保预测与真值文件名一致）
    name_list = os.listdir(pred_mask_path)
    apls_total = 0  # 变量名修改：避免与函数名apls冲突
    valid_count = 0  # 新增：统计有效图像数量（避免除以0）

    # 3. 遍历每张图计算APLS
    for name in name_list:
        # 跳过非PNG文件（如隐藏文件）
        if not name.endswith(".png"):
            continue

        # 3.1 拼接完整路径
        pred_img_path = os.path.join(pred_mask_path, name)
        gt_img_path = os.path.join(gt_mask_path, name)

        # 3.2 读取PNG图像（单通道模式）
        # 注意：若你的掩码图是RGB格式，需改为cv2.IMREAD_GRAYSCALE后再转二值
        pred_img = cv2.imread(pred_img_path, cv2.IMREAD_UNCHANGED)  # 读取单通道PNG
        gt_img = cv2.imread(gt_img_path, cv2.IMREAD_UNCHANGED)

        # 3.3 检查图像是否读取成功
        if pred_img is None or gt_img is None:
            print(f"警告：未找到图像 {name}，跳过")
            continue

        # 3.4 调用build_graph处理图像（传递图像数据，而非路径）
        # 原错误代码：pred_k, pred_v = build_graph(pred_mask_path, name)
        pred_edges = build_graph(pred_img)  # 正确：传递图像数据
        gt_edges = build_graph(gt_img)

        # 3.5 检查是否成功提取边（避免空图报错）
        if not pred_edges or not gt_edges:
            print(f"警告：图像 {name} 未提取到拓扑边，跳过")
            continue

        # 3.6 计算APLS值
        # 注意：compute_bidirectional_apls参数需为"边列表"（pred_edges/gt_edges）
        apls_score = compute_bidirectional_apls(gt_edges, pred_edges, radius=8.0)
        print(f"图像 {name} 的双向APLS得分：{apls_score:.4f}")

        # 3.7 累积得分与计数
        apls_total += apls_score
        valid_count += 1

    # 3.8 计算并输出平均APLS
    if valid_count > 0:
        avg_apls = apls_total / valid_count
        print(f"\n所有有效图像的平均APLS得分：{avg_apls:.4f}")
    else:
        print("\n无有效图像用于计算APLS")
