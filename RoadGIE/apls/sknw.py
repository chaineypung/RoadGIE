import numpy as np
import networkx as nx
from numba import njit


# get neighbors d index
@njit
def neighbors(shape):
    dim = len(shape)
    size = 3 ** dim - 1
    offsets = np.empty((size, dim), dtype=np.int8)
    count = 0
    for i in range(3 ** dim):
        offset = np.empty(dim, dtype=np.int8)
        skip = True
        for d in range(dim):
            offset[d] = (i // (3 ** d)) % 3 - 1
            if offset[d] != 0:
                skip = False
        if skip:
            continue
        offsets[count] = offset
        count += 1
    acc = np.empty(dim, dtype=np.int64)
    acc[-1] = 1
    for i in range(dim - 2, -1, -1):
        acc[i] = acc[i + 1] * shape[i + 1]
    result = np.empty((offsets.shape[0],), dtype=np.int64)
    for i in range(offsets.shape[0]):
        result[i] = 0
        for j in range(dim):
            result[i] += offsets[i][j] * acc[j]
    return result

@njit
def mark(img):  # mark the array use (0, 1, 2)
    nbs = neighbors(img.shape)
    img = img.ravel()
    for p in range(len(img)):  # 遍历img上的每一个点
        if img[p] == 0: continue
        s = 0
        for dp in nbs:         # 遍历每一个道路点的周围3x3的点
            if img[p + dp] != 0: s += 1  # 该为道路的点的3x3区域内有点也为道路目标
        if s == 2:
            img[p] = 1  # 如果该为道路的点的3x3区域内有两个点为道路目标
        else:
            img[p] = 2

@njit
def idx2rc(idx, acc):
    rst = np.zeros((len(idx), len(acc)), dtype=np.int16)
    for i in range(len(idx)):
        for j in range(len(acc)):
            rst[i, j] = idx[i] // acc[j]
            idx[i] -= rst[i, j] * acc[j]
    rst -= 1
    return rst

@njit
def fill(img, p, num, nbs, acc, buf):
    back = img[p]  # 目前为2
    img[p] = num
    buf[0] = p
    cur = 0;
    s = 1;

    while True:
        p = buf[cur]    
        for dp in nbs:  # 遍历当前点周围3x3范围，看有没有和它一样性质的点
            cp = p + dp
            if img[cp] == back:
                img[cp] = num
                buf[s] = cp
                s += 1
        cur += 1
        if cur == s: break  #如果当前点周围3x3范围没有和它一样性质的点，则遍历完周围后就会结束
    return idx2rc(buf[:s], acc)

def trace(img, p, nbs, acc, buf):  # 从一个周围有重要节点的不重要节点开始，到下一个周围有有重要节点的不重要节点结束
    c1 = 0                         # 该不重要节点（1）有个先验，即周围必有两个道路节点，且一个是重要节点（2），另一个不确定
    c2 = 0   
    newp = 0
    cur = 0

    while True:
        buf[cur] = p
        img[p] = 0
        cur += 1
        for dp in nbs:  # 该不重要节点（1）有个先验，即周围必有且仅有两个道路节点，且一个是重要节点（2），另一个不确定
            cp = p + dp
            if img[cp] >= 10:  # 为重要节点
                if c1 == 0:    # 为第一个重要节点
                    c1 = img[cp]
                else:          # 为第二个重要节点
                    c2 = img[cp]
            if img[cp] == 1:   # 为不重要节点
                newp = cp
        p = newp
        if c2 != 0: break   # 直到遇到一个不重要节点（1），其周围也有一个重要道路节点
    return (c1 - 10, c2 - 10, idx2rc(buf[:cur], acc))  # c1和c2分别指示第一个起始重要道路节点，和第二和终止道路节点，边由很多不重要节点表示

def parse_struc(img):
    nbs = neighbors(img.shape)
    acc = np.cumprod((1,) + img.shape[::-1][:-1])[::-1]
    img = img.ravel()
    pts = np.array(np.where(img == 2))[0]
    buf = np.zeros(131072, dtype=np.int64)
    num = 10
    nodes = []
    for p in pts:   # 找出所有原本为2的节点
        nds = fill(img, p, num, nbs, acc, buf)
        num += 1
        nodes.append(nds)  

    edges = []
    for p in pts:  # 遍历每一个重要点（节点）
        for dp in nbs:  # 主要遍历该重要节点的8个方向
            if img[p + dp] == 1: # 该节点周围的非重要道路点
                edge = trace(img, p + dp, nbs, acc, buf)
                edges.append(edge)
    return nodes, edges

def build_graph(nodes, edges, multi=False):
    graph = nx.MultiGraph() if multi else nx.Graph()  # 无向多重图
    for i in range(len(nodes)):
        graph.add_node(i, pts=nodes[i], o=np.int32(nodes[i].mean(axis=0))) # o代表该点集区域的几何中心
    for s, e, pts in edges:
        l = np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum()  # 计算路径长度
        graph.add_edge(s, e, pts=pts, weight=l)  # s和e本质上指示上面节点nodes点集的序号，pts是中间的路径，l为路径长度
    return graph

def buffer(ske):
    buf = np.zeros(tuple(np.array(ske.shape) + 2), dtype=np.uint16)  # [514, 514]全0数组
    buf[tuple([slice(1, -1)] * buf.ndim)] = ske
    return buf

def build_sknw(ske, multi=False):
    buf = buffer(ske)
    mark(buf)  # img=0代表是孤立点，img=1代表该点周围有两个点，可能是连接点，img=2代表该点周围有1个点（作为终点），或2个以上点（作为分叉连接点）
    nodes, edges = parse_struc(buf) # 重要节点和节点簇，以及每一个重要节点或节点簇连接成边
    return build_graph(nodes, edges, multi)