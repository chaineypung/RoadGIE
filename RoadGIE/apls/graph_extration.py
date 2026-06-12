from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
import numpy as np
import cv2
from .sknw import build_sknw
from itertools import tee


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def flatten(l):
    return [item for sublist in l for item in sublist]

def preprocess(img, thresh):
    img = (img > thresh).astype(np.bool_)
    remove_small_objects(img, 300)  
    remove_small_holes(img, 300)  
    return img

def graph2lines(G):
    node_lines = []
    edges = list(G.edges())
    if len(edges) < 1:
        return []
    prev_e = edges[0][1]
    current_line = list(edges[0])
    added_edges = {edges[0]}
    for s, e in edges[1:]:
        if (s, e) in added_edges:
            continue
        if s == prev_e:
            current_line.append(e)
        else:
            node_lines.append(current_line)
            current_line = [s, e]
        added_edges.add((s, e))
        prev_e = e
    if current_line:
        node_lines.append(current_line)
    return node_lines

def line_points_dist(line1, pts):
    return np.cross(line1[1] - line1[0], pts - line1[0]) / np.linalg.norm(line1[1] - line1[0])

def make_skeleton(img, thresh=0.3, fix_borders=False):
    replicate = 5
    clip = 2
    rec = replicate + clip

    if fix_borders:
        img = cv2.copyMakeBorder(img, replicate, replicate, replicate, replicate, cv2.BORDER_REPLICATE)
    img = preprocess(img, thresh)  
    ske = skeletonize(img).astype(np.uint16) 
    if fix_borders:
        ske = ske[rec:-rec, rec:-rec]
        ske = cv2.copyMakeBorder(ske, clip, clip, clip, clip, cv2.BORDER_CONSTANT, value=0)
    return ske

def add_direction_change_nodes(pts, s, e, s_coord, e_coord):
    if len(pts) > 3:
        ps = pts.reshape(pts.shape[0], 1, 2).astype(np.int32)
        approx = 4 
        ps = cv2.approxPolyDP(ps, approx, False)
        ps = np.squeeze(ps, 1)
        st_dist = np.linalg.norm(ps[0] - s_coord)
        en_dist = np.linalg.norm(ps[-1] - s_coord)
        if st_dist > en_dist:
            s, e = e, s
            s_coord, e_coord = e_coord, s_coord
        ps[0] = s_coord
        ps[-1] = e_coord
    else:
        ps = np.array([s_coord, e_coord], dtype=np.int32)
    return ps

def build_graph(img, thresh=0.5, fix_borders=True):
    ske = make_skeleton(img, thresh, fix_borders)
    if ske is None:
        return []
    G = build_sknw(ske, multi=True)
    node_lines = graph2lines(G)
    if not node_lines:
        return []
    node = G.nodes()
    deg = dict(G.degree())
    terminal_points = [i for i, d in deg.items() if d == 1]

    terminal_lines = {}
    visual_point_edge = []
    for w in node_lines:
        for s, e in pairwise(w):
            vals = flatten([[v] for v in G[s][e].values()])
            for ix, val in enumerate(vals):
                s_coord, e_coord = node[s]['o'], node[e]['o'] 
                pts = val.get('pts', [])
                if s in terminal_points:
                    terminal_lines[s] = (s_coord, e_coord)
                if e in terminal_points:
                    terminal_lines[e] = (e_coord, s_coord)

                ps = add_direction_change_nodes(pts, s, e, s_coord, e_coord) 

                if len(ps.shape) < 2 or len(ps) < 2:
                    continue
                if len(ps) == 2 and np.all(ps[0] == ps[1]):
                    continue
                visual_point_edge.append([ps]) 

    return visual_point_edge