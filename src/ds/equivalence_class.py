from dataclasses import dataclass
import itertools
import re

from src.ds.causal_graph import CausalGraph


@dataclass(frozen=True)
class EquivalenceClassSpec:
    vertices: tuple
    directed_edges: tuple
    undirected_edges: tuple


def _normalize_undirected_edge(v1, v2):
    return tuple(sorted((v1, v2)))


def _skeleton_neighbors(vertices, directed_edges, undirected_edges):
    neighbors = {v: set() for v in vertices}
    for src, dst in directed_edges:
        neighbors[src].add(dst)
        neighbors[dst].add(src)
    for v1, v2 in undirected_edges:
        neighbors[v1].add(v2)
        neighbors[v2].add(v1)
    return neighbors


def _unshielded_colliders(vertices, directed_edges, skeleton_neighbors):
    parents = {v: set() for v in vertices}
    for src, dst in directed_edges:
        parents[dst].add(src)

    colliders = set()
    for mid in vertices:
        for left, right in itertools.combinations(sorted(parents[mid]), 2):
            if right not in skeleton_neighbors[left]:
                colliders.add((left, mid, right))
    return colliders


def read_equivalence_class(path):
    mode = None
    vertices = []
    directed_edges = []
    undirected_edges = []

    with open(path) as file:
        try:
            for i, raw_line in enumerate(file, 1):
                line = raw_line.strip()
                if not line:
                    continue

                match = re.match(r"<([A-Z]+)>", line)
                if match:
                    mode = match.group(1)
                    continue

                if mode == "NODES":
                    if not line.isidentifier():
                        raise ValueError("invalid identifier")
                    vertices.append(line)
                    continue

                if mode != "EDGES":
                    raise ValueError("unknown mode")

                if "<->" in line:
                    raise ValueError("bidirected edges are not supported in equivalence class files")
                if "->" in line:
                    src, dst = map(str.strip, line.split("->"))
                    directed_edges.append((src, dst))
                    continue
                if "--" in line:
                    v1, v2 = map(str.strip, line.split("--"))
                    undirected_edges.append(_normalize_undirected_edge(v1, v2))
                    continue
                raise ValueError("invalid edge type")
        except Exception as exc:
            raise ValueError(f"Error parsing line {i}: {exc}: {line}")

    vertex_set = set(vertices)
    if len(vertex_set) != len(vertices):
        raise ValueError("duplicate vertices are not allowed")

    normalized_directed = []
    seen_directed = set()
    for src, dst in directed_edges:
        if src not in vertex_set or dst not in vertex_set:
            raise ValueError(f"unknown vertex in directed edge {src}->{dst}")
        if src == dst:
            raise ValueError("self loops are not allowed")
        if (src, dst) in seen_directed:
            continue
        seen_directed.add((src, dst))
        normalized_directed.append((src, dst))

    normalized_undirected = []
    seen_undirected = set()
    for v1, v2 in undirected_edges:
        if v1 not in vertex_set or v2 not in vertex_set:
            raise ValueError(f"unknown vertex in undirected edge {v1}--{v2}")
        if v1 == v2:
            raise ValueError("self loops are not allowed")
        if (v1, v2) in seen_undirected:
            continue
        seen_undirected.add((v1, v2))
        normalized_undirected.append((v1, v2))

    overlap = seen_undirected.intersection({_normalize_undirected_edge(src, dst) for src, dst in seen_directed})
    if overlap:
        raise ValueError(f"edge specified as both directed and undirected: {sorted(overlap)}")

    try:
        CausalGraph(vertices, normalized_directed, [])
    except ValueError as exc:
        raise ValueError(f"directed edges must already be acyclic: {exc}")

    return EquivalenceClassSpec(
        vertices=tuple(vertices),
        directed_edges=tuple(normalized_directed),
        undirected_edges=tuple(normalized_undirected),
    )


def enumerate_dags_in_class(spec, max_dags=None):
    skeleton_neighbors = _skeleton_neighbors(spec.vertices, spec.directed_edges, spec.undirected_edges)
    target_colliders = _unshielded_colliders(spec.vertices, spec.directed_edges, skeleton_neighbors)

    results = []
    undirected_edges = list(spec.undirected_edges)
    for orientation_bits in itertools.product((0, 1), repeat=len(undirected_edges)):
        directed_edges = list(spec.directed_edges)
        for bit, (v1, v2) in zip(orientation_bits, undirected_edges):
            directed_edges.append((v1, v2) if bit == 0 else (v2, v1))

        try:
            candidate = CausalGraph(list(spec.vertices), directed_edges, [])
        except ValueError:
            continue

        colliders = _unshielded_colliders(spec.vertices, directed_edges, skeleton_neighbors)
        if colliders != target_colliders:
            continue

        results.append(candidate)
        if max_dags is not None and len(results) >= max_dags:
            break

    return results
