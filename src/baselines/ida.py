import itertools

from src.ds.equivalence_class import read_equivalence_class


def skeleton_neighbors(spec):
    neighbors = {v: set() for v in spec.vertices}
    for src, dst in spec.directed_edges:
        neighbors[src].add(dst)
        neighbors[dst].add(src)
    for v1, v2 in spec.undirected_edges:
        neighbors[v1].add(v2)
        neighbors[v2].add(v1)
    return neighbors


def directed_parents(spec, node):
    if node not in set(spec.vertices):
        raise ValueError("{} is not in equivalence class vertices".format(node))
    return {src for src, dst in spec.directed_edges if dst == node}


def undirected_neighbors(spec, node):
    if node not in set(spec.vertices):
        raise ValueError("{} is not in equivalence class vertices".format(node))
    neighbors = set()
    for v1, v2 in spec.undirected_edges:
        if v1 == node:
            neighbors.add(v2)
        elif v2 == node:
            neighbors.add(v1)
    return neighbors


def is_clique(nodes, neighbors):
    nodes = tuple(nodes)
    for left, right in itertools.combinations(nodes, 2):
        if right not in neighbors[left]:
            return False
    return True


def ida_adjustment_sets(spec, treatment, outcome=None):
    parents = directed_parents(spec, treatment)
    possible = undirected_neighbors(spec, treatment)
    if outcome is not None:
        parents.discard(outcome)
        possible.discard(outcome)
    possible = sorted(possible)
    neighbors = skeleton_neighbors(spec)

    adjustment_sets = []
    for size in range(len(possible) + 1):
        for subset in itertools.combinations(possible, size):
            candidate = frozenset(parents.union(subset))
            if is_clique(subset, neighbors):
                adjustment_sets.append(tuple(sorted(candidate)))

    seen = set()
    unique = []
    for candidate in adjustment_sets:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def outcome_is_undirected_neighbor(spec, treatment, outcome):
    return outcome in undirected_neighbors(spec, treatment)


def ida_adjustment_sets_from_file(path, treatment, outcome=None):
    return ida_adjustment_sets(read_equivalence_class(path), treatment, outcome=outcome)
