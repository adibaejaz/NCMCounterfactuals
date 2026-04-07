from src.ds import CTF, CTFTerm, CausalGraph, graph_search


valid_atomic_queries = {"y1_dox1", "y0_dox0", "y1_dox0", "y0_dox1"}
atomic_query_complements = {
    "y1_dox1": "y0_dox1",
    "y0_dox1": "y1_dox1",
    "y1_dox0": "y0_dox0",
    "y0_dox0": "y1_dox0",
}


def search_var(G: CausalGraph, var_text):
    treatment = set()
    for V in G.set_v:
        if var_text in V:
            treatment.add(V)

    return treatment


def get_mediators(G: CausalGraph, treatment, outcome):
    possible_mediators = set()
    for V in treatment:
        possible_mediators = possible_mediators.union(graph_search(G, V))

    possible_mediators = possible_mediators.difference(treatment)

    mediators = set()
    for V in possible_mediators:
        if graph_search(G, V, outcome):
            mediators.add(V)

    return mediators


def is_q_id_in_G(graph_name, query_name):
    query_name = query_name.upper()
    id_graphs = {
        "ATE": {"backdoor", "frontdoor", "napkin", "simple", "m", "med", "expl", "zid_a", "gid_a", "gid_b",
                "expl_dox", "expl_xm_dox", "expl_xy_dox", "expl_my", "expl_my_dox"},
        "ETT": {"backdoor", "frontdoor", "simple", "m", "med", "expl",
                "expl_dox", "expl_my", "expl_my_dox"},
        "NDE": {"backdoor", "frontdoor", "napkin", "simple", "m", "med", "expl", "zid_a", "gid_a", "gid_b",
                "expl_dox", "expl_xm_dox", "expl_xy_dox"},
        "CTFDE": {"backdoor", "frontdoor", "simple", "m", "med", "expl", "expl_dox"}
    }

    if query_name not in id_graphs:
        return False

    return graph_name in id_graphs[query_name]


def get_query(graph_name, query_name):
    G = CausalGraph.read("dat/cg/{}.cg".format(graph_name))
    query_name = query_name.upper()
    eval_query = None
    opt_query = None

    treatment = search_var(G, 'X')
    mediators = get_mediators(G, treatment, 'Y')

    treat_dict = {V: 1 for V in treatment}
    notreat_dict = {V: 0 for V in treatment}

    if query_name == "ATE":
        y1dox1 = CTFTerm({'Y'}, treat_dict, {'Y': 1})
        y1dox0 = CTFTerm({'Y'}, notreat_dict, {'Y': 1})
        y0dox1 = CTFTerm({'Y'}, treat_dict, {'Y': 0})
        y0dox0 = CTFTerm({'Y'}, notreat_dict, {'Y': 0})

        py1dox1 = CTF({y1dox1}, set(), name="ATE")
        py1dox0 = CTF({y1dox0}, set(), name="ATE")
        py0dox1 = CTF({y0dox1}, set(), name="ATE")
        py0dox0 = CTF({y0dox0}, set(), name="ATE")

        if len(treatment) == 1:
            eval_query = [[py1dox1, 1], [py1dox0, -1]]
            opt_query = [py1dox1, py0dox0], [py1dox0, py0dox1]
        elif len(treatment) > 1:
            eval_query = py1dox0
            opt_query = py1dox0, py0dox0

    elif query_name == "ETT":
        y1dox1 = CTFTerm({'Y'}, treat_dict, {'Y': 1})
        y1dox0 = CTFTerm({'Y'}, notreat_dict, {'Y': 1})
        y0dox1 = CTFTerm({'Y'}, treat_dict, {'Y': 0})
        y0dox0 = CTFTerm({'Y'}, notreat_dict, {'Y': 0})
        x1 = CTFTerm(treatment, {}, treat_dict)

        py0dox0_x1 = CTF({y0dox0}, {x1}, name="ETT")
        py1dox0_x1 = CTF({y1dox0}, {x1}, name="ETT")
        py0dox1_x1 = CTF({y0dox1}, {x1}, name="ETT")
        py1dox1_x1 = CTF({y1dox1}, {x1}, name="ETT")

        if len(treatment) == 1:
            eval_query = [[py1dox1_x1, 1], [py1dox0_x1, -1]]
            opt_query = [py1dox1_x1, py0dox0_x1], [py0dox1_x1, py1dox0_x1]
        elif len(treatment) > 1:
            eval_query = py1dox0_x1
            opt_query = py1dox0_x1, py0dox0_x1

    elif query_name == "NDE":
        if len(mediators) == 0:
            return get_query(graph_name, "ATE")

        mx0_term = CTFTerm(mediators, {**notreat_dict}, {})
        mx0 = CTF({mx0_term}, set(), name="NDE")
        y1dox1mx0 = CTFTerm({'Y'}, {**treat_dict, "nested": mx0}, {'Y': 1})
        y0dox1mx0 = CTFTerm({'Y'}, {**treat_dict, "nested": mx0}, {'Y': 0})
        y1dox0 = CTFTerm({'Y'}, notreat_dict, {'Y': 1})
        y0dox0 = CTFTerm({'Y'}, notreat_dict, {'Y': 0})

        py0med = CTF({y0dox1mx0}, set(), name="NDE")
        py1med = CTF({y1dox1mx0}, set(), name="NDE")
        py0 = CTF({y0dox0}, set(), name="NDE")
        py1 = CTF({y1dox0}, set(), name="NDE")

        eval_query = [[py1med, 1], [py1, -1]]
        opt_query = [py1med, py0], [py0med, py1]

    elif query_name == "CTFDE":
        if len(mediators) == 0:
            return get_query(graph_name, "ETT")

        mx0_term = CTFTerm(mediators, {**notreat_dict}, {})
        mx0 = CTF({mx0_term}, set(), name="CTFDE")
        y1dox1mx0 = CTFTerm({'Y'}, {**treat_dict, "nested": mx0}, {'Y': 1})
        y0dox1mx0 = CTFTerm({'Y'}, {**treat_dict, "nested": mx0}, {'Y': 0})
        y1dox0 = CTFTerm({'Y'}, notreat_dict, {'Y': 1})
        y0dox0 = CTFTerm({'Y'}, notreat_dict, {'Y': 0})
        x1 = CTFTerm(treatment, {}, treat_dict)

        py0med = CTF({y0dox1mx0}, {x1}, name="CTFDE")
        py1med = CTF({y1dox1mx0}, {x1}, name="CTFDE")
        py0 = CTF({y0dox0}, {x1}, name="CTFDE")
        py1 = CTF({y1dox0}, {x1}, name="CTFDE")

        eval_query = [[py1med, 1], [py1, -1]]
        opt_query = [py1med, py0], [py0med, py1]

    return eval_query, opt_query


def get_atomic_query(graph_name, query_name):
    G = CausalGraph.read("dat/cg/{}.cg".format(graph_name))
    query_name = query_name.lower()

    if query_name not in valid_atomic_queries:
        raise ValueError("unknown atomic query '{}'".format(query_name))

    treatment = search_var(G, 'X')
    if len(treatment) != 1:
        raise ValueError(
            "atomic optimization queries require exactly one treatment variable, got {}".format(
                sorted(treatment)
            )
        )

    treat_dict = {V: 1 for V in treatment}
    notreat_dict = {V: 0 for V in treatment}
    query_specs = {
        "y1_dox1": (treat_dict, {'Y': 1}, "P(Y=1 | do(X=1))"),
        "y0_dox0": (notreat_dict, {'Y': 0}, "P(Y=0 | do(X=0))"),
        "y1_dox0": (notreat_dict, {'Y': 1}, "P(Y=1 | do(X=0))"),
        "y0_dox1": (treat_dict, {'Y': 0}, "P(Y=0 | do(X=1))"),
    }
    do_vals, var_vals, name = query_specs[query_name]
    return CTF({CTFTerm({'Y'}, do_vals, var_vals)}, set(), name=name)


def get_atomic_query_complement(graph_name, query_name):
    query_name = query_name.lower()
    if query_name not in atomic_query_complements:
        raise ValueError("unknown atomic query '{}'".format(query_name))
    return get_atomic_query(graph_name, atomic_query_complements[query_name])


def is_atomic_query_in_G(graph_name):
    G = CausalGraph.read("dat/cg/{}.cg".format(graph_name))
    treatment = search_var(G, 'X')
    return len(treatment) == 1


def get_experimental_variables(graph_name):
    graph_name = graph_name.lower()
    zid_set = {"zid_a", "zid_b", "zid_c"}
    gid_set = {"gid_a", "gid_b", "gid_c", "gid_d"}
    expl_dox_set = {"expl_dox", "expl_xm_dox", "expl_xy_dox", "expl_my_dox"}

    if graph_name in zid_set:
        return [{}, {"Z": 0}, {"Z": 1}]
    elif graph_name in gid_set:
        return [{"X1": 0}, {"X1": 1}, {"X2": 0}, {"X2": 1}]
    elif graph_name in expl_dox_set:
        return [{}, {"X": 0}, {"X": 1}]
    else:
        return [{}]
