def build_indexed_graph(edgelist_path: str):
    """
    Build an undirected NetworkX graph from the edge list and
    return (G, node_id_to_idx), where node_id_to_idx maps original
    node IDs (as in the edgelist) to integer indices compatible
    with the embedding matrix.
    """
    df = load_edge_list(edgelist_path)

    if df.shape[1] < 2:
        raise ValueError(
            f"Expected at least 2 columns in edge list, but got shape {df.shape}"
        )

    # Usa automaticamente as duas primeiras colunas como endpoints
    src_col, dst_col = df.columns[:2]
    print(f"Using columns '{src_col}' and '{dst_col}' as edge endpoints.")

    src = df[src_col].to_numpy()
    dst = df[dst_col].to_numpy()

    # Cria mapeamento de IDs originais -> índice [0..N-1]
    unique_nodes = np.unique(np.concatenate([src, dst]))
    node_id_to_idx = {nid: i for i, nid in enumerate(unique_nodes)}

    # Constrói o grafo com índices já mapeados
    G = nx.Graph()
    mapped_edges = [
        (node_id_to_idx[u], node_id_to_idx[v]) for u, v in zip(src, dst)
    ]
    G.add_edges_from(mapped_edges)

    return G, node_id_to_idx
