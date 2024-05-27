from rdkit import Chem
from graph_tool import topology
import numpy as np
import graph_tool as gt



def graph_from_adgacency_matrix(adjacency_matrix, atomicnums):
    adj = np.triu(adjacency_matrix)
    G = gt.Graph(directed=False)
    G.add_edge_list(np.transpose(adj.nonzero()))

    vprop = G.new_vertex_property("short")
    vprop.a = atomicnums
    G.vertex_properties["atomicnum"] = vprop
    return G


def num_vertices(G):
    return G.num_vertices()


def match_graphs(G1, G2):
    maps = topology.subgraph_isomorphism(
        G1,
        G2,
        vertex_label=(G1.vertex_properties["atomicnum"], G2.vertex_properties["atomicnum"],),
        subgraph=False,
    )

    return [np.array(m.a) for m in maps]


possible_bond_type_list = ["AROMATIC", "TRIPLE", "DOUBLE", "SINGLE", "misc"]


def safe_index(l, e):
    try:
        return l.index(e) + 1
    except:
        return len(l)


def safe_index_bond(bond):
    return safe_index(possible_bond_type_list, str(bond.GetBondType()))


def atomGetnum(mol):
    atomnums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atom2bonds = [0 for _ in range(len(atomnums))]
    if len(mol.GetBonds()) > 0:
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_num = safe_index_bond(bond)
            atom2bonds[i] += bond_num
            atom2bonds[j] += bond_num

    for i in range(len(atomnums)):
        atomnums[i] = atomnums[i] * 100 + atom2bonds[i]

    return atomnums


def isomorphic_core(mol):
    try:
        atomnums = atomGetnum(mol)
        adj_mat = Chem.rdmolops.GetAdjacencyMatrix(mol)
        G = graph_from_adgacency_matrix(adj_mat, atomnums)
        return match_graphs(G, G)
    except:
         return [np.arange(mol.GetNumAtoms())]


if __name__ == "__main__":
    import time
    smiles = "CCSc1nnc(NC(=O)CSc2ncccn2)s1"
    mol = Chem.MolFromSmiles(smiles)
    a = time.time()
    result = isomorphic_core(mol)
    print(time.time() - a)
    for res in result:
        print(res)