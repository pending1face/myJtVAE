import rdkit
import rdkit.Chem as Chem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
import threading
import torch

MST_MAX_WEIGHT = 100
MAX_NCAND = 2000

def sanitize(mol):
    try:
        smarts = Chem.MolToSmarts(mol)
        mol = Chem.MolFromSmarts(smarts)
    except Exception as e:
        return None
    return mol
def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol


def get_clique_mol(mol, atoms):
    # try:
    #     smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    # except Exception as e:
    smarts = Chem.MolFragmentToSmarts(mol, atoms)
    new_mol = Chem.MolFromSmarts(smarts)
    new_mol = copy_edit_mol(new_mol).GetMol()
    # new_mol = copy_edit_mol(new_mol)

    # Chem.SanitizeMol(new_mol)  # We assume this is not None
    return new_mol


def atom_equal(a1, a2):
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()

# nei_amap   neighbor_nid: neighbor_atom_idx: my_atom_idx
def attach_mols(ctr_mol, neighbors, prev_nodes, nei_amap):

    prev_nids = [node.nid for node in prev_nodes]
    for nei_node in prev_nodes + neighbors:
        nei_id, nei_mol = nei_node.nid, nei_node.mol
        amap = nei_amap[nei_id]
        # 所有相邻节点里的atom
        for atom in nei_mol.GetAtoms():
            # 如果atom不在self node里
            if atom.GetIdx() not in amap:
                new_atom = copy_atom(atom)
                amap[atom.GetIdx()] = ctr_mol.AddAtom(new_atom)

        # 如果邻居是个atom
        if nei_mol.GetNumBonds() == 0:
            nei_atom = nei_mol.GetAtomWithIdx(0)
            ctr_atom = ctr_mol.GetAtomWithIdx(amap[0])
            ctr_atom.SetAtomMapNum(nei_atom.GetAtomMapNum())
        else:
            for bond in nei_mol.GetBonds():
                a1 = amap[bond.GetBeginAtom().GetIdx()]
                a2 = amap[bond.GetEndAtom().GetIdx()]
                if ctr_mol.GetBondBetweenAtoms(a1, a2) is None:
                    if a1==a2:
                        continue
                    ctr_mol.AddBond(a1, a2, bond.GetBondType())

                elif nei_id in prev_nids:  # father node overrides
                    ctr_mol.RemoveBond(a1, a2)
                    ctr_mol.AddBond(a1, a2, bond.GetBondType())
    return ctr_mol


# local_attach(node.mol, neighbors[:depth+1], prev_nodes, amap)
def local_attach(ctr_mol, neighbors, prev_nodes, amap_list):
    # 复制的的新mol
    ctr_mol = copy_edit_mol(ctr_mol)

    # 需要连接的node
    nei_amap = {nei.nid: {} for nei in prev_nodes + neighbors}

    for nei_id, ctr_atom, nei_atom in amap_list:
        nei_amap[nei_id][nei_atom] = ctr_atom

    # nei_amap   neighbor_nid: neighbor_atom_idx: my_atom_idx
    ctr_mol = attach_mols(ctr_mol, neighbors, prev_nodes, nei_amap)
    # print(ctr_mol.GetNumAtoms())
    return ctr_mol.GetMol()


# cand_amap = enum_attach(node.mol, nei_node, cur_amap, singletons)
# This version records idx mapping between ctr_mol and nei_mol
# amap:nei_id,atom_idx, _
# amap 连接图 nei_nid,atom_idx, atom2_idx
def enum_attach(ctr_mol, nei_node, amap, singletons):
    nei_mol, nei_idx = nei_node.mol, nei_node.nid
    # 每一个原子和邻居连接的可能性
    att_confs = []
    # 邻居是单原子节点的 原子
    black_list = [atom_idx for nei_id, atom_idx, _ in amap if nei_id in singletons]
    # 内部原子。如果该原子已经连接了单原子节点，那么这个原子就不可能连接其它的原子。所以需要剔除。
    ctr_atoms = [atom for atom in ctr_mol.GetAtoms() if atom.GetIdx() not in black_list]

    if nei_mol.GetNumBonds() == 0:  # neighbor singleton
        nei_atom = nei_mol.GetAtomWithIdx(0)
        used_list = [atom_idx for _, atom_idx, _ in amap]
        for atom in ctr_atoms:
            if atom_equal(atom, nei_atom) and atom.GetIdx() not in used_list:
                new_amap = amap + [(nei_idx, atom.GetIdx(), 0)]
                att_confs.append(new_amap)

    elif nei_mol.GetNumBonds() == 1:  # neighbor is a bond
        bond = nei_mol.GetBondWithIdx(0)
        bond_val = int(bond.GetBondTypeAsDouble())
        b1, b2 = bond.GetBeginAtom(), bond.GetEndAtom()

        for atom in ctr_atoms:
            # Optimize if atom is carbon (other atoms may change valence)

            # if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() < bond_val:
            #     continue

            if atom_equal(atom, b1):
                new_amap = amap + [(nei_idx, atom.GetIdx(), b1.GetIdx())]
                att_confs.append(new_amap)
            elif atom_equal(atom, b2):
                new_amap = amap + [(nei_idx, atom.GetIdx(), b2.GetIdx())]
                att_confs.append(new_amap)
    else:
        # intersection is an atom
        for a1 in ctr_atoms:
            for a2 in nei_mol.GetAtoms():
                if atom_equal(a1, a2):
                    # Optimize if atom is carbon (other atoms may change valence)
                    # if a1.GetAtomicNum() == 6 and a1.GetTotalNumHs() + a2.GetTotalNumHs() < 4:
                    #     continue
                    new_amap = amap + [(nei_idx, a1.GetIdx(), a2.GetIdx())]
                    att_confs.append(new_amap)
    return att_confs


# Try rings first: Speed-Up
def enum_assemble(node, neighbors, prev_nodes=[], prev_amap=[]):
    # print(Chem.MolToSmiles(node.mol))
    all_attach_confs = []
    # 单个原子的nid
    singletons = [nei_node.nid for nei_node in neighbors + prev_nodes if nei_node.size== 1]

    # 遍历
    def search(cur_amap, depth):

        if len(all_attach_confs) > MAX_NCAND:
            return
        # 已经遍历完所有的邻居了
        if depth == len(neighbors):
            all_attach_confs.append(cur_amap)
            return
        # print(str(str(neighbors[depth].nid))+" : "+Chem.MolToSmiles(neighbors[depth].mol))
        nei_node = neighbors[depth]
        cand_amap = enum_attach(node.mol, nei_node, cur_amap, singletons)
        # print("cand_amap:" + str(cand_amap))
        cand_smiles = set()
        candidates = []
        for i, amap in enumerate(cand_amap):
            cand_mol = local_attach(node.mol, neighbors[:depth + 1], prev_nodes, amap)
            cand_mol = sanitize(cand_mol)
            if cand_mol is None:
                continue

            smiles = Chem.MolToSmiles(cand_mol)
            if smiles in cand_smiles:
                continue
            cand_smiles.add(smiles)
            candidates.append(amap)
        # print("cand_smiles: " + str(cand_smiles))
        # print("candidates:" + str(candidates))
        if len(candidates) == 0:
            return

        for new_amap in candidates:
            search(new_amap, depth + 1)

    search(prev_amap, 0)
    cand_smiles = set()
    # candidates = []
    candidates = {}

    # aroma_score = []
    for amap in all_attach_confs:
        cand_mol = local_attach(node.mol, neighbors, prev_nodes, amap)
        # try:
        #     cand_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cand_mol))
        # except:
        #
        #     continue
        smiles = Chem.MolToSmiles(cand_mol)
        if smiles in cand_smiles or check_singleton(cand_mol, node, neighbors) == False:
            continue
        cand_smiles.add(smiles)
        candidates[smiles] = amap
        # aroma_score.append(check_aroma(cand_mol, node, neighbors))

    return candidates


def check_singleton(cand_mol, ctr_node, nei_nodes):
    rings = [node for node in nei_nodes + [ctr_node] if node.mol.GetNumAtoms() > 2]
    singletons = [node for node in nei_nodes + [ctr_node] if node.mol.GetNumAtoms() == 1]
    if len(singletons) > 0 or len(rings) == 0: return True

    n_leaf2_atoms = 0
    for atom in cand_mol.GetAtoms():
        nei_leaf_atoms = [a for a in atom.GetNeighbors() if not a.IsInRing()]  # a.GetDegree() == 1]
        if len(nei_leaf_atoms) > 1:
            n_leaf2_atoms += 1

    return n_leaf2_atoms == 0


    # Only used for debugging purpose


def dfs_assemble(cur_mol, global_amap, fa_amap, cur_node, fa_node):
    fa_nid = fa_node.nid if fa_node is not None else -1
    prev_nodes = [fa_node] if fa_node is not None else []

    children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
    neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
    neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
    singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
    neighbors = singletons + neighbors

    cur_amap = [(fa_nid, a2, a1) for nid, a1, a2 in fa_amap if nid == cur_node.nid]
    cands = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)

    cand_smiles, cand_amap = zip(*cands)
    label_idx = cand_smiles.index(cur_node.label)
    label_amap = cand_amap[label_idx]

    for nei_id, ctr_atom, nei_atom in label_amap:
        if nei_id == fa_nid:
            continue
        global_amap[nei_id][nei_atom] = global_amap[cur_node.nid][ctr_atom]

    cur_mol = attach_mols(cur_mol, children, [], global_amap)  # father is already attached
    for nei_node in children:
        if not nei_node.is_leaf:
            dfs_assemble(cur_mol, global_amap, label_amap, nei_node, cur_node)

