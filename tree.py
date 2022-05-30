import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import  Draw
import torch
from rdkit import DataStructs
from Chemutils import *
import functools
# 判断两个片段是否为同一个片段
def IsUnit(a, b):
    # 有交集
    if list(set(a)&set(b)) :
        return True
    else:
        return False

def cmp_fragments(x, y):
    return len(x)-len(y)

class  MolNode(object):
    def __init__(self, smarts):
        # print(smarts)
        self.smarts = smarts
        self.mol = Chem.MolFromSmarts(self.smarts)
        self.mol.UpdatePropertyCache(strict=False)
        self.size = self.mol.GetNumAtoms()
        # vid: vacabulary id
        self.vid = ...
        self.nid = ...
        # nid: node_id
        self.neighbor_nid = set()
        self.neighbors = []
        self.isLeaf = False

    def recover(self, original_mol):
        # print(str(self.nid)+":"+str(self.fragment))
        #把每个片段的节点的smarts新编号都标记为叶子节点编号，然后变成新的mol# 再还原
        # 目的：获取更新后的smiles
        fragments = []
        fragments.extend(self.fragment)
        if not self.isLeaf:
            for atom_idx in self.fragment:
                original_mol.GetAtomWithIdx(atom_idx).SetAtomMapNum(self.nid + 1)

        for nei_node in self.neighbors:
            fragments.extend(nei_node.fragment)
            if nei_node.isLeaf: #Leaf node, no need to mark
                continue
            for atom_idx in nei_node.fragment:
                #allow singleton node override the atom mapping
                if atom_idx not in self.fragment or len(nei_node.fragment) == 1:
                    atom = original_mol.GetAtomWithIdx(atom_idx)
                    atom.SetAtomMapNum(nei_node.nid + 1)

        # 将原有的片段，更新为新的mol
        fragments = list(set(fragments))
        label_mol = get_clique_mol(original_mol, fragments)
        self.label = Chem.MolToSmiles(label_mol)
        # if self.label not in self.cands.keys():

        # 再还原
        for atom_idx in fragments:
            original_mol.GetAtomWithIdx(atom_idx).SetAtomMapNum(0)

        if not self.isLeaf:
            for atom in self.mol.GetAtoms():
                atom.SetAtomMapNum(self.nid + 1)

        return self.label

    def assemble(self):

        # 非单个节点
        neighbors = [nei for nei in self.neighbors if nei.size > 1]
        # 由大到小
        neighbors = sorted(neighbors, key=lambda x: x.size, reverse=True)
        # 单个节点
        singletons = [nei for nei in self.neighbors if nei.size == 1]
        # 顺序是：单个节点：由大到小多个节点

        neighbors = singletons + neighbors
        # print(str(self.nid)+" ,len:" + str(self.size) +" ,neighbors:"+str(self.neighbor_nid))
        # for node in self.neighbors:
        #     print(str(node.nid) +";  len:"+str(node.size))

        self.cands = enum_assemble(self, neighbors)
        # if not self.label in list(self.cands.keys()):
        #     print(str(self.nid)+ ":" + self.label)
        # print(str(self.nid) + ":" + str(self.neighbor_nid))
        # if not self.label in self.cands.keys():
        #     print(self.label in self.cands.keys())
        # print(self.label)
        # print(self.cands.keys())

        # if not self.label in list(self.cands.keys()):
        #     print(str(self.isLeaf)+":"+str(self.nid))
            # print(self.label)
            # print(list(self.cands.keys()))
            # print(self.label)
        # self.cands = enum_assemble(self, neighbors)
        # neighbor_nid self_idx neighbor_idx

class MolTree(object):
    # 单原子
    def __init__(self, smiles, if_build = True):
        if smiles == "":
            self.nodes = list()
            return
        # print(smiles)
        # print(str(typ e(smiles))+":"+smiles)
        self.smiles = smiles

        self.mol = Chem.MolFromSmiles(smiles)

        ringInfo =  self.mol.GetRingInfo()


        fragments = ringInfo.AtomRings()

        num_ring = ringInfo.NumRings()
        fragments = [list(fragment) for fragment in list(fragments)]
        # 合并有交集的结构

        # 我知道循环写的很屎，但是python的循环迭代真的蚌埠住了
        for i in range(num_ring):
            if i>=num_ring:
                continue
            for j in range(num_ring):
                if i == j or j >=num_ring:
                    continue
                if IsUnit(fragments[i], fragments[j]):
                    fragments.append(list(set(fragments[i]) | set(fragments[j])))
                    fragments.remove(fragments[i])
                    fragments.remove(fragments[j-1])
                    i-=1
                    j-=1
                    num_ring-=1


        # 获取所有的非环内部的键
        ringBonds = ringInfo.BondRings()

        bonds = set(bond.GetIdx() for bond in self.mol.GetBonds())
        for ringBond in ringBonds:
            bonds = bonds - set(ringBond)

        pairs = []
        for bond in bonds:
            startBondIdx = self.mol.GetBondWithIdx(bond).GetBeginAtomIdx()
            endBondIdx = self.mol.GetBondWithIdx(bond).GetEndAtomIdx()
            pairs.append([startBondIdx, endBondIdx])

        singles = []
        self.singles = []
        for i in range(self.mol.GetNumAtoms()):
            num_cnt = 0
            for li in pairs + fragments:
                num_cnt += li.count(i)
            if(num_cnt > 2):
                singles.append([i])
                self.singles.append(i)
        # 合并原子和结构
        self.fragments = singles + pairs + fragments



        for i, fragment in enumerate(self.fragments):
            if min(fragment) == 0:
                self.fragments[0], self.fragments[i] = self.fragments[i], self.fragments[0]
                break

        # print(self.fragments)
        #  获取所有的原子+结构的smarts

        self.smartses = [Chem.rdmolfiles.MolFragmentToSmarts(self.mol, fragment) for fragment in self.fragments]
        self.size = len(self.fragments)


        if(if_build):
            # 为了建树临时做了一个标记
            self.temp_node_mark=[i for i in range(self.size)]
            self.nodes = [MolNode(self.smartses[i]) for i in range(self.size)]
            for i in range(self.size):
                self.nodes[i].nid = i
                self.nodes[i].fragment = self.fragments[i]
            if (self.size == 1):
                self.nodes[0].isLeaf = True
            else:
                self.build_tree(0)

    def build_tree(self, node_id):

        # 已访问
        self.temp_node_mark.remove(node_id)

        # 获取该节点的所有原子
        atoms = self.fragments[node_id]
        neighbor_num = 0
        for atom in atoms:
            # 如果是连接单原子节点（自己是单原子节点，就不可能连接单原子
            if atom in self.singles and len(self.fragments[node_id])>1:
                for node_id_b in range(self.size):
                    if node_id_b == node_id:
                        continue
                    if atom in self.fragments[node_id_b] and len(self.fragments[node_id_b]) == 1:
                        self.nodes[node_id].neighbor_nid.add(node_id_b)
                        self.nodes[node_id].neighbors.append(self.nodes[node_id_b])
                        neighbor_num += 1
                        # 如果该原子没有访问过
                        if node_id_b in self.temp_node_mark:
                            self.build_tree(node_id_b)
                        break
            # 连接的是边or环
            else:
                for node_id_b in range(self.size):
                    if node_id_b == node_id:
                        continue
                    # 如果其它片段内有原子
                    if atom in self.fragments[node_id_b]:
                        self.nodes[node_id].neighbor_nid.add(node_id_b)
                        self.nodes[node_id].neighbors.append(self.nodes[node_id_b])
                        neighbor_num +=1
                        # 如果该原子没有访问过
                        if node_id_b in self.temp_node_mark:
                            self.build_tree(node_id_b)

        if neighbor_num == 1:
            self.nodes[node_id].isLeaf = True


    def set_vocab_id(self, vocab_list):
        mol_vocab=[Chem.RDKFingerprint(x) for x in vocab_list]
    def assemble(self):
        for node in self.nodes:
            node.assemble()
    def recover(self):
        for node in self.nodes:
            node.recover(self.mol)

    def get_vid(self, vocab):
        for node in self.nodes:
            node.vid = vocab.index(node.smarts)
if __name__ == '__main__':
    # smile = "Cc1cc2n(C[C@H](O)CO[C@H](c3ccccc3)c3ccccc3C)c(=O)c3ccccc3n2n1"
    # mol = Chem.MolFromSmiles(smile)
    # tree = MolTree(smile, True)
    # tree.recover()
    # tree.assemble()

    with open("data/test.txt", 'r') as f:
        smiles = [line.strip().split()[0] for line in (f.readlines())]
    for smile in smiles:
        tree = MolTree(smile, True)
        tree.recover()
        tree.assemble()
    # mol = Chem.MolFromSmiles(smile)
    # #
    # # # 获取所有的结构
    # ssr = Chem.GetSymmSSSR(mol)
    # num_ring = len(ssr)
    # print(num_ring)
    #
    # fragments = list(ssr)
    # # 合并有交集的结构
    # for ring_a in ssr:
    #     for ring_b in ssr:
    #         if IsUnit(ring_a, ring_b):
    #             ssr.remove(ring_a)
    #             ssr.remove(ring_b)
    #             ssr.append(list(set(ring_a) | set(ring_b)))
    # fragments = ssr
    # print(fragments)