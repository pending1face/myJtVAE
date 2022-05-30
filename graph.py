import rdkit
import rdkit.Chem as Chem
import torch
from tree import *
ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']
ATOM_FEATURE_DIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FEATURE_DIM = 5 + 6
class TreeGraph(object):
   def __init__(self, molTree, n, vocab_size):
      self.n = n
      self.molTree = molTree
      self.size = self.molTree.size
      # nxn:depth, nid
      self.mask = torch.zeros([n, n],dtype=torch.int)
      # nxn: nid, nid[fa, son]
      self.sonEncoding = torch.zeros([n, n],dtype=torch.int)


      # n x vocab_size
      self.nodeFeature = torch.zeros([n, vocab_size])
      self.vids = torch.zeros([n],dtype= torch.long)
      # int
      self.depth = 0
      self.start_build()
      # print(self.root_nid)
   def start_build(self):
      self.mark_nid = torch.zeros(self.molTree.size)
      self.buidGraph(0, 0)

   def buidGraph(self, nid, dep):
      # 已访问
      self.mark_nid[nid] = 1

      self.depth = max(self.depth, dep + 1)


      self.nodeFeature[nid, self.molTree.nodes[nid].vid] = 1
      # self.vids[nid] = self.molTree.nodes[nid].vid
      # 建造它的子节点们
      for neighbor_nid in self.molTree.nodes[nid].neighbor_nid:
         if self.mark_nid[neighbor_nid] == 0:
            self.sonEncoding[nid, neighbor_nid] = 1
            self.buidGraph(neighbor_nid, dep + 1)
            self.mask[nid, neighbor_nid] = 1
            self.mask[neighbor_nid, nid] = 1


def onek_encoding_unk(x, allowable_set):
   if x not in allowable_set:
      x = allowable_set[-1]
   return map(lambda s: x == s, allowable_set)


def GetBondFeature(bond):
   bt = bond.GetBondType()
   stereo = int(bond.GetStereo())
   fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
   fstereo = list(onek_encoding_unk(stereo, [0, 1, 2, 3, 4, 5]))
   return torch.Tensor(fbond + fstereo)
def GetAtomFeature(atom):
   return torch.Tensor(list(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST))
                       + list(onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]))
                       + list(onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]))
                       + list(onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3]))
                       + list([atom.GetIsAromatic()]))
class AtomGraph(object):
   def __init__(self, molTree, n):
      # self.molTree = molTree.
      self.mol = molTree.mol
      self.size = self.mol.GetNumAtoms()
      # n x ATOM_FEATURE_DIM
      # print(ATOM_FEATURE_DIM)
      self.n = n
      self.atomFeature = torch.zeros(n, ATOM_FEATURE_DIM)
      for nid, atom in enumerate(self.mol.GetAtoms()):
         self.atomFeature[atom.GetIdx(), :]=GetAtomFeature(atom)

      # [nid, nid, bond_no]
      self.bondEncoding = torch.zeros(n, n)
      #
      self.bondFeature = torch.zeros(n, n, BOND_FEATURE_DIM)

      for bid, bond in enumerate(self.mol.GetBonds()):
         a1 = bond.GetBeginAtom().GetIdx()
         a2 = bond.GetEndAtom().GetIdx()
         self.bondEncoding[a1, a2] = self.bondEncoding[a2, a1] = 1
         self.bondFeature[a1, a2, :] = self.bondFeature[a2, a1, :] =GetBondFeature(bond)

      self.fragments = molTree.fragments

