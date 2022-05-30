import torch
import rdkit.Chem as Chem
import copy
import torch.nn as nn
from tree import MolTree, MolNode
from graph import TreeGraph
import torch.nn.functional as F
from Chemutils import *
from data import SmilesToTreeGraph
import encoder

def dfs(graph, index):
    # print(index)
    path = []
    len = graph.size(0)
    for i in range(len):
        if graph[index, i] > 0:
            path.append([index, 1])
            path = path + dfs(graph, i)
    path.append([index, 0])
    return path
def getPath(graph, nid):
    path = dfs(graph, 0)
    return path

def can_assemble(node_x, node_next):
    node_x.nid = 1
    node_x.isLeaf = False
    for atom in node_x.mol.GetAtoms():
        atom.SetAtomMapNum(node_x.nid + 1)
    neis = node_x.neighbors + [node_next]

    neighbors = [nei for nei in neis if nei.mol.GetNumAtoms() > 1]
    neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
    singletons = [nei for nei in neis if nei.mol.GetNumAtoms() == 1]
    neighbors = singletons + neighbors
    cands = enum_assemble(node_x, neighbors)
    return len(cands) > 0

def if_atom_equal(atom_a, atom_b):
    if atom_a.GetSymbol() == atom_b.GetSymbol() and atom_a.GetFormalCharge() == atom_b.GetFormalCharge():
        return True
    else:
        return False

def have_same_atom(node_x, node_next):
    for atom_x in node_x.mol.GetAtoms():
        for atom_next in node_next.mol.GetAtoms():
            if if_atom_equal(atom_x, atom_next):
                return True
    return  False

class TreeDecoder(nn.Module):
    def __init__(self, vocab, hidden_size, latent_size):
        super(TreeDecoder, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        vocab_size = self.vocab_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        # GRU
        self.W_z = nn.Linear(vocab_size + hidden_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_r = nn.Linear(vocab_size, hidden_size)
        self.W_h = nn.Linear(vocab_size + hidden_size, hidden_size)


        # Stop Prediction Weights
        # 拓扑预测
        # 公式 11
        #
        # self.x_embedding = nn.Embedding(vocab_size, hidden_size)
        self.W_d12 = nn.Linear(vocab_size + latent_size, hidden_size)
        self.W_d3 = nn.Linear(hidden_size, hidden_size)
        self.U_d = nn.Linear(hidden_size * 2, 1)


        # 标签预测
        # 公式 12
        self.W_l = nn.Linear(hidden_size + latent_size, hidden_size)
        self.U_l = nn.Linear(hidden_size, self.vocab_size)
        #


    def forward(self,  latent_code, realTreeGraph):
        # print(realTreeGraph.molTree.smarts)
        #  node_id -> node_id h
        n = realTreeGraph.n
        hidden = torch.zeros(n, n, self.hidden_size)
        pred_neighbor_encoding = torch.zeros(n, n)

        path = getPath(realTreeGraph.sonEncoding, 0)

        real_stops =[]
        real_labels =[]
        pred_stops = []
        pred_labels = []
        # 预测root
        real_labels.append(realTreeGraph.molTree.nodes[0].vid)
        w_l_input = torch.cat([latent_code, torch.zeros(self.hidden_size)], dim=0)
        pred_label = F.relu(self.U_l(self.W_l(w_l_input)))
        pred_labels.append(pred_label)



        for i, [node_id, real_stop] in enumerate(path):
            # embeddinged_x = self.x_embedding(realTreeGraph.nodeFeature[node_id])
            real_stops.append(real_stop)
            w_d12_input = torch.cat([realTreeGraph.nodeFeature[node_id], latent_code])
            # hidden_size
            w_d12_output = self.W_d12(w_d12_input)

            # hidden_size
            hidden_sum= hidden[:, node_id].sum(dim = 0)

            # hidden_size
            w_d3_output = self.W_d3(hidden_sum)
            pred_stop = self.U_d(F.relu(torch.cat([w_d12_output, w_d3_output], dim = 0)))
            pred_stops.append(pred_stop)

            # 最后一个节点，没有下一个节点
            if i == len(path)-1:
                break


            next_node_id = path[i+1][0]

            # GRU
            pred_neighbor_encoding[node_id, next_node_id] = pred_neighbor_encoding[next_node_id, node_id] = 0

            mask = torch.repeat_interleave(pred_neighbor_encoding[:, node_id].unsqueeze(-1), repeats=self.hidden_size,
                                           dim=1)

            s = (hidden[:, node_id] * mask).sum(dim=0)

            z = torch.sigmoid(self.W_z(torch.cat([realTreeGraph.nodeFeature[node_id], s], dim=0)))
            # hiden_size
            r_1 = self.W_r(realTreeGraph.nodeFeature[node_id])
            # n x hidden_size
            r_2 = self.U_r(hidden[:, node_id] * mask)
            # n x hidden_size
            r = torch.sigmoid(r_1 + r_2)
            input_w_h = torch.cat([realTreeGraph.nodeFeature[node_id], r.sum(dim=0)], dim=0)
            gated_h = torch.tanh(self.W_h(input_w_h))
            new_h = (1 - z) * s + z * gated_h

            hidden[node_id, next_node_id, :] = new_h
            pred_neighbor_encoding[node_id, next_node_id] = pred_neighbor_encoding[next_node_id, node_id] = 1

            # Label Prediction
            if(real_stop == 1):

                real_label = realTreeGraph.molTree.nodes[next_node_id].vid
                pred_label = self.U_l(F.relu(self.W_l(torch.cat([latent_code, hidden[node_id, next_node_id]]))))
                # print(pred_label.size())
                pred_label = torch.softmax(pred_label, dim=0)
                # print("pred_label " + str(pred_label))
                real_labels.append(real_label)
                pred_labels.append(pred_label)


        pred_stops = torch.stack(pred_stops).squeeze(dim=1)
        # pred_stops = torch.sigmoid(pred_stops)
        # print("pred_stop" + str(pred_stops))
        return torch.tensor(real_stops).float(), \
               pred_stops,\
               torch.tensor(real_labels), \
               torch.stack(pred_labels)

    def decode(self, latent_code, n):
        pred_neighbor_encoding = torch.zeros(n, n)
        hidden = torch.zeros(n, n, self.hidden_size)
        pred_nodeFeature =  torch.zeros(n, self.vocab_size)
        pred_molTree = MolTree("", if_build=False)
        stack = []


        init_hiddens = torch.zeros(1, self.hidden_size)
        # latent_code latent_size
        # hidden_size
        # print(latent_code.size())
        w_l_input = torch.cat([latent_code, torch.zeros(self.hidden_size)], dim=0)
        pred_label = F.relu(self.U_l(self.W_l(w_l_input)))
        # print(pred_label.size())
        _, root_vid = torch.max(pred_label, dim=0)
        root_vid = root_vid.item()
        root_node = MolNode(self.vocab[root_vid])
        root_node.nid = 0
        pred_molTree.nodes.append(root_node)

        path = [0]
        for step in range(2*n):
            cur_nid = path[-1]
            node_x = pred_molTree.nodes[cur_nid]

            # 树的大小是否超限额   非单原子节点的邻居是否超限额
            if(len(pred_molTree.nodes)>= n or (node_x.size>1 and len(node_x.neighbors)==2)):
                backtrack = True
            else:
                hidden_sum = hidden[:, cur_nid].sum(dim = 0)
                w_d12_input = torch.cat([pred_nodeFeature[cur_nid], latent_code], dim=0)
                w_d12_output = self.W_d12(w_d12_input)
                w_d3_output = self.W_d3(hidden_sum)

                stop_score = self.U_d( F.relu( torch.cat([w_d12_output, w_d3_output], dim=0) ))
                backtrack = (stop_score <= 0)


            if not backtrack:
                # GRU
                mask = torch.repeat_interleave(pred_neighbor_encoding[:, cur_nid].unsqueeze(-1), repeats=self.hidden_size, dim=1)

                s = (hidden[:, cur_nid] * mask).sum(dim=0)

                z = torch.sigmoid(self.W_z(torch.cat([pred_nodeFeature[cur_nid], s], dim=0)))
                # hiden_size
                r_1 = self.W_r(pred_nodeFeature[cur_nid])
                # n x hidden_size
                r_2 = self.U_r(hidden[:, cur_nid] * mask)
                # n x hidden_size
                r = torch.sigmoid(r_1 + r_2)
                input_w_h = torch.cat([pred_nodeFeature[cur_nid], r.sum(dim=0)], dim=0)
                gated_h = torch.tanh(self.W_h(input_w_h))
                new_h = (1 - z) * s + z * gated_h



                w_l_input = torch.cat([latent_code, new_h], dim=0)
                pred_label = F.relu(self.U_l(self.W_l(w_l_input)))


                _, sort_vid = torch.sort(pred_label, descending=True)

                sort_vid = sort_vid.tolist()
                next_vid = None
                for vid in sort_vid[:10]:
                    node_next = MolNode(self.vocab[vid])
                    if have_same_atom(node_x, node_next) and can_assemble(node_x, node_next):
                        node_next.vid = vid
                        next_vid = vid
                        break


                if next_vid is None:
                    backtrack = True
                else:
                    # print(next_vid)
                    next_nid = len(pred_molTree.nodes)
                    # print(next_nid)
                    hidden[cur_nid,  next_nid, :] = new_h
                    pred_neighbor_encoding[cur_nid, next_nid] = pred_neighbor_encoding[next_nid, cur_nid] = 1
                    pred_nodeFeature[next_nid, next_vid] = 1
                    path.append(next_nid)

                    # 建树
                    pred_molTree.nodes.append(node_next)
                    node_next.nid = next_nid

                    node_x.neighbor_nid.add(next_nid)
                    node_x.neighbors.append(node_next)

                    node_next.neighbor_nid.add(node_x.nid)
                    node_next.neighbors.append(node_x)



            if  backtrack:
                # print("backtrack")
                if len(path) == 1:
                    break
                next_nid = path[-2]
                pred_neighbor_encoding[cur_nid, next_nid] = pred_neighbor_encoding[next_nid, cur_nid] = 0

                mask = torch.repeat_interleave(pred_neighbor_encoding[:, cur_nid].unsqueeze(-1),
                                               repeats=self.hidden_size, dim=1)

                s = (hidden[:, cur_nid] * mask).sum(dim=0)

                z = torch.sigmoid(self.W_z(torch.cat([pred_nodeFeature[cur_nid], s], dim=0)))
                # hiden_size
                r_1 = self.W_r(pred_nodeFeature[cur_nid])
                # n x hidden_size
                r_2 = self.U_r(hidden[:, cur_nid] * mask)
                # n x hidden_size
                r = torch.sigmoid(r_1 + r_2)
                input_w_h = torch.cat([pred_nodeFeature[cur_nid], r.sum(dim=0)], dim=0)
                gated_h = torch.tanh(self.W_h(input_w_h))
                new_h = (1 - z) * s + z * gated_h

                hidden[cur_nid, next_nid] = new_h
                pred_neighbor_encoding[cur_nid, next_nid] = pred_neighbor_encoding[next_nid, cur_nid] = 1
                path.pop()

        pred_molTree.size = len(pred_molTree.nodes)
        return pred_molTree
ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

ATOM_DIM = len(ELEM_LIST) + 6 + 5 + 1
BOND_DIM = 5
MAX_NODE_ATOM = 20

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return map(lambda s: x == s, allowable_set)

def atom_features(atom):
    return torch.Tensor(list(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST))
            + list(onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]))
            + list(onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0]))
            + list([atom.GetIsAromatic()]))

def bond_features(bond):
    bt = bond.GetBondType()
    return torch.Tensor([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()])



def getUnit(a, b):
    c = list(set(a)&set(b))
    if (len(c)>1):
        print("怎么会有两个共同点，快去查查！")
    else:
        return c[0]

def dfsGraphDecoderPath(graph, index, fragments):
    # print(index)
    path = []
    len = graph.size(0)
    for i in range(len):
        if graph[index, i] > 0:
            path.append([i, getUnit(fragments[index], fragments[i])])
            path = path + dfsGraphDecoderPath(graph, i, fragments)
    return path

# nextvist_nid, 父节点已经被用过的atom
def getGraphDecoderPath(sonGraph, fragments):
    path = dfsGraphDecoderPath(sonGraph, 0, fragments)
    return path


class AtomNodeGraph(object):
   def __init__(self, molNode, n):
      # self.molTree = molTree.
      self.mol = molNode.mol
      self.size = self.mol.GetNumAtoms()
      # n x ATOM_FEATURE_DIM
      # print(ATOM_FEATURE_DIM)
      self.n = n
      self.atomFeature = torch.zeros(n, ATOM_DIM)
      for nid, atom in enumerate(self.mol.GetAtoms()):
         self.atomFeature[atom.GetIdx(), :]=atom_features(atom)

      # [nid, nid, bond_no]
      self.bondEncoding = torch.zeros(n, n)
      #
      self.bondFeature = torch.zeros(n, n, BOND_DIM)

      for bid, bond in enumerate(self.mol.GetBonds()):
         a1 = bond.GetBeginAtom().GetIdx()
         a2 = bond.GetEndAtom().GetIdx()
         self.bondEncoding[a1, a2] = self.bondEncoding[a2, a1] = 1
         self.bondFeature[a1, a2, :] = self.bondFeature[a2, a1, :] = bond_features(bond)


class GraphDecoder(nn.Module):
    def __init__(self, hidden_size, latent_size, depth =  MAX_NODE_ATOM, n = MAX_NODE_ATOM):
        super(GraphDecoder, self).__init__()
        self.n = n
        self.depth = depth
        self.hidden_size = hidden_size
        self.Latent2Hidden = nn.Linear(latent_size, hidden_size)
        self.W_a12 = nn.Linear(ATOM_DIM + BOND_DIM, hidden_size, bias = False)
        self.W_a3 = nn.Linear(hidden_size, hidden_size, bias = False)
        self.U_g = nn.Linear(ATOM_DIM + hidden_size, hidden_size)

    def forward(self, latent_code, realTree, tree_message):
        # print(realTree.smiles)
        # nextvist_nid, used_atom_idx
        realTree.recover()
        realTree.assemble()

        nodeGraphs = [AtomNodeGraph(node, self.n) for node in realTree.nodes]
        scores = []
        # latent_code 是nan
        latent_code = self.Latent2Hidden(latent_code)
        # print("latent_code :"+str(latent_code))
        for i, node in enumerate(realTree.nodes):
            if node.size==1 or len(node.cands) <= 1 or node.label not in node.cands.keys():
                continue
            unconnect_index = torch.where(nodeGraphs[i].bondEncoding == 0)
            # w_a12_input n x n x (ATOM_DIM + BOND_DIM)
            w_a12_input = torch.cat(
                [torch.repeat_interleave(nodeGraphs[i].atomFeature.unsqueeze(0), repeats=self.n, dim=0), nodeGraphs[i].bondFeature],
                dim=2)
            w_a12_output = self.W_a12(w_a12_input)
            hidden = []
            # print(len(node.cands.values()))
            for key, cand in node.cands.items():
                cand_message = F.relu(w_a12_output)
                cand_message[unconnect_index] *=0
                # 初始化的时候用掉了一次机会
                for t in range(self.depth -1):
                    cand_message_hat = cand_message.sum(dim = 1) - cand_message
                    for neighbor_nid, atom_idx, _ in cand:
                        cand_message_hat[atom_idx, :] += tree_message[neighbor_nid - 1, node.nid]
                        cand_message_hat[:, atom_idx] += tree_message[neighbor_nid - 1, node.nid]

                    cand_message = F.relu(w_a12_output + self.W_a3(cand_message_hat))
                    cand_message[unconnect_index] *= 0

                # n x hidden_size
                cand_atom_hidden = F.relu(self.U_g(torch.cat([nodeGraphs[i].atomFeature, cand_message.sum(dim =1)], dim = 1)))
                # hidden_size
                cand_hidden = cand_atom_hidden.sum(dim = 0)/node.size
                hidden.append(cand_hidden.clone())
                # print("cand_hidden " + str(cand_hidden))


                # print(cand_hidden.size())
                # print(latent_code.size())



                if(node.label == key):
                    hidden[-1], hidden[0] = hidden[0], hidden[-1]
                # print("score " + str(score))


            hidden = torch.stack(hidden)
            hidden = torch.sigmoid(hidden)
            # print("hidden " + str(hidden))
            score = torch.matmul(
                hidden.unsqueeze(1),
                latent_code.unsqueeze(1)
            ).squeeze()
            score = torch.softmax(score, dim=0)
            # print("score " + str(score))
            # cand_score = (cand_hidden*latent_code).sum()

            # score = torch.tensor(score)
            # print("score " + str(score))
            scores.append(score)
            # print("scores " + str(scores))


        return scores



    def dfs_assemble(self, latent_code, tree_mess, predTree, cur_mol, global_amap,  cur_node, fa_amap, fa_node):
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []
        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid, a2, a1) for nid, a1, a2 in fa_amap if nid == cur_node.nid]
        cands= enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)

        if len(cands) >1:
            # latent_code = self.Latent2Hidden(latent_code)
            nodeGraph = AtomNodeGraph(cur_node, self.n)
            unconnect_index = torch.where(nodeGraph.bondEncoding == 0)
            # w_a12_input n x n x (ATOM_DIM + BOND_DIM)
            w_a12_input = torch.cat(
                [torch.repeat_interleave(nodeGraph.atomFeature.unsqueeze(0), repeats=self.n, dim=0),
                 nodeGraph.bondFeature],
                dim=2)
            w_a12_output = self.W_a12(w_a12_input)
            hidden = []
            # print(len(node.cands.values()))
            for key, cand in cands.items():
                cand_message = F.relu(w_a12_output)
                cand_message[unconnect_index] *= 0
                # 初始化的时候用掉了一次机会
                for t in range(self.depth - 1):
                    cand_message_hat = cand_message.sum(dim=1) - cand_message
                    for neighbor_nid, atom_idx, _ in cand:
                        cand_message_hat[atom_idx, :] += tree_mess[neighbor_nid - 1, cur_node.nid]
                        cand_message_hat[:, atom_idx] += tree_mess[neighbor_nid - 1, cur_node.nid]

                    cand_message = F.relu(w_a12_output + self.W_a3(cand_message_hat))
                    cand_message[unconnect_index] *= 0

                # n x hidden_size
                cand_atom_hidden = F.relu(
                    self.U_g(torch.cat([nodeGraph.atomFeature, cand_message.sum(dim=1)], dim=1)))
                # hidden_size
                cand_hidden = cand_atom_hidden.sum(dim=0) / cur_node.size
                hidden.append(cand_hidden.clone())
                # print("cand_hidden " + str(cand_hidden))

                # print(cand_hidden.size())
                # print(latent_code.size())
                # print("score " + str(score))

            hidden = torch.stack(hidden)
            hidden = torch.sigmoid(hidden)
            # print("hidden " + str(hidden))
            scores = torch.matmul(
                hidden.unsqueeze(1),
                latent_code.unsqueeze(1)
            ).squeeze()
            scores = torch.softmax(scores, dim=0)
            # print("score " + str(score))
            # cand_score = (cand_hidden*latent_code).sum()


        else:
            scores = torch.Tensor([1.0])

        _, cand_idx = torch.sort(scores, descending=True)
        backup_mol = Chem.RWMol(cur_mol)
        pre_mol = cur_mol

        for i in range(cand_idx.numel()):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = list(cands.values())[i]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id, ctr_atom, nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]

            cur_mol = attach_mols(cur_mol, children, [], new_global_amap)  # father is already attached
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None: continue

            has_error = False
            for nei_node in children:
                if nei_node.isLeaf: continue

                tmp_mol, tmp_mol2 = self.dfs_assemble(latent_code, tree_mess, predTree, cur_mol, new_global_amap, nei_node, pred_amap,  cur_node)
                if tmp_mol is None:
                    has_error = True
                    if i == 0: pre_mol = tmp_mol2
                    break
                cur_mol = tmp_mol

            if not has_error: return cur_mol, cur_mol

        return None, pre_mol

    def decode(self, latent_code, predTree, tree_message):
        latent_code = self.Latent2Hidden(latent_code)
        cur_mol = copy_edit_mol(predTree.nodes[0].mol)
        global_amap = [{}] + [{} for node in predTree.nodes]
        global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}
        cur_mol,_ = self.dfs_assemble(latent_code, tree_message, predTree, cur_mol, global_amap, predTree.nodes[0],[], None)

        if cur_mol is None:
            cur_mol = copy_edit_mol(predTree.nodes[0].mol)
            global_amap = [{}] + [{} for node in predTree.nodes]
            global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}
            cur_mol, pre_mol = self.dfs_assemble(latent_code, tree_message, predTree, cur_mol, global_amap,predTree.nodes[0], [], None)
            if cur_mol is None: cur_mol = pre_mol

        if cur_mol is None:
            return None

        cur_mol = cur_mol.GetMol()
        for atom in cur_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        # Chem.kekulize(cur_mol)
        return Chem.MolToSmiles(cur_mol) if cur_mol is not None else None












