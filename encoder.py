import torch
import torch.nn as nn
from tree import MolTree, MolNode
import torch.nn.functional as F
from graph import *
MAX_NB = 15
class GraphGRU(nn.Module):

    def __init__(self, vocab_size, hidden_size, depth):
        super(GraphGRU, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.depth = depth

        self.W_z = nn.Linear(vocab_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(vocab_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(vocab_size + hidden_size, hidden_size)

    #
    # messgae:n x n x hidden_size [fa, son, h]
    def forward(self, message, treeGraph):

        # 更新depth，确保每个点都更新到信息
        n = message.size(0)
        unconnect_index = torch.where(treeGraph.mask <= 0)
        for it in range(treeGraph.depth):
            # print(it)
            # 公式 4
            # message; n x n x hidden_size
            # [fa_nid, h hidden_size] n x n x hidden_size
            message_neighbor_sum = message.sum(dim = 1) - message
            # 公式 5
            # nodeFeature: n x vocab_size [nid, vid]
            z_input = torch.cat([torch.repeat_interleave(treeGraph.nodeFeature.unsqueeze(0), repeats=n, dim=0), message_neighbor_sum], dim = 2)
            # nxhidden_size
            z = torch.sigmoid(self.W_z(z_input))


            # 公式6
            # r_1 n x hidden_size
            # r_1 = self.W_r(treeGraph.nodeFeature)
            r_1 = self.W_r(treeGraph.nodeFeature)


            r_2 = self.U_r(message)
            # r n x n x hidden_size
            r = torch.sigmoid(r_1 + r_2)


            gated_h = r * message

            # n x hidden_size
            sum_gated_h = gated_h.sum(dim = 0)

            # message_input (n, vocab_size+hidden_size)
            message_input = torch.cat([treeGraph.nodeFeature, sum_gated_h], dim=1)
            #公式 7
            message_h = torch.tanh(self.W_h(message_input))
            # 公式 8
            message = (1.0 - z) * message_neighbor_sum + z * message_h

            message[unconnect_index] *=0

        return message

class TreeEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size, max_depth):
        super(TreeEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.GRU = GraphGRU(vocab_size, hidden_size, max_depth)
        # self.embedding = nn.embedding(vocab_size, hidden_size)
        # self.outputNN = nn.Sequential(
        #     nn.Linear(2 * hidden_size, hidden_size),
        #     nn.ReLU()
        # )
        self.outputNN = nn.Sequential(
            nn.Linear(vocab_size + hidden_size, hidden_size),
            nn.ReLU()
        )
    def forward(self, treeGraph):
        messages = torch.zeros(treeGraph.n, treeGraph.n, self.hidden_size)
        messages = self.GRU(messages, treeGraph)

        # n x  hidden_size
        messages_neighbor_sum = messages.sum(dim = 1) + messages.sum(dim = 0)

        # n x (hidden_size + vocab_size)
        # embeddinged_nodeFeature = self.embedding(treeGraph.nodeFeature)

        hidden_input = torch.cat([treeGraph.nodeFeature, messages_neighbor_sum], dim = 1)

        hidden_code =  self.outputNN(hidden_input)
        # print(hidden_code[0])
        return hidden_code[0], messages


class GraphEncoder(nn.Module):
    def __init__(self, hidden_size, depth, max_node):
        super(GraphEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.max_node = max_node
        self.W_12 = nn.Linear(ATOM_FEATURE_DIM + BOND_FEATURE_DIM, hidden_size, bias=False)
        self.W_3 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U = nn.Linear(ATOM_FEATURE_DIM + hidden_size, hidden_size)
    
    def forward(self, atomGraph):
        n = atomGraph.n
        # n x n x BOND_FEATURE_DIM
        unconnect_index = torch.where(atomGraph.bondEncoding==0)

        w12_input = torch.cat([torch.repeat_interleave(atomGraph.atomFeature.unsqueeze(0), repeats=n, dim=0), atomGraph.bondFeature],dim = 2)
        # n x n x hidden_size
        w12_output = self.W_12(w12_input)
        message = F.relu(w12_output)
        message[unconnect_index] *= 0
        # print(self.depth)
        for i in range(self.depth-1):
            # message.sum(dim = 1) n x hidden_size
            # n x n x hidden_size
            w3_input = message.sum(dim = 1) - message
            message = F.relu(w12_output + self.W_3(w3_input))
            # print(message)
            message[unconnect_index] *= 0
            message = torch.sigmoid(message)
            # print("message " +str(i) +":"+ str(message))

        # n x hidden_size
        message_neighbor = message.sum(dim = 1)
        # molGraph.atomFeature: n x ATOM_FEATURE_DIM
        # n x hidden_size
        # print("message_neighbor :"+str(message_neighbor))
        atom_hidden = F.relu(self.U(torch.cat([atomGraph.atomFeature, message_neighbor], dim=1)))

        hidden = atom_hidden.sum(dim = 0)/atomGraph.size
        return hidden

