import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from tree import *
from graph import *
from encoder import *
from decoder import *
from pytorch_lightning.loggers import TensorBoardLogger

from data import *
from pytorch_lightning.callbacks import ModelCheckpoint
# def SmilesToTreeGraph(smiles, vocab, max_node):
#     tree = MolTree(smiles)
#     graph = TreeGraph(tree, max_node)
#     return
def buildMolTree(smile, vocab):
    tree = MolTree(smile)
    tree.get_vid(vocab)
    tree.recover()
    tree.assemble()
    return tree


class myModel(pl.LightningModule):
    def __init__(self,
                 vocab,
                 max_node: int = 50,
                 max_atom: int =150,
                 max_depth: int=30,
                 hidden_size: int=100,
                 latent_size: int=56,
                 result_path: str ="result1.txt",
                 valid_path: str ="valid.txt"):
        super().__init__()
        vocab_size = int(len(vocab))
        self.result_path = result_path
        self.valid_path = valid_path
        self.max_node = max_node
        self.vocab = vocab
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.max_atom = max_atom
        self.max_depth = max_depth

        self.graphEncoder = GraphEncoder(hidden_size, max_depth, max_node)
        self.treeEncoder = TreeEncoder(vocab_size, hidden_size, max_depth)

        self.T_mean = nn.Linear(hidden_size, latent_size)
        self.T_var = nn.Linear(hidden_size, latent_size)
        self.G_mean = nn.Linear(hidden_size, latent_size)
        self.G_var = nn.Linear(hidden_size, latent_size)

        self.treeDecoder = TreeDecoder(vocab, hidden_size, latent_size)
        self.graphDecoder = GraphDecoder(hidden_size, latent_size)
        self.loss_stop =  nn.BCEWithLogitsLoss(reduction='sum')
        self.loss_label = nn.CrossEntropyLoss(reduction='sum')
        self.loss_graph = nn.CrossEntropyLoss(reduction='sum')
    def rsample(self, z_vecs, W_mean, W_var):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs))  # Following Mueller et al.
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean)
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def forward(self, batched_data):
        # smiles 转 graph
        batched_tree = [buildMolTree(smiles, self.vocab) for smiles in batched_data]
        batched_treeGraph = [TreeGraph(tree, self.max_node, len(self.vocab)) for tree in batched_tree]
        batched_atomGraph = [AtomGraph(tree, self.max_atom) for tree in batched_tree]
        # 编码
        batched_hidden_graph = [self.graphEncoder(atomGraph) for atomGraph in batched_atomGraph]
        batched_hidden_graph = torch.stack(batched_hidden_graph)

        batched_treeEncoder_res = [self.treeEncoder(treeGraph) for treeGraph in batched_treeGraph]
        batched_hidden_tree = []
        batched_tree_message = []
        for treeEncoder_res in batched_treeEncoder_res:
            batched_hidden_tree.append(treeEncoder_res[0])
            batched_tree_message.append(treeEncoder_res[1])
        batched_hidden_tree = torch.stack(batched_hidden_tree)
        batched_tree_message = torch.stack(batched_tree_message)

        # 采样
        z_tree, tree_kl = self.rsample(batched_hidden_tree, self.T_mean, self.T_var)
        z_graph, graph_kl = self.rsample(batched_hidden_graph, self.G_mean, self.G_var)
        # 解码
        batched_treeDecoderRes = [self.treeDecoder(z_tree[i], treeGraph) for i, treeGraph in enumerate(batched_treeGraph)]
        batched_scores = [self.graphDecoder(z_graph[i], tree, batched_tree_message[i]) for i, tree in enumerate(batched_tree)]

        return batched_treeDecoderRes, batched_scores

    def training_step(self, batched_data, batch_idx):
        batched_treeDecoderRes, batched_scores=self(batched_data)
        loss_t = 0
        batch_size = len(batched_treeDecoderRes)
        for real_stops, pred_stops, real_labels, pred_labels in batched_treeDecoderRes:
            stop_loss = self.loss_stop(pred_stops, real_stops)
            label_loss = self.loss_label(pred_labels, real_labels)
            loss_t += stop_loss + label_loss
        # print(loss_t)
        loss_g = 0
        # print("batched_scores " + str(batched_scores))
        for scores in batched_scores:
            for score in scores:
                # print(score)
                loss_g += self.loss_graph(score.unsqueeze(0), torch.zeros(1, dtype=int))

        loss_t /= batch_size
        loss_g /= batch_size
        loss = loss_g + loss_t
        # log = {'train_loss_g': loss_g, 'train_loss_t': loss_t, 'train_loss': loss}
        # self.log_dict(log, prog_bar=True,  logger=True)
        self.log('train_loss_g', loss_g,  on_step=True, prog_bar=True,  logger=True)
        self.log('train_loss_t', loss_t, on_step=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, logger=True)
        return loss

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=1e-3)
        return opt

    def test_step(self,batched_data, batch_idx):
        # smiles 转 graph
        tree = buildMolTree(batched_data[0], self.vocab)
        treeGraph = TreeGraph(tree, self.max_node, len(self.vocab))
        atomGraph = AtomGraph(tree, self.max_atom)
        # 编码
        hidden_graph = self.graphEncoder(atomGraph)

        hidden_tree, tree_message= self.treeEncoder(treeGraph)

        # 采样
        z_tree, tree_kl = self.rsample(hidden_tree, self.T_mean, self.T_var)
        z_mol, graph_kl = self.rsample(hidden_graph, self.G_mean, self.G_var)

        pred_tree = self.treeDecoder.decode(z_tree, self.max_node)

        # if pred_tree.size == 0: return None
        if pred_tree.size == 1:
            # print("单词分子：")
            pred_mol = Chem.MolFromSmarts(pred_tree.nodes[0].smarts)
            for atom in pred_mol.GetAtoms():
                atom.SetAtomMapNum(0)
            pred_smiles = Chem.MolToSmiles(Chem.MolFromSmarts(Chem.MolToSmiles(pred_mol)))

        else:
            pred_treeGraph = TreeGraph(pred_tree, self.max_node, len(self.vocab))
            _, tree_mess = self.treeEncoder(pred_treeGraph)
            pred_smiles = self.graphDecoder.decode(z_mol, pred_tree, tree_mess)
            # print("多词分子：")

        print("pred_smiles  " + str(pred_smiles))
        return pred_smiles
    def test_epoch_end(self, output_results):
        with open(self.valid_path, 'w') as f:
            for smiles in output_results:
                if smiles!=None:
                    f.write(smiles + '\n')
        return output_results

    # def test_step(self, batched_data, batch_idx):
    #     z_tree = torch.randn(self.latent_size)
    #     z_mol = torch.randn(self.latent_size)
    #
    #     pred_tree = self.treeDecoder.decode(z_tree, self.max_node)
    #
    #     # if pred_tree.size == 0: return None
    #     if pred_tree.size == 1:
    #         # print("单词分子：")
    #         pred_mol = Chem.MolFromSmarts(pred_tree.nodes[0].smarts)
    #         for atom in pred_mol.GetAtoms():
    #             atom.SetAtomMapNum(0)
    #         pred_smiles = Chem.MolToSmiles(Chem.MolFromSmarts(Chem.MolToSmiles(pred_mol)))
    #
    #     else:
    #         pred_treeGraph = TreeGraph(pred_tree, self.max_node, len(self.vocab))
    #         _, tree_mess = self.treeEncoder(pred_treeGraph)
    #         pred_smiles = self.graphDecoder.decode(z_mol, pred_tree, tree_mess)
    #         # print("多词分子：")
    #
    #     print("pred_smiles  "+ str(pred_smiles))
    #     return pred_smiles
    #
    # def test_epoch_end(self, output_results):
    #     with open(self.result_path, 'w') as f:
    #         for smiles in output_results:
    #             if smiles!=None:
    #                 f.write(smiles + '\n')
    #     return output_results

if __name__ == '__main__':
    with open("vocab.txt", 'r') as f:
        vocab = [line.strip().split()[0] for line in (f.readlines())]


    metric = 'train_loss'
    dirpath = f'/lightning_logs/checkpoints'
    model = myModel.load_from_checkpoint("./lightning_logs/version_0/checkpoints/epoch=2-step=749.ckpt", vocab=vocab)

    checkpoint_callback = ModelCheckpoint(
        monitor=metric,
        dirpath=dirpath,
        auto_insert_metric_name = True,
        save_top_k=5,
        save_last=True,
        mode='min',
        every_n_train_steps = 50
    )
    dm = myDataModule()
    # logger = TensorBoardLogger("tb_logs", name="my_model")

    # trainer = pl.Trainer()
    trainer = pl.Trainer(max_epochs=3, log_every_n_steps =1, check_val_every_n_epoch=0)
    trainer.callbacks.append(checkpoint_callback)
    # trainer.fit(model=model, datamodule=dm)
    # result = trainer.test(model, datamodule=dm)
    result = trainer.test(model, datamodule=dm)
    # print("result "+str(result))