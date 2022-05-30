from pytorch_lightning import LightningDataModule
from tree import *
from graph import *
from encoder import *
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from functools import partial

def SmilesToTreeGraph(smiles, vocab, max_node):
    # print(smiles)
    tree = MolTree(smiles)
    graph = TreeGraph(tree, max_node, vocab)
    return graph

def myCollater(smileses, vocab, max_node):
    # print(smileses)
    batched_data = [SmilesToTreeGraph(smiles, vocab, max_node) for smiles in smileses]
    return batched_data
class myDataModule(LightningDataModule):
    def __init__(
            self,
            vocab_path: str = "./vocab.txt",
            train_path: str = "./data/test.txt",
            valid_path: str = "./data/test1.txt",
            num_workers: int = 0,
            batch_size: int = 4,
            max_node : int = 40,
            ):
        super(myDataModule, self).__init__()

        self.vocab_path = vocab_path
        self.train_path = train_path
        self.valid_path = valid_path

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab = ...
        self.dataset_train = ...
        self.dataset_valid = ...
        self.dataset_test = ...
        # self.max_node = max_node

        with open(self.vocab_path, 'r') as f:
            self.vocab = [line.strip().split()[0] for line in (f.readlines())]

        with open(self.train_path, 'r') as f:
            self.dataset_train = [line.strip().split()[0] for line in f.readlines()]

        with open(self.valid_path, 'r') as f:
            self.dataset_valid = [line.strip().split()[0] for line in f.readlines()]
    def train_dataloader(self):
        # print(len(self.dataset_train))
        loader = DataLoader(
            self.dataset_train,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            # collate_fn = partial(myCollater, vocab = self.vocab, max_node = self.max_node)
        )
        # print('len(train_dataloader)', len(loader))
        return loader
    def val_dataloader(self):
        loader = DataLoader(
            self.dataset_valid,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            # collate_fn = partial(myCollater, vocab = self.vocab, max_node = self.max_node)
        )
        return loader
    def test_dataloader(self):
        # loader = DataLoader(
        #     [i for i in range(1000)],
        #     batch_size=1,
        #     shuffle=False,
        #     num_workers=self.num_workers,
        # )
        print("ss")
        loader = DataLoader(
            self.dataset_valid,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            # collate_fn = partial(myCollater, vocab = self.vocab, max_node = self.max_node)
        )
        # print('len(test_dataloader)', len(loader))
        return loader
    # def val_dataloader(self):
    #     loader = DataLoader(
    #         self.dataset_valid,
    #         batch_size = self.batch_size,
    #         num_workers = self.num_workers,
    #         collate_fn=partial(myCollater, vocab=self.vocab, max_node=self.max_node)
    #     )
    #     print('len(val_dataloader)', len(loader))
    #     return loader
    #
    # def test_dataloader(self):
    #     loader = DataLoader(
    #         self.dataset_test,
    #         batch_size = self.batch_size,
    #         num_workers = self.num_workers,
    #         collate_fn=partial(myCollater, vocab=self.vocab, max_node=self.max_node)
    #     )
    #     print('len(test_dataloader)', len(loader))
    #     return loader
if __name__ == '__main__':

    # vocab_path = "D:/本科/大四下/毕设/icml18-jtnn-master/data/mydata/myvocab.txt"
    # train_path = "D:/本科/大四下/毕设/icml18-jtnn-master/data/mydata/train.txt"
    # with open(vocab_path, 'r') as f:
    #     vocab = [line.strip().split()[0] for line in (f.readlines())]
    #
    # with open(train_path, 'r') as f:
    #     dataset_train = [line.strip().split()[0] for line in f.readlines()]
    # print(len(dataset_train))
    # batch_size = 16
    # max_node = 30
    # num_workers = 0
    #
    # loader = DataLoader(
    #     dataset_train,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     collate_fn=partial(myCollater, vocab=vocab, max_node=max_node)
    # )
    # print('len(train_dataloader)',len(loader))
    # dm = MyDataModule()
    # loader = dm.val_dataloader
    # for batch_data in (loader):
    #     # print(str(i)+":")
    #     print(batch_data)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--dataset_path", dest="dataset_path")
    # parser.add_argument("-v", "--vocab_path", dest="vocab_path")
    # opts = parser.parse_args()
    #
    # vocab_path = "D:/本科/大四下/毕设/icml18-jtnn-master/data/mydata/myvocab.txt"
    # # dataset_path = "D:/本科/大四下/毕设/icml18-jtnn-master/data/mydata/train.txt"
    with open("D:/本科/大四下/毕设/icml18-jtnn-master/data/mydata/myvocab.txt", 'r') as f:
        vocab = [line.strip().split()[0] for line in (f.readlines())]
    graph = SmilesToTreeGraph('', vocab, 30)
    # print(graph.sonEncoding )
    #
    # max_node = 30
    # with open(opts.dataset_path, 'r') as f:
    #     graphes = [SmilesToTreeGraph(line, vocab, max_node) for line in f.readlines()]


    # for i, graph in enumerate(graphes):
    #     unloader = transforms.ToPILImage()
    #     image = graph.treeEncoding.clone()  # clone the Tensor
    #     image = image.squeeze(0)  # remove the fake batch dimension
    #     image = unloader(image)
    #     image.save('example'+str(i)+'.jpg')

    # print(graph[0].treeEncoding)