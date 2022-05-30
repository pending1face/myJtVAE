import functools
import argparse
from tree import MolTree
import rdkit.Chem as Chem

def cmp_smarts(x, y):
    return len(x)-len(y)

def getVocab(data_path,vocab_path):
    vocab_set=set()
    with open(data_path, 'r') as f:
        for line in (f.readlines()):
            smiles = line.strip().split()[0]
            tree = MolTree(smiles, if_build=False)
            for smarts in tree.smartses:
                vocab_set.add(smarts)
    vocab_list=sorted(list(vocab_set), key=functools.cmp_to_key(cmp_smarts))
    with open(vocab_path, 'w') as vocab_file:
        for x in vocab_list:
            print(x)
            vocab_file.write(x + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", dest="dataset", default="data/test.txt")
    parser.add_argument("-v", "--vocab", dest="vocab", default="vocab.txt")
    opts = parser.parse_args()

    getVocab(opts.dataset, opts.vocab)
