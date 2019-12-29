import argparse
from tqdm import tqdm
import re


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='smiles files to extend vocab with', required=True, type=str)
    parser.add_argument('-s', help='sep to use', default=' ', type=str)
    parser.add_argument('-c', help='column where smiles is', default=0, type=int)
    parser.add_argument('--header', action='store_true')

    return parser.parse_args()


def smi_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return tokens

def get_vocab(loc='data/vocab.txt'):
    with open("data/vocab.txt", 'r') as fin:
        vocab = {v.strip() : k for k,v in enumerate(fin.readlines())}
    return vocab


if __name__ == '__main__':
    args = get_args()
    if 't' in args.s:
        args.s = '\t'

    with open("data/vocab.txt", 'r') as fin:
        vocab = {v :k for k,v in enumerate(fin.readlines())}

    counter = len(vocab)
    with open("data/vocab.txt", 'a') as fout:
        with open(args.i, 'r') as fin:
            for cnt, line in tqdm(enumerate(fin)):
                if cnt == 0 and args.header:
                    continue
                try:
                    tokens = smi_tokenizer(line.split(args.s)[args.c])
                except AssertionError:
                    print(line.split(args.s)[args.c], cnt, line)
                    exit()
                for tok in tokens:
                    if tok not in vocab:
                        vocab[tok] = counter
                        counter += 1
                        fout.write(tok + "\n")

    print("Finished computing vocabulary.")



