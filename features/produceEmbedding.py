import argparse
import shlex
from gensim.models import Word2Vec
from smiles import smi_tokenizer, get_vocab
from tqdm import tqdm


def get_args(st=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='smiles files to extend vocab with', required=True, type=str)
    parser.add_argument('-s', help='sep to use', default=' ', type=str)
    parser.add_argument('-c', help='column where smiles is', default=0, type=int)
    parser.add_argument('--header', action='store_true')
    parser.add_argument('-w', help='workers to use', type=int, default=8)
    if st is None:
        return parser.parse_args()
    else:
        return shlex.split(shlex.split(st))


data_path = '/vol/ml/candle_aesp/databases/'
choices = {
    'ENAMINE': ('--header -c 0 -s "\t" -i tmp', ['ENAMIN/Enamine_REAL_full_smiles_Part_01.smiles',
                                                 # 'ENAMIN/Enamine_REAL_full_smiles_Part_02.smiles',
                                                 'ENAMIN/Enamine_REAL_full_smiles_Part_03.smiles',
                                                 # 'ENAMIN/Enamine_REAL_full_smiles_Part_04.smiles'
                                                 ]),
    'ZINC': ('-c 0 -s " " -i tmp', ['ZINC/6_p0.smi', 'ZINC/6_p1.smi', 'ZINC/6_p2.smi', 'ZINC/6_p3.smi']),
}


def make_generator(args):
    vocab = get_vocab()

    for k, _ in vocab.items():
        yield k

    with open("data/vocab.txt", 'a') as fout:
        with open(args.i, 'r') as fin:
            for cnt, line in tqdm(enumerate(fin)):
                if cnt == 0 and args.header:
                    continue
                tokens = smi_tokenizer(line.split(args.s)[args.c])
                yield tokens

    for source, (source_args, files) in choices.items():
        args = get_args(source_args)
        for file in files:
            with open(data_path + file, 'r') as fin:
                for cnt, line in tqdm(enumerate(fin)):
                    if cnt == 0 and args.header:
                        continue
                    tokens = smi_tokenizer(line.split(args.s)[args.c])
                    yield tokens


if __name__ == '__main__':
    args = get_args()
    vocab = get_vocab()
    embedding_size = len(vocab)


    model = Word2Vec(size=96, window=8, min_count=1, workers=args.w, alpha=0.025, min_alpha=0.025)

    model.build_vocab(make_generator(args))

    for epoch in range(10):
        model.train(make_generator(args), total_examples=model.corpus_count, epochs=1)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
        model.init_sims(replace=True)

    for key, val in model.wv.vocab.items():
        print(key)  # this is the word
    print("vocab length", len(model.wv.vocab))
