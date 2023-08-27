from collections import Counter

def convert_by_vocab(vocab, tokens):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for token in tokens:
        output.append(vocab[token])
    return output

class FreqVocab(object):
    """Runs end-to-end tokenziation."""

    def __init__(self):
        # layout of the  ulary
        # item_id based on freq
        # special token
        # user_id based on nothing
        self.counter = Counter()
        self.frequency = []

    def update(self, eoa2seq):
        for eoa in eoa2seq.keys():
            seq = eoa2seq[eoa]
            self.counter[eoa] = len(seq)
            self.counter.update(map(lambda x:x[0], seq))

    def generate_vocab(self):
        self.token_count = len(self.counter.keys())
        self.special_tokens = ["[MASK]", "[pad]", '[NO_USE]']
        self.token_to_ids = {}  # index begin from 1
        #first items

        # first special tokens for frequency factorization
        for token in self.special_tokens:
            self.token_to_ids[token] = len(self.token_to_ids) + 1

        # then normal item
        for token, count in self.counter.most_common():
            self.token_to_ids[token] = len(self.token_to_ids) + 1

        # add count
        for token in self.special_tokens:
            self.counter[token] = 0

        self.id_to_tokens = {v: k for k, v in self.token_to_ids.items()}
        self.vocab_words = list(self.token_to_ids.keys())

        id_list = sorted(list(self.token_to_ids.values()))
        for id in id_list:
            token = self.id_to_tokens[id]
            self.frequency.append(self.counter[token]) # used for negative sampling

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.token_to_ids, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.id_to_tokens, ids)

