
# import lmdb

import numpy as np
from torch.utils.data import DataLoader, Dataset


def list_to_bytes(l):
    return np.array(l).tobytes()


def list_from_bytes(string, dtype=np.int):
    return np.frombuffer(string, dtype=dtype)


class Preprocessing():

    def __init__(self, lowercase=True):
        self.lowercase = lowercase

    def transform(self, sentences):
        """
        sentences: list(str) 
        output: list(str)
        """
        return [s.lower() for s in sentences]

 
class CharVectorizer():
    def __init__(
        self, maxlen=10, padding='pre', truncating='pre', 
        alphabet="""abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/| #$%ˆ&*˜‘+=<>()[]{}"""):
        
        self.alphabet = alphabet
        self.maxlen = maxlen
        self.padding = padding
        self.truncating = truncating

        self.char_dict = {'_pad_': 0, '_unk_': 1, ' ': 2}
        for i, k in enumerate(self.alphabet, start=len(self.char_dict)):
            self.char_dict[k] = i

    def transform(self,sentences):
        """
        sentences: list of string
        list of review, review is a list of sequences, sequences is a list of int
        """
        sequences = []

        for sentence in sentences:
            seq = [self.char_dict.get(char, self.char_dict["_unk_"]) for char in sentence]
            
            if self.maxlen:
                length = len(seq)
                if self.truncating == 'pre':
                    seq = seq[-self.maxlen:]
                elif self.truncating == 'post':
                    seq = seq[:self.maxlen]

                if length < self.maxlen:

                    diff = np.abs(length - self.maxlen)

                    if self.padding == 'pre':
                        seq = [self.char_dict['_pad_']] * diff + seq

                    elif self.padding == 'post':
                        seq = seq + [self.char_dict['_pad_']] * diff
            sequences.append(seq)                

        return sequences        
    
    def get_params(self):
        params = vars(self)
        return params


# class TupleLoader(Dataset):

#     def __init__(self, path=""):
#         self.path = path

#         self.env = lmdb.open(path, max_readers=opt.nthreads, readonly=True, lock=False, readahead=False, meminit=False)
#         self.txn = self.env.begin(write=False)

#     def __len__(self):
#         return list_from_bytes(self.txn.get('nsamples'.encode()))[0]

#     def __getitem__(self, i):
#         xtxt = list_from_bytes(self.txn.get(('txt-%09d' % i).encode()), np.int)
#         lab = list_from_bytes(self.txn.get(('lab-%09d' % i).encode()), np.int)[0]
#         return xtxt, lab
