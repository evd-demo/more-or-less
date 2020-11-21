
import pickle
import pandas as pd
from collections import OrderedDict as OD
from pandas import concat, DataFrame as DF
from beautifultable import BeautifulTable as BT


class Results(object):
    
    def __init__(self, name=None):
        self.name = name                
        self.time = list()
        self.epoch = list()
        self.train_loss = list()
        self.train_accy = list()
        self.valid_loss = list()
        self.valid_accy = list()
        self.data = {
            'time': self.time,
            'train_loss': self.train_loss,
            'train_accy': self.train_accy,
            'valid_loss': self.valid_loss,
            'valid_accy': self.valid_accy}

    def update(self, args:tuple):
        self.train_loss.append(args[0])
        self.train_accy.append(args[1])
        self.valid_loss.append(args[2])
        self.valid_accy.append(args[3])
        self.epoch.append(args[4])
        self.time.append(args[5])
        return

    # def to_DF(self):
    #     df = pd.DataFrame(self.data)

    def save_pickle(self, path):
        with open(path, 'wb') as result:
            pickle.dump(self.data, result, pickle.HIGHEST_PROTOCOL)   
        return


class EnsembleResults(object):

    def __init__(self, size, name=None):
        self.name = name
        self.data = OD()
        for n in range(1,1+size):
            self.data['net_{}'.format(n)] = Results()
        self.data['ensemble'] = Results()

    def update(self, args:tuple):
        for n, result in self.data.items():
            self.data[n].train_loss.append(args[0][n])
            self.data[n].train_accy.append(args[1][n])
            self.data[n].valid_loss.append(args[2][n])
            self.data[n].valid_accy.append(args[3][n])
            self.data[n].epoch.append(args[4])
            self.data[n].time.append(args[5])
        return

    def save_pickle(self, path):
        with open(path, 'wb') as result:
            pickle.dump(self.data, result, pickle.HIGHEST_PROTOCOL)
        return



class Experiment(object):
    
    def __init__(self, name:str, deep:Results, ensemble:EnsembleResults):
        self.name = name
        self.data = dict(
            single=None,
            ensemble=None)
        self.models = dict(
            single=None,
            ensemble=None)

    def save(self, path):
        with open(path, 'wb') as experiment:
            pickle.dump(self, experiment, pickle.HIGHEST_PROTOCOL)
        return

