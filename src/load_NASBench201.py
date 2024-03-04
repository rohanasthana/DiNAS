import sys, pathlib
import os, glob, json
import torch
import numpy as np 
import itertools
import tqdm, hashlib
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F

from collections import OrderedDict
from pathlib import Path
import copy



if __name__ == "__main__":
    
    # Set package for relative imports
    if __package__ is None or len(__package__) == 0:                  
        DIR = pathlib.Path(__file__).resolve().parent.parent
        print(DIR)
        sys.path.insert(0, str(DIR.parent))
        __package__ = DIR.name

##############################################################################
#
#                              Dataset Code
#
##############################################################################



if __name__ == "__main__":
    from .Generator import scores_to_adj, adj_to_scores
    from nasbench201 import api
    import Settings
else:
    from Generator import scores_to_adj, adj_to_scores
    from src.datasets.nasbench201 import api
    import Settings


# Useful constants
OP_PRIMITIVES_NB201 = [
    'output',
    'input',
    'nor_conv_1x1',
    'nor_conv_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'none',
]

INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'nor_conv_3x3'
CONV1X1 = 'nor_conv_1x1'
AVGPOOL3X3 = 'avg_pool_3x3'
SKIP = 'skip_connect'
NONE = 'none'

OPS_by_IDX_NB201 = {OP_PRIMITIVES_NB201.index(i):i for i in OP_PRIMITIVES_NB201}
OPS_NB201 = {i:OP_PRIMITIVES_NB201.index(i) for i in OP_PRIMITIVES_NB201}

OPS = [CONV1X1, CONV3X3, AVGPOOL3X3, SKIP, NONE]

all_list = []
for i in range(7):
    operations_list = [OPS for _ in range(i)]
    for comb in itertools.product(*operations_list):
        all_list.append(comb[::-1])
len_all_list = list(map(len, all_list))

ADJACENCY_NB201 = np.array([[0, 1, 1, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 1 ,0 ,0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0]])

MATRIX = torch.triu(torch.ones(8,8), diagonal=1)


nasbench = api.NASBench201API('/home/asthana/Documents/DiGress/DiGress/data/nasbench201_samples/NAS-Bench-201-v1_1-096897.pth')



cifar10_val_acc_max =  0.9161 #0.9160666465759277
cifar10_test_acc_max = 0.9437 #0.9437333345413208
cifar100_val_acc_max = 0.7349 #0.7349333167076111
cifar100_test_acc_max = 0.7351 #0.7351333498954773
Imagenet16_20_val_acc_max = 0.4673 # 0.4673333466053009
Imagenet16_20_test_acc_max = 0.4731 #0.4731111228466034


INPUT = 'input'
OUTPUT = 'output'
#OPS = ['avg_pool_3x3', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'skip_connect']
NUM_OPS = len(OPS)
OP_SPOTS = 6
LONGEST_PATH_LENGTH = 3

class Dataset:
            
    ##########################################################################
    def __init__(
            self,
            batch_size,
            sample_size = 50,
            only_prediction = False,
            dataset = 'cifar10_valid_converged'#'cifar10_valid_converged' ##ImageNet16-120, cifar100
        ):


        
        if __name__ == "__main__":
            path = os.path.join(".","NASBench201") 
        else:
            path = os.path.join(".","datasets","NASBench201") #for debugging
        
        file_cache = os.path.join(path, "cache_"+dataset)
        self.file_cache = file_cache
        file_hash = os.path.join(path, "hash.torch")
        self.file_hash = file_hash

        ############################################        

        if not os.path.isfile(file_cache):
            self.data = []
            for index in tqdm.tqdm(range(len(nasbench))):
                item = nasbench.query_meta_info_by_index(index)
                self.data.append(Dataset.map_item(index, dataset=dataset))
                self.data[-1].edge_index =  Dataset.map_network(item)[0]
                self.data[-1].x = Dataset.map_network(item)[1]
                self.data[-1].x_binary = Dataset.map_network(item)[2]
                self.data[-1].scores = Dataset.map_network(item)[3]
                self.data[-1].y = Dataset.map_network(item)[4]
                self.data[-1].g = Dataset.map_network(item)[4]
                self.data[-1].num_nodes = 8
            
            print(f"Saving data to cache: {file_cache}")
            torch.save(self.data, file_cache)
        
        else:
            print(f"Loading data from cache: {file_cache}")
            self.data = torch.load(file_cache)        
        

        ############################################
        self.train_data, self.test_data  = Dataset.sample(self.data, sample_size, only_prediction)
        
        self.length = len(self.data)

        self.train_dataloader = DataLoader(
            self.train_data,
            shuffle = True,
            num_workers = 4,
            pin_memory = True,
            batch_size = batch_size
        )

        self.test_dataloader = DataLoader(
            self.test_data,
            shuffle = False,
            num_workers = 4,
            pin_memory = False,
            batch_size = batch_size
        )

        self.dataloader = DataLoader(
            self.data,
            shuffle = True,
            num_workers = 4,
            pin_memory = True,
            batch_size = batch_size
        )

        self.dim = 56
        self.idx_ops = {
            0:             range(0, 7),
            1:             range(7, 14),
            2:             range(14, 21),
            3:             range(21, 28),
            4:             range(28, 35),
            5:             range(35, 42),
            6:             range(42, 49),
            7:             range(49, 56),
        }
        self.idx_ops_noop = {k: [v[-1]] for k, v in self.idx_ops.items()}
        self.idx_ops_in   = {k: [v[1]]  for k, v in self.idx_ops.items()}
        self.idx_ops_out  = {k: [v[0]]  for k, v in self.idx_ops.items()}


    ##########################################################################
    @staticmethod
    def map_item(item, dataset='cifar10_valid_converged'):
        if dataset == 'cifar10_valid_converged':
            valid_acc, val_acc_avg, time_cost, test_acc, test_acc_avg = Dataset.train_and_eval(item, nepoch=None, dataname='cifar10-valid', use_converged_LR=True)
        else:
            valid_acc, val_acc_avg, time_cost, test_acc, test_acc_avg = Dataset.train_and_eval(item, nepoch=199, dataname=dataset, use_converged_LR=False)
        acc = torch.FloatTensor([valid_acc/100.0 ])
        test_acc = torch.FloatTensor([test_acc/100.0 ])
        val_acc_avg = torch.FloatTensor([val_acc_avg/100.0])
        test_acc_avg = torch.FloatTensor([test_acc_avg/100.0 ])
        training_time = torch.FloatTensor([time_cost/100.0])
        
        return Data(val_acc=val_acc_avg, acc=test_acc_avg, training_time=training_time)

    ##########################################################################
    @staticmethod
    def map_network(item):

        nodes = ['input']
        steps = item.arch_str.split('+')
        steps_coding = ['0', '0', '1', '0', '1', '2']
        cont = 0
        for step in steps:
            step = step.strip('|').split('|')
            for node in step:
                n, idx = node.split('~')
                assert idx == steps_coding[cont]
                cont += 1
                nodes.append(n)
        nodes.append('output')

        ops = [OPS_NB201[k] for k in nodes]

        node_attr = torch.LongTensor(ops)
        edge_index = torch.tensor(np.nonzero(ADJACENCY_NB201))
        adj = torch.from_numpy(ADJACENCY_NB201)
        scores = adj_to_scores(adj, l=28)
        x_binary = torch.nn.functional.one_hot(node_attr, num_classes=7)
        y = torch.cat((x_binary.reshape(-1).float(), scores.float()))


        return edge_index.long(), node_attr, x_binary, scores, y
            
    ##########################################################################
    @staticmethod
    def train_and_eval(arch_index, nepoch=None, dataname=None, use_converged_LR=True):
        assert dataname !='cifar10', 'Do not allow cifar10 dataset'
        if use_converged_LR and dataname=='cifar10-valid':
            assert nepoch == None, 'When using use_converged_LR=True, please set nepoch=None, use 12-converged-epoch by default.'


            info = nasbench.get_more_info(arch_index, dataname, None, True)
            valid_acc, time_cost = info['valid-accuracy'], info['train-all-time'] + info['valid-per-time']
            valid_acc_avg = nasbench.get_more_info(arch_index, 'cifar10-valid', None, False, False)['valid-accuracy']
            test_acc = nasbench.get_more_info(arch_index, 'cifar10', None, False, True)['test-accuracy']
            test_acc_avg = nasbench.get_more_info(arch_index, 'cifar10', None, False, False)['test-accuracy']

        elif not use_converged_LR:

            assert isinstance(nepoch, int), 'nepoch should be int'
            xoinfo = nasbench.get_more_info(arch_index, 'cifar10-valid', None, True)
            xocost = nasbench.get_cost_info(arch_index, 'cifar10-valid', False)
            info = nasbench.get_more_info(arch_index, dataname, nepoch, False, True)
            cost = nasbench.get_cost_info(arch_index, dataname, False)
            # The following codes are used to estimate the time cost.
            # When we build NAS-Bench-201, architectures are trained on different machines and we can not use that time record.
            # When we create checkpoints for converged_LR, we run all experiments on 1080Ti, and thus the time for each architecture can be fairly compared.
            nums = {'ImageNet16-120-train': 151700, 'ImageNet16-120-valid': 3000,
                    'cifar10-valid-train' : 25000,  'cifar10-valid-valid' : 25000,
                    'cifar100-train'      : 50000,  'cifar100-valid'      : 5000}
            estimated_train_cost = xoinfo['train-per-time'] / nums['cifar10-valid-train'] * nums['{:}-train'.format(dataname)] / xocost['latency'] * cost['latency'] * nepoch
            estimated_valid_cost = xoinfo['valid-per-time'] / nums['cifar10-valid-valid'] * nums['{:}-valid'.format(dataname)] / xocost['latency'] * cost['latency']
            try:
                valid_acc, time_cost = info['valid-accuracy'], estimated_train_cost + estimated_valid_cost
            except:
                valid_acc, time_cost = info['est-valid-accuracy'], estimated_train_cost + estimated_valid_cost
            test_acc = info['test-accuracy']
            test_acc_avg = nasbench.get_more_info(arch_index, dataname, None, False, False)['test-accuracy']
            valid_acc_avg = nasbench.get_more_info(arch_index, dataname, None, False, False)['valid-accuracy']
        else:
            # train a model from scratch.
            raise ValueError('NOT IMPLEMENT YET')
        return valid_acc, valid_acc_avg, time_cost, test_acc, test_acc_avg
        
    ##########################################################################
    @staticmethod
    def sample(dataset,
        sample_size = 50,
        only_prediction = False,
        seed = 999         
    ):
        random_shuffle = np.random.permutation(range(len(dataset)))

        test_data = copy.deepcopy([dataset[i] for i in random_shuffle[:-sample_size]])
        
        if only_prediction:
            train_data = copy.deepcopy([dataset[i] for i in random_shuffle[-sample_size:]])

        else:
            train_data = copy.deepcopy([dataset[i] for i in random_shuffle[-sample_size:]])
        
        return train_data, test_data

    ##########################################################################
    @staticmethod
    def pg_graph_to_nb201(data, dataset):
        # first tensor node attributes, second is the edge list
        ops = [OPS_by_IDX_NB201[i] for i in data.x.cpu().numpy()]
        try: 
            steps_coding = ['0', '0', '1', '0', '1', '2']

            node_1='|'+ops[1]+'~'+steps_coding[0]+'|'
            node_2='|'+ops[2]+'~'+steps_coding[1]+'|'+ops[3]+'~'+steps_coding[2]+'|'
            node_3='|'+ops[4]+'~'+steps_coding[3]+'|'+ops[5]+'~'+steps_coding[4]+'|'+ops[6]+'~'+steps_coding[5]+'|'
            nodes_nb201=node_1+'+'+node_2+'+'+node_3
            index = nasbench.query_index_by_arch(nodes_nb201)
            data_acc = Dataset.map_item(index, dataset)                
            return data_acc
        except:
            pass
    ##########################################################################

    @staticmethod
    def get_nb201_index(data):
        # first tensor node attributes, second is the edge list
        ops = [OPS_by_IDX_NB201[i] for i in data.x.cpu().numpy()]
        matrix = np.array(to_dense_adj(data.edge_index)[0].cpu().numpy())
        try: 
            steps_coding = ['0', '0', '1', '0', '1', '2']

            node_1='|'+ops[1]+'~'+steps_coding[0]+'|'
            node_2='|'+ops[2]+'~'+steps_coding[1]+'|'+ops[3]+'~'+steps_coding[2]+'|'
            node_3='|'+ops[4]+'~'+steps_coding[3]+'|'+ops[5]+'~'+steps_coding[4]+'|'+ops[6]+'~'+steps_coding[5]+'|'
            nodes_nb201=node_1+'+'+node_2+'+'+node_3
            index = nasbench.query_index_by_arch(nodes_nb201)
            return index
        except:
            pass
              

    ##########################################################################
    @staticmethod
    def get_info_generated_graph(item, dataset):
        if isinstance(item , list):
            data = []
            for graph in item:
                try: 
                    data_acc = Dataset.pg_graph_to_nb201(graph, dataset)
                    graph.val_acc = data_acc.val_acc   
                    graph.acc = data_acc.acc  
                    graph.training_time = data_acc.training_time  
                    data.append(graph)
                except:
                    continue

        else:
            try:
                data_acc = Dataset.pg_graph_to_nb201(item, dataset)
                item.val_acc = data_acc.val_acc   
                item.acc = data_acc.acc  
                item.training_time = data_acc.training_time  
                data = item
            except:
                pass


        return data
    ##########################################################################
    def hash_simple(self, y_binary=None, y=None):
        if y is not None:
            y_np = y.detach().cpu().numpy()
            y_binary = np.where(y_np > 0, 1, 0)
        assert y_binary is not None, "No parameters given."
        
        fingerprints = ["".join([str(i) for i in k.tolist()]) for k in y_binary.astype(int)]
        return fingerprints
    ##########################################################################
    def load_hashes(self):
        
        data = self.data
        self.accs_val =  torch.stack([d.val_acc for d in data])
        self.accs_test =  torch.stack([d.acc for d in data])
        self.y_data = torch.stack([d.y for d in data])
        if os.path.isfile(self.file_hash):
            self.hash_simple2idx = torch.load(self.file_hash)
        else:
            hash_simple = self.hash_simple(y=self.y_data)
            self.hash_simple2idx = {hash_simple[i]:i for i in range(len(hash_simple))}
            if self.file_hash is not None:
                torch.save(
                    obj = self.hash_simple2idx,
                    f = self.file_hash
                )
    ##########################################################################
    def query(self, data):

        hash_simple = self.hash_simple(y=data)
        q = []
        for i, h in enumerate(hash_simple):
            if h not in self.hash_simple2idx:
                raise RuntimeError("Architecture not in data.")
            idx = self.hash_simple2idx[h]
            # q.append([idx, self.accs_val[idx], self.accs_test[idx]])
            q.append(idx)
        return q


##############################################################################
#
#                              Debugging
#
##############################################################################

if __name__ == "__main__":
    
    def print_keys(d, k=None, lvl=0):
        if k is not None:
            print(f"{'---'*(lvl)}{k}")
        if type(d) == list and len(d) == 1:
            d = d[0]
        if type(d) == dict:
            for k in d:
                print_keys(d[k], k, lvl+1)
                
    # print_keys(json.load(open(os.path.join(".","NASBench301","results_0.json"), "r")))
    ds = Dataset(10)
    print(ds.data)
    for batch in ds.dataloader:
        print(batch)
        break