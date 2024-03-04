import sys, pathlib
import os, glob, json
import torch
import numpy as np 
import itertools
import tqdm, hashlib
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

from nasbench import api
from nasbench.lib import graph_util
import copy
import Settings

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

##############################################################################
# NAS-Bench-101 Data STRUCTURE .tfrecorf
##############################################################################
# ---nasbench.hash_iterator() : individual hash for each graph in the whole .tfrecord dataset
# ------nasbench.get_metrics_from_hash(unique_hash): metrics of data sample given by the hash
# ---------fixed_metrics: {'module_adjacency': array([[0, 1, 0, 0, ...type=int8),
#                         'module_operations': ['input', 'conv3x3-bn-relu', 'maxpool3x3', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'output'], 
#                         'trainable_parameters': 8555530}
# ---------computed_metrics : dict including the epoch and all metrics which are computed for each architecture{108: [{...}, {...}, {...}]}
# ------------computed_metrics[108]: [{'final_test_accuracy': 0.9211738705635071, 
#                                  'final_train_accuracy': 1.0, 
#                                  'final_training_time': 1769.1279296875,
#                                  'final_validation_accuracy': 0.9241786599159241,
#                                  'halfway_test_accuracy': 0.7740384340286255, 
#                                  'halfway_train_accuracy': 0.8282251358032227,
#                                  'halfway_training_time': 883.4580078125, 
#                                  'halfway_validation_accuracy': 0.7776442170143127},
#                                  {...},{...}]
##############################################################################
#if __name__ == "__main__":
 #   from .Generator import scores_to_adj, adj_to_scores
#else:
 #   from Generator import scores_to_adj, adj_to_scores


OP_PRIMITIVES_NB101 = [
    'output',
    'input',
    'conv1x1-bn-relu',
    'conv3x3-bn-relu',
    'maxpool3x3'
]

OP_ONEHOT_NB101 = {i:np.eye(5)[i] for i in range(5)}
OP_ONEHOT_BY_PRIMITIVE_NB101 ={i:OP_ONEHOT_NB101[OP_PRIMITIVES_NB101.index(i)] for i in OP_PRIMITIVES_NB101}
OPS_NB101 = {i:OP_PRIMITIVES_NB101.index(i) for i in OP_PRIMITIVES_NB101}



# Useful constants
INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix

OPS = [CONV1X1, CONV3X3, MAXPOOL3X3]
OPS_INCLUSIVE = [INPUT, OUTPUT, *OPS]
OP_SPOTS = NUM_VERTICES - 2


epoch=108
OPS_by_IDX_NB101 = {OP_PRIMITIVES_NB101.index(i):i for i in OP_PRIMITIVES_NB101}

all_list = []
for i in range(6):
    operations_list = [OPS for _ in range(i)]
    for comb in itertools.product(*operations_list):
        all_list.append(comb[::-1])
len_all_list = list(map(len, all_list))
MATRIX = torch.triu(torch.ones(7,7), diagonal=1)


nasbench = api.NASBench(Settings.PATH_NB101)

val_acc_max = 0.9506 #0.9505542516708374
test_acc_max = 0.9432 # 0.943175733089447
val_acc_mean = 0.90243375
val_acc_std = 0.05864741
test_acc_mean = 0.8967984
test_acc_std = 0.05799569

class Dataset:
            
    ##########################################################################
    def __init__(
            self,
            batch_size,
            sample_size = 50,
            only_prediction = False,
        ):
        if __name__ == "__main__":
            path = os.path.join(".","NASBench101") 
        else:
            path = os.path.join(".","datasets","NASBench101") #for debugging
        
        
        file_cache = os.path.join(path, "cache")
        self.file_cache = file_cache
        file_hash = os.path.join(path, "hash.torch")
        self.file_hash = file_hash

        ############################################        

        if not os.path.isfile(file_cache):
            self.data = []
            for unique_hash in tqdm.tqdm(nasbench.hash_iterator()):
                fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)
                self.data.append(Dataset.map_network(fixed_metrics)[0])
                self.data[-1].scores = Dataset.map_network(fixed_metrics)[1]
                self.data[-1].val_acc = Dataset.map_item(computed_metrics)[0]
                self.data[-1].acc = Dataset.map_item(computed_metrics)[1]  

            print(f"Saving data to cache: {file_cache}")
            torch.save(self.data, file_cache)
        
        else:
            print(f"Loading data from cache: {file_cache}")
            self.data = torch.load(file_cache)


        self.train_data, self.test_data = Dataset.sample(self.data, sample_size, only_prediction)
        self.length = len(self.data)

        
        self.train_dataloader = DataLoader(
            self.train_data,
            shuffle = True,
            num_workers = 0,
            pin_memory = True,
            batch_size = batch_size
        )

        self.test_dataloader = DataLoader(
            self.test_data,
            shuffle = False,
            num_workers = 0,
            pin_memory = False,
            batch_size = batch_size
        )

        self.dataloader = DataLoader(
            self.data,
            shuffle = True,
            num_workers = 0,
            pin_memory = True,
            batch_size = batch_size
        )
        # For hash Generation 
        self.dim = 56
        self.idx_row = {
            0: (35, 36, 38,  41, 45, 50),
            1: (    37, 39,  42, 46, 51),
            2: (        40,  43, 47, 52),
            3: (             44, 48, 53),
            4: (                 49, 54),
            5: (                    [55])
        }
        self.idx_col = {
            1: (                    [35]),
            2:             range(35, 38),
            3:             range(38, 41),
            4:             range(41, 45),
            5:             range(45, 50),
            6:             range(50, 56)
        }
        self.idx_ops = {
            0:             range(0, 5),
            1:             range(5, 10),
            2:             range(10, 15),
            3:             range(15, 20),
            4:             range(20, 25),
            5:             range(25, 30),
            6:             range(30, 35)
        }
        self.idx_ops_noop = {k: [v[-1]] for k, v in self.idx_ops.items()}
        self.idx_ops_in   = {k: [v[0]]  for k, v in self.idx_ops.items()}
        self.idx_ops_out  = {k: [v[1]]  for k, v in self.idx_ops.items()}

    ##########################################################################
    @staticmethod
    def map_item(item):
        test_acc = 0.0
        val_acc = 0.0
        for repeat_index in range(len(item[epoch])):
            assert len(item[epoch])==3, 'len(computed_metrics[epoch]) should be 3'
            data_point = item[epoch][repeat_index]
            val_acc += data_point['final_validation_accuracy']
            test_acc += data_point['final_test_accuracy']
            # training_time += data_point['final_training_time']
        val_acc = val_acc/3.0
        test_acc = test_acc/3.0
        # training_time_avg = training_time/3.0
        #Runtime needed for BO? or only during BO self?
        
        return torch.FloatTensor([val_acc]), torch.FloatTensor([test_acc])
        
    ##########################################################################
    @staticmethod
    def map_network(item):
        matrix= item['module_adjacency']

        # item = item["optimized_hyperparamater_config"]
        node_operations = item['module_operations']

        node_attr = [OPS_NB101[attr] for attr in node_operations]

        num_nodes = len(node_attr)
       
        adj_flat = torch.tensor(matrix).flatten()
        if len(adj_flat) != 49:
            adj_flat = F.pad(adj_flat, pad=(0,49-len(adj_flat)), value=0)

        scores = np.zeros(21, int)
        i = 0
        for node in range(1,matrix.shape[0]):
            parents = np.nonzero(matrix[:,node])[0]
            scores[i + parents] = 1
            i +=node
        
        edge_index = torch.tensor(np.nonzero(matrix))
        node_attr  = torch.tensor(node_attr)
        y_nodes = node_attr
        if len(y_nodes) != 7:
            y_nodes = F.pad(y_nodes, pad=(0,7-len(y_nodes)), value=OPS_NB101['output'])
        x_binary = torch.nn.functional.one_hot(y_nodes, num_classes=5)

        scores = torch.from_numpy(scores) 
        y = torch.cat((x_binary.reshape(-1).float(), scores.float()))
        
        return Data(edge_index=edge_index.long(), x=node_attr, x_binary = x_binary, num_nodes=num_nodes, y=y, g=y), scores.float()

    
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
    def get_info_generated_graph(item, dataset=None):
        
        if isinstance(item , list):
            data = []
            for graph in item:
                adjacency_matrix = to_dense_adj(graph.edge_index)[0].cpu().numpy().astype(int)
                ops = [OPS_by_IDX_NB101[attr.item()] for attr in graph.x.cpu()]
                try:
                    spec = api.ModelSpec(matrix = adjacency_matrix, ops= ops)
                    fixed_metrics, computed_metrics = nasbench.get_metrics_from_spec(spec)
                    graph.edge_index = Dataset.map_network(fixed_metrics)[0].edge_index
                    graph.x = Dataset.map_network(fixed_metrics)[0].x
                    graph.x_binary = Dataset.map_network(fixed_metrics)[0].x_binary
                    graph.y = Dataset.map_network(fixed_metrics)[0].y
                    graph.y_generated = graph.y
                    graph.scores = Dataset.map_network(fixed_metrics)[1]
                    graph.val_acc = Dataset.map_item(computed_metrics)[0]
                    graph.acc = Dataset.map_item(computed_metrics)[1]
                    data.append(graph)  
                except:
                    continue

        else:
            adjacency_matrix = to_dense_adj(item.edge_index)[0].cpu().numpy().astype(int)
            ops = [OPS_by_IDX_NB101[attr.item()] for attr in item.x.cpu()]
            try:
                spec = api.ModelSpec(matrix = adjacency_matrix, ops = ops)
                fixed_metrics, computed_metrics = nasbench.get_metrics_from_spec(spec)
                item.edge_index = Dataset.map_network(fixed_metrics)[0].edge_index
                item.x = Dataset.map_network(fixed_metrics)[0].x
                item.x_binary = Dataset.map_network(fixed_metrics)[0].x_binary
                item.y = Dataset.map_network(fixed_metrics)[0].y
                item.y_generated = item.y
                item.scores = Dataset.map_network(fixed_metrics)[1]
                item.val_acc = Dataset.map_item(computed_metrics)[0]
                item.acc = Dataset.map_item(computed_metrics)[1] 
                data = item  
            except:
                pass
        return data    

    ##########################################################################
    def unique(self, y):
        y_np = y.detach().cpu().numpy()
        y_binary = np.where(y_np > 0, 1, 0)
        uq = np.unique(y_binary, axis=0, return_index=True)
        uq = len(uq[0]), *uq
        return uq
    ##########################################################################
    def y_to_matrix(self, y):
        if type(y) == torch.Tensor:
            y = y.cpu().detach().numpy()
        matrix = np.zeros((7,7))
        labeling = []
        for i in range(6):
            matrix[i, (i+1):] = y[list(self.idx_row[i])]
            labeling += [y[self.idx_ops[i]]]
        labeling += [y[self.idx_ops[6]]]
        return matrix, labeling
    ##########################################################################
    def print_y(self, y):
        if type(y) == torch.Tensor:
            y = y.cpu().detach().numpy()
        matrix, labeling = self.y_to_matrix(y)
        print("".join(["#"]*10))
        print("Adjacency")
        print("".join(["#"]*10))
        for row in matrix:
            print("{:1.0f} {:1.0f} {:1.0f} {:1.0f} {:1.0f} {:1.0f} {:1.0f}".format(*[i>0 for i in row]))
        print("".join(["#"]*10))
        for l in labeling:
            if type(l) == torch.Tensor:
                l = l.detach().cpu().numpy()
            print("{:1.0f} {:1.0f} {:1.0f} {:1.0f} {:1.0f} {:1.0f}".format(*[i>0 for i in l]))
        print("".join(["#"]*10))

    ##########################################################################
    # from NASbench101 API
    def hash_module(self, y_binary=None, y=None):
        if y is not None:
            y_np = y.detach().cpu().numpy()
            y_binary = np.where(y_np > 0, 1, 0)
        assert y_binary is not None, "No parameters given."
        
        fingerprints = []
        
        for _y in y_binary:
            matrix = np.zeros((7,7))
            labeling = []
            for i in range(6):
                matrix[i, (i+1):] = _y[list(self.idx_row[i])]
                labeling += [_y[self.idx_ops[i]]]
            labeling += [_y[self.idx_ops[6]]]
            
            vertices = np.shape(matrix)[0]
            in_edges = np.sum(matrix, axis=0).tolist()
            out_edges = np.sum(matrix, axis=1).tolist()
    
            assert len(in_edges) == len(out_edges) == len(labeling)
            hashes = list(zip(out_edges, in_edges, labeling))
            hashes = [hashlib.md5(str(h).encode('utf-8')).hexdigest() for h in hashes]
            for _ in range(vertices):
                new_hashes = []
                for v in range(vertices):
                    in_neighbors = [hashes[w] for w in range(vertices) if matrix[w, v]]
                    out_neighbors = [hashes[w] for w in range(vertices) if matrix[v, w]]
                    new_hashes.append(hashlib.md5(
                    (''.join(sorted(in_neighbors)) + '|' +
                     ''.join(sorted(out_neighbors)) + '|' +
                     hashes[v]).encode('utf-8')).hexdigest())
                hashes = new_hashes
                fingerprint = hashlib.md5(str(sorted(hashes)).encode('utf-8')).hexdigest()
    
            fingerprints += [fingerprint]
        return fingerprints
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
            self.hash_simple2fp, self.hash_fp2idx = torch.load(self.file_hash)
        else:
            hash_simple = self.hash_simple(y=self.y_data)
            hash_fp = self.hash_module(y=self.y_data)
            self.hash_simple2fp = {hash_simple[i]:hash_fp[i] for i in range(len(hash_simple))}
            self.hash_fp2idx = {h:i for i, h in enumerate(hash_fp)}
            if self.file_hash is not None:
                torch.save(
                    obj = (self.hash_simple2fp, self.hash_fp2idx),
                    f = self.file_hash
                )
    ##########################################################################
    def query(self, data):
        hash_simple = self.hash_simple(y=data)
        q = []
        for i, h in enumerate(hash_simple):
            if h not in self.hash_simple2fp:
                h_fp = self.hash_module(y=data[i].unsqueeze(0))[0]
                self.hash_simple2fp[h] = h_fp
            else:
                h_fp = self.hash_simple2fp[h]
            try:
                idx = self.hash_fp2idx[h_fp]
            except KeyError:
                print("WHAT!")
                print(h_fp)
                self.print_y(data[i])
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
                
    ds = Dataset(10, sample_size = 10)
    print(ds.data)
    for batch in ds.dataloader:
        print(batch)
        break