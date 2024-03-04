import sys, pathlib
import os, glob, json, tqdm
import torch
import numpy as np 
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj, subgraph
import itertools
import copy
import src.Settings

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
# JSON STRUCTURE
##############################################################################
# ---final_metric_score
# ---optimized_hyperparamater_config
# ------ImageAugmentation:augment
# ------ImageAugmentation:autoaugment
# ------ImageAugmentation:cutout
# ------ImageAugmentation:cutout_holes
# ------ImageAugmentation:fastautoaugment
# ------NetworkSelectorDatasetInfo:darts:drop_path_prob
# ------NetworkSelectorDatasetInfo:network
# ------CreateImageDataLoader:batch_size
# ------ImageAugmentation:cutout_length
# ------LossModuleSelectorIndices:loss_module
# ------NetworkSelectorDatasetInfo:darts:auxiliary
# ------NetworkSelectorDatasetInfo:darts:edge_normal_0
# ------NetworkSelectorDatasetInfo:darts:edge_normal_1
# ------NetworkSelectorDatasetInfo:darts:edge_reduce_0
# ------NetworkSelectorDatasetInfo:darts:edge_reduce_1
# ------NetworkSelectorDatasetInfo:darts:init_channels
# ------NetworkSelectorDatasetInfo:darts:inputs_node_normal_3
# ------NetworkSelectorDatasetInfo:darts:inputs_node_normal_4
# ------NetworkSelectorDatasetInfo:darts:inputs_node_normal_5
# ------NetworkSelectorDatasetInfo:darts:inputs_node_reduce_3
# ------NetworkSelectorDatasetInfo:darts:inputs_node_reduce_4
# ------NetworkSelectorDatasetInfo:darts:inputs_node_reduce_5
# ------NetworkSelectorDatasetInfo:darts:layers
# ------OptimizerSelector:optimizer
# ------OptimizerSelector:sgd:learning_rate
# ------OptimizerSelector:sgd:momentum
# ------OptimizerSelector:sgd:weight_decay
# ------SimpleLearningrateSchedulerSelector:cosine_annealing:T_max
# ------SimpleLearningrateSchedulerSelector:cosine_annealing:eta_min
# ------SimpleLearningrateSchedulerSelector:lr_scheduler
# ------SimpleTrainNode:batch_loss_computation_technique
# ------SimpleTrainNode:mixup:alpha
# ------NetworkSelectorDatasetInfo:darts:edge_normal_11
# ------NetworkSelectorDatasetInfo:darts:edge_normal_13
# ------NetworkSelectorDatasetInfo:darts:edge_normal_2
# ------NetworkSelectorDatasetInfo:darts:edge_normal_3
# ------NetworkSelectorDatasetInfo:darts:edge_normal_5
# ------NetworkSelectorDatasetInfo:darts:edge_normal_6
# ------NetworkSelectorDatasetInfo:darts:edge_reduce_12
# ------NetworkSelectorDatasetInfo:darts:edge_reduce_13
# ------NetworkSelectorDatasetInfo:darts:edge_reduce_3
# ------NetworkSelectorDatasetInfo:darts:edge_reduce_4
# ------NetworkSelectorDatasetInfo:darts:edge_reduce_6
# ------NetworkSelectorDatasetInfo:darts:edge_reduce_7
# ---budget
# ---info
# ------train_loss
# ------train_accuracy
# ------train_cross_entropy
# ------val_accuracy
# ------val_cross_entropy
# ------epochs
# ------model_parameters
# ------learning_rate
# ------train_datapoints
# ------val_datapoints
# ------dataset_path
# ------dataset_id
# ------train_loss_final
# ------train_accuracy_final
# ------train_cross_entropy_final
# ------val_accuracy_final
# ------val_cross_entropy_final
# ---test_accuracy
# ---runtime
# ---optimizer
# ---learning_curves
# ------Train/train_loss
# ------Train/train_accuracy
# ------Train/train_cross_entropy
# ------Train/val_accuracy
# ------Train/val_cross_entropy

##############################################################################

if __name__ == "__main__":
    from src.Generator import scores_to_adj, adj_to_scores
    from src.procedures_darts import TENAS

else:
    from src.Generator import scores_to_adj, adj_to_scores
    from src.procedures_darts import TENAS


OP_PRIMITIVES = [
    'identity',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NODE_OP_PRIMITIVES = [
    'output', # very last output of reduction cell 0
    'input', #very frist input of normal cell 1
    'input_1', #second input of normal cell c_k-1 2
    'identity', #3
    'max_pool_3x3', #4
    'avg_pool_3x3', #5
    'skip_connect', #6
    'sep_conv_3x3', #7
    'sep_conv_5x5', #8
    'dil_conv_3x3', #9 
    'dil_conv_5x5' #10
]

OP_ONEHOT = {i:np.eye(8)[i] for i in range(8)}
OP_ONEHOT_BY_PRIMITIVE ={i:OP_ONEHOT[OP_PRIMITIVES.index(i)] for i in OP_PRIMITIVES}

def sort_edge_index(edge_index):
    num_nodes = np.max(edge_index)+1
    idx = edge_index[1] * num_nodes + edge_index[0]
    perm = idx.argsort()
    return edge_index[:, perm]

adj = 1-np.tri(7)
adj[:,6] =adj[:,1]=0
EDGE_LIST_ALL = np.array(np.nonzero(adj))
EDGE_LIST_ALL=  sort_edge_index(EDGE_LIST_ALL)

L = [(x,y) for x,y in EDGE_LIST_ALL.T] 
L_inverse = {(x,y): i for i,(x,y) in enumerate(L)}


# # Load NB301 Surrogate Model
# import os
from collections import namedtuple

from ConfigSpace.read_and_write import json as cs_json

import nasbench301 as nb

version = '0.9'

current_dir = '/home/asthana/Documents/GenerativeNAS/AG-Net-main/datasets/nasbench301'

models_0_9_dir = os.path.join(current_dir, 'nb_models_0.9')
model_paths_0_9 = {
    model_name : os.path.join(models_0_9_dir, '{}_v0.9'.format(model_name))
    for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
}
models_1_0_dir = os.path.join(current_dir, 'nb_models_1.0')
model_paths_1_0 = {
    model_name : os.path.join(models_1_0_dir, '{}_v1.0'.format(model_name))
    for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
}
model_paths = model_paths_0_9 if version == '0.9' else model_paths_1_0

# Load the performance surrogate model
#NOTE: Loading the ensemble will set the seed to the same as used during training (logged in the model_configs.json)
#NOTE: Defaults to using the default model download path
print("==> Loading performance surrogate model...")
ensemble_dir_performance = model_paths['xgb']
print(ensemble_dir_performance)
performance_model = nb.load_ensemble(ensemble_dir_performance)

# Load the runtime surrogate model
#NOTE: Defaults to using the default model download path
print("==> Loading runtime surrogate model...")
ensemble_dir_runtime = model_paths['lgb_runtime']
runtime_model = nb.load_ensemble(ensemble_dir_runtime)

class Dataset:
            
    ##########################################################################
    def __init__(
            self,
            batch_size,
            sample_size = 50,
            only_prediction = False,
            generation = False
        ):

        if __name__ == "__main__":
            path = os.path.join(".","NASBench301") 
        else:
            path = os.path.join(".","datasets","NASBench301") #for debugging
        
        path_2 = '/work/jlukasik/GANS_NAS/AG_Net/datasets/NASBench301/'

        if generation:
            file_cache = os.path.join(path, "cache_1k")
        else:
            file_cache = os.path.join(path, "cache_1k_acc")


        ############################################    
        
        if not os.path.isfile(file_cache):
            files = [os.path.join(root, name) for root, dirs, files in os.walk(path_2) for name in files if name.endswith((".json"))]
            self.data = []
            for file in tqdm.tqdm(files):#change to go through each folder!
                d = json.load(open(file, "r"))
                self.data.append(Dataset.map_network(d))
                if not generation:
                    self.data[-1].acc    = Dataset.map_item(d)[0] 
                    self.data[-1].training_time = Dataset.map_item(d)[1]
                if len(self.data) > 999:
                    break

            if generation:
                files = 'datasets/NASBench301/normal_cell_topologies.json'
                d = json.load(open(files,"r"))
                data = list(d.values())
                flat_list = [item for sublist in data for item in sublist]
                for file in tqdm.tqdm(flat_list):##change to go through each folder!
                    self.data.append(Dataset.map_network(file))
                    if len(self.data) > 999:
                        break
            
            print(f"Check data for duplicates: {file_cache}")
            ys = [np.concatenate((graph.y_normal.detach().numpy(), graph.y_reduce.detach().numpy())) for graph in self.data]            
            ys_np = np.array(ys)
            u, ind = np.unique(ys_np, axis=0, return_index=True)
            self.data = [self.data[i] for i in ind]
            print(f"Length data : {len(self.data)}")
            print(f"Saving data from cache: {file_cache}")
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

    ##########################################################################
    @staticmethod
    def map_item(item):
        acc = item["test_accuracy"]/100.0
        training_time = item["runtime"]
        
        return torch.FloatTensor([acc]),  torch.FloatTensor([training_time])

    ##########################################################################
    @staticmethod
    def map_network_cell(item, cell="normal"):
        if "optimized_hyperparamater_config" in item:
            item = item["optimized_hyperparamater_config"]
        
        edge_u = []
        edge_v = []
        edge_attr = []
        for i in range(14):
            idx = f"NetworkSelectorDatasetInfo:darts:edge_{cell}_{i}"
            if idx in item:
                u, v = L[i]
                edge_u += [u]
                edge_v += [v]
                edge_attr += [NODE_OP_PRIMITIVES.index(item[idx])]

        edge_u    += [2, 3, 4, 5]
        edge_v    += [6]*4
        edge_attr += [NODE_OP_PRIMITIVES.index("identity")]*4
        
        edge_index = torch.tensor([edge_u, edge_v])
        edge_attr  = torch.tensor(edge_attr)
        
        return edge_index, edge_attr

    ##########################################################################
    @staticmethod
    def map_network_cell_like(normal_cell):

        def make_adj(edge_index):
            adj = torch.zeros(11, 11)

            adj[0][2]=1
            adj[1][3]=1

            adj[2][-1] = 1
            adj[3][-1] = 1
            adj[4][-1] = 1
            adj[5][-1] = 1
            adj[6][-1] = 1
            adj[7][-1] = 1
            adj[8][-1] = 1
            adj[9][-1] = 1

            ##Edge 3 : Node  4,5
            subgraph=[]
            for i in range(len(edge_index[0])):
                if edge_index[1][i]==3:
                    subgraph.append((edge_index[0][i].item(), edge_index[1][i].item()))    

            if subgraph[0][0]>=2:
                if subgraph[0][0]%2!=0:
                    adj[subgraph[0][0]+1][4]=1
                    adj[subgraph[0][0]+2][4]=1
                else:
                    adj[subgraph[0][0]][4]=1
                    adj[subgraph[0][0]+1][4]=1
            else:
                adj[subgraph[0][0]][4]=1
            if subgraph[1][0]>=2:
                if subgraph[1][0]%2!=0:
                    adj[subgraph[1][0]+1][5]=1
                    adj[subgraph[1][0]+2][5]=1
                else:
                    adj[subgraph[1][0]][5]=1
                    adj[subgraph[1][0]+1][5]=1
            else:
                adj[subgraph[1][0]][5]=1

            ##Edge 4 : Node  6,7
            subgraph=[]
            for i in range(len(edge_index[0])):
                if edge_index[1][i]==4:
                    subgraph.append((edge_index[0][i].item(), edge_index[1][i].item()))    


            if subgraph[0][0]>=2:
                if subgraph[0][0]%2!=0:
                    adj[subgraph[0][0]+1][6]=1
                    adj[subgraph[0][0]+2][6]=1
                else:
                    adj[subgraph[0][0]][6]=1
                    adj[subgraph[0][0]+1][6]=1
            else:
                adj[subgraph[0][0]][6]=1
            if subgraph[1][0]>=2:
                if subgraph[1][0]%2!=0:
                    adj[subgraph[1][0]+1][7]=1
                    adj[subgraph[1][0]+2][7]=1
                else:
                    adj[subgraph[1][0]][7]=1
                    adj[subgraph[1][0]+1][7]=1
            else:
                adj[subgraph[1][0]][7]=1

            ##Edge 5 : Node  8,9
            subgraph=[]
            for i in range(len(edge_index[0])):
                if edge_index[1][i]==5:
                    subgraph.append((edge_index[0][i].item(), edge_index[1][i].item()))    

            if subgraph[0][0]>=2:
                if subgraph[0][0]%2!=0:
                    adj[subgraph[0][0]+1][8]=1
                    adj[subgraph[0][0]+2][8]=1
                else:
                    adj[subgraph[0][0]][8]=1
                    adj[subgraph[0][0]+1][8]=1
            else:
                adj[subgraph[0][0]][8]=1
            if subgraph[1][0]>=2:
                if subgraph[1][0]%2!=0:
                    adj[subgraph[1][0]+1][9]=1
                    adj[subgraph[1][0]+2][9]=1
                else:
                    adj[subgraph[1][0]+2][9]=1
                    adj[subgraph[1][0]+3][9]=1
            else:
                adj[subgraph[1][0]][9]=1

            return adj

        normal_adj = make_adj(normal_cell[0])            
        edge_index = torch.nonzero(normal_adj, as_tuple=False).T

        node_attr = [NODE_OP_PRIMITIVES.index('input'), NODE_OP_PRIMITIVES.index('input_1')]
        node_attr.extend(normal_cell[1][:8])
        node_attr.extend([NODE_OP_PRIMITIVES.index('output')])
        
        return edge_index, torch.tensor(node_attr)

    @staticmethod
    def map_network(item):
        edge_index_normal, edge_attr_normal = Dataset.map_network_cell(item, cell="normal")
        edge_index_reduce, edge_attr_reduce = Dataset.map_network_cell(item, cell="reduce")
        edge_index_normal, node_attr_normal = Dataset.map_network_cell_like((edge_index_normal, edge_attr_normal))
        edge_index_reduce, node_attr_reduce = Dataset.map_network_cell_like((edge_index_reduce,edge_attr_reduce ))

        x_normal = node_attr_normal
        x_reduce = node_attr_reduce
        num_nodes = x_normal.shape[0]
        adj_normal = to_dense_adj(edge_index_normal)[0]
        adj_reduce = to_dense_adj(edge_index_reduce)[0]
        scores_normal = adj_to_scores(adj_normal, l=55)
        scores_reduce = adj_to_scores(adj_reduce, l=55)

        x_binary_normal = torch.nn.functional.one_hot(x_normal, num_classes=11)
        x_binary_reduce = torch.nn.functional.one_hot(x_reduce, num_classes=11)
        y_normal = torch.cat((x_binary_normal.reshape(-1).float(), scores_normal.float()))
        y_reduce = torch.cat((x_binary_reduce.reshape(-1).float(), scores_reduce.float()))
        
        # return edge_index, edge_attr
        return Data(edge_index_normal=edge_index_normal.long(),
                    edge_index_reduce=edge_index_reduce.long(),
                    x_normal=x_normal, x_reduce=x_reduce,
                    num_nodes=num_nodes,
                    x_binary_normal=x_binary_normal, 
                    x_binary_reduce=x_binary_reduce, 
                    y_normal=y_normal,  y_reduce=y_reduce, 
                    scores_normal=scores_normal,  scores_reduce=scores_reduce
                    )
    ##########################################################################
    @staticmethod
    def transform_node_atts_to_darts_cell(matrix):
        r'input already normal matrix or reduced!!'

        darts_adjacency_matrix = np.zeros((7, 7)) 

        darts_adjacency_matrix[:,0] = matrix[:7,0]
        darts_adjacency_matrix[:,1] = matrix[:7,1]
        d_c = 2
        for i in range(2,10,2):
            a = np.zeros((7))
            a[:2] = np.sum(matrix[:2,i:i+2],1)
            column_conc = np.sum(matrix[2:-1, i:i+2],1)
            row = [np.sum(column_conc[i:i+2]) for i in range(0,8,2)]
            a[2:-1] = row
            a[-1] = np.sum(matrix[-1,i:i+2])
            darts_adjacency_matrix[:,d_c] = a
            d_c += 1
        a = np.zeros((7))
        a[:2] = matrix[:2,-1]
        row = [np.sum(matrix[i:i+2,-1]) for i in range(2,10,2)]
        a[2:-1] = row
        a[-1] = matrix[-1,-1]
        darts_adjacency_matrix[:,-1] = a


        # darts_adjacency_matrix[darts_adjacency_matrix >= 2] = 1  
        # # prune edges if too many edges:
        # for column in darts_adjacency_matrix[:,2:6].T:
        #     if sum(column) > 2:
        #     column[np.where(column==1)[0][2]:]=0

        return darts_adjacency_matrix
    ##########################################################################
    @staticmethod
    def tansform_generated_graph_to_darts_matrix(graph):
        matrix = to_dense_adj(graph.edge_index)[0].cpu().numpy().astype(int)
        
        log_sig = torch.nn.LogSigmoid() 
        edges = log_sig(graph.g[-55:])
        score_matrix = scores_to_adj(edges, num_nodes=11).cpu().numpy()

        normal_darts = Dataset.transform_node_atts_to_darts_cell(matrix)

        score_matrix_darts = Dataset.transform_node_atts_to_darts_cell(score_matrix)

        normal_darts[normal_darts >= 2] = 1  

        # prune edges if too many edges:
        for i,column in enumerate(normal_darts[:,2:6].T):
            column[2+i:] = 0
            if sum(column) > 2:
                # column[np.where(column==1)[0][2]:]=0

                # other idea:
                score_column = score_matrix_darts[:,2+i]
                ind = np.argsort(score_column[np.where(column==1)[0]])[:-2]
                column[np.where(column==1)[0][ind]]= 0
        
#         graph.adj = torch.tensor(normal_darts)
        return  normal_darts
    ##########################################################################
    @staticmethod
    def tansform_spec_generated_graph_to_darts_matrix(graph, normal=False):
        log_sig = torch.nn.LogSigmoid()             

        if normal:
            matrix = to_dense_adj(graph.edge_index_normal)[0].cpu().numpy().astype(int)
            edges = log_sig(graph.scores_normal)
        else:
            matrix = to_dense_adj(graph.edge_index_reduce)[0].cpu().numpy().astype(int)
            edges = log_sig(graph.scores_reduce)
        
        
        
        score_matrix = scores_to_adj(edges, num_nodes=11).cpu().numpy()

        normal_darts = Dataset.transform_node_atts_to_darts_cell(matrix)
        score_matrix_darts = Dataset.transform_node_atts_to_darts_cell(score_matrix)

        normal_darts[normal_darts >= 2] = 1  

        # prune edges if too many edges:
        for i,column in enumerate(normal_darts[:,2:6].T):
            column[2+i:] = 0
            if sum(column) > 2:
                score_column = score_matrix_darts[:,2+i]
                ind = np.argsort(score_column[np.where(column==1)[0]])[:-2]
                column[np.where(column==1)[0][ind]]= 0
        
#         graph.adj = torch.tensor(normal_darts)
        return  normal_darts
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
            # sampled_acc = torch.stack([g.val_acc for g in train_data])

        else:
            train_data = copy.deepcopy(dataset)
#             for i in random_shuffle[:-sample_size]:
#                 train_data[i].val_acc *= -1

            # sampled_acc = torch.stack([train_data[i].val_acc for i in random_shuffle[-sample_size:]])
        
        return train_data, test_data #, sampled_acc

    ##########################################################################
    @staticmethod
    def generate_genotype(matrix, ops):
        r'already normal or reduced matrix and ops'
        #print(matrix)
        cell = []
        i = 0
        for ind in range(2,6):
            edge_0 = np.where(matrix[:,ind]==1)[0][0]
            edge_1 = np.where(matrix[:,ind]==1)[0][1]
            cell.append((ops[i],edge_0))
            cell.append((ops[i+1],edge_1))
            i += 1
        return cell

    ##########################################################################
    @staticmethod
    def get_info_generated_graph(item, dataset=None):

        Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

        if isinstance(item , list):
            data = []
            for graph in item:
                ops_n =[NODE_OP_PRIMITIVES[attr] for attr in graph.x_normal.cpu().numpy()]
                ops_r =[NODE_OP_PRIMITIVES[attr] for attr in graph.x_reduce.cpu().numpy()]
                try:
                    normal_darts_adj = Dataset.tansform_spec_generated_graph_to_darts_matrix(graph, normal=True)
                    reduce_darts_adj = Dataset.tansform_spec_generated_graph_to_darts_matrix(graph, normal=False)


                    ops_normal = ops_n[2:10]
                    ops_reduce =  ops_r[2:10]

                    normal_cell = Dataset.generate_genotype(normal_darts_adj, ops_normal)
                    reduce_cell = Dataset.generate_genotype(reduce_darts_adj, ops_reduce)

                    genotype_config = Genotype(normal=normal_cell, normal_concat=[2, 3, 4, 5], reduce=reduce_cell, reduce_concat=[2, 3, 4, 5])
                    # Predict
                    prediction_genotype = performance_model.predict(config=genotype_config, representation="genotype", with_noise=False)
                    runtime_genotype = runtime_model.predict(config=genotype_config, representation="genotype")

                    graph.acc = torch.FloatTensor([prediction_genotype/100.0])
                    graph.training_time = torch.FloatTensor([runtime_genotype])
                    graph.genotype=genotype_config
                    data.append(graph)  
                except:
                    continue

        else:
            try:
                normal_darts_adj = Dataset.tansform_spec_generated_graph_to_darts_matrix(item, normal=True)
                reduce_darts_adj = Dataset.tansform_spec_generated_graph_to_darts_matrix(item, normal=False)

                ops_n = [NODE_OP_PRIMITIVES[attr] for attr in item.x_normal.cpu().numpy()]
                ops_r = [NODE_OP_PRIMITIVES[attr] for attr in item.x_reduce.cpu().numpy()]
                ops_normal = ops_n[2:10]
                ops_reduce =  ops_r[2:10]

                normal_cell = (Dataset.generate_genotype(normal_darts_adj, ops_normal))
                reduce_cell = (Dataset.generate_genotype(reduce_darts_adj, ops_reduce))

                genotype_config = Genotype(normal=normal_cell, normal_concat=[2, 3, 4, 5], reduce=reduce_cell, reduce_concat=[2, 3, 4, 5])

                # Predict
                prediction_genotype = performance_model.predict(config=genotype_config, representation="genotype", with_noise=False)
                runtime_genotype = runtime_model.predict(config=genotype_config, representation="genotype")

                item.acc = torch.FloatTensor([prediction_genotype/100.0])
                item.training_time = torch.FloatTensor([runtime_genotype])
                #item.genoytpe=genotype_config
                data = item

            except:
                return
        return data

    ##########################################################################
    @staticmethod
    def get_genotype(item):

        Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

        if isinstance(item , list):
            genotypes = []
            for graph in item:

                adjacency_matrix_n = to_dense_adj(graph.edge_index_normal)[0].cpu().numpy().astype(int)
                adjacency_matrix_r = to_dense_adj(graph.edge_index_reduce)[0].cpu().numpy().astype(int)
                ops_n =[NODE_OP_PRIMITIVES[attr] for attr in graph.x_normal.cpu().numpy()]
                ops_r =[NODE_OP_PRIMITIVES[attr] for attr in graph.x_reduce.cpu().numpy()]
                try:
                    normal_darts_adj = Dataset.transform_node_atts_to_darts_cell(adjacency_matrix_n[:11,:11]) 
                    normal_darts_adj[normal_darts_adj >= 2] = 1

                    reduce_darts_adj = Dataset.transform_node_atts_to_darts_cell(adjacency_matrix_r[:11,:11]) 
                    reduce_darts_adj[reduce_darts_adj >= 2] = 1

                    ops_normal = ops_n[2:10]
                    ops_reduce =  ops_r[2:10]

                    normal_cell = Dataset.generate_genotype(normal_darts_adj, ops_normal)
                    reduce_cell = Dataset.generate_genotype(reduce_darts_adj, ops_reduce)

                    genotype = Genotype(normal=normal_cell, normal_concat=[2, 3, 4, 5], reduce=reduce_cell, reduce_concat=[2, 3, 4, 5])
                    # Predict
                    # prediction_genotype = performance_model.predict(config=genotype_config, representation="genotype", with_noise=False)
                    # runtime_genotype = runtime_model.predict(config=genotype_config, representation="genotype")

                    # graph.acc = torch.FloatTensor([prediction_genotype/100.0])
                    # graph.training_time = torch.FloatTensor([runtime_genotype])
                    
                    genotypes.append((normal_cell,reduce_cell))  
                except:
                    continue

        else:
            adjacency_matrix_n = to_dense_adj(item.edge_index_normal)[0].cpu().numpy().astype(int)  
            adjacency_matrix_r = to_dense_adj(item.edge_index_reduce)[0].cpu().numpy().astype(int)  
            try:

                normal_darts_adj = Dataset.transform_node_atts_to_darts_cell(adjacency_matrix_n[:11,:11]) ##dataset
                normal_darts_adj[normal_darts_adj >= 2] = 1

                reduce_darts_adj = Dataset.transform_node_atts_to_darts_cell(adjacency_matrix_r[:11,:11]) ##dataset
                reduce_darts_adj[reduce_darts_adj >= 2] = 1

                ops_n = [NODE_OP_PRIMITIVES[attr] for attr in item.x_normal.cpu().numpy()]
                ops_r = [NODE_OP_PRIMITIVES[attr] for attr in item.x_reduce.cpu().numpy()]
                ops_normal = ops_n[2:10]
                ops_reduce =  ops_r[2:10]

                normal_cell = Dataset.generate_genotype(normal_darts_adj, ops_normal)
                reduce_cell = Dataset.generate_genotype(reduce_darts_adj, ops_reduce)

                #genotype = Genotype(normal=normal_cell, normal_concat=[2, 3, 4, 5], reduce=reduce_cell, reduce_concat=[2, 3, 4, 5])
                
                genotypes = (normal_cell,reduce_cell)
            except:
                pass
        return genotypes

    ##########################################################################
    @staticmethod
    def get_nb301_ntk_lr(tenas_darts, genotype=None, data=None):


        ntk, lr = tenas_darts.calculate_ntk_lr(genotype)

        return ntk[0], lr[0]

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
    for batch in ds.dataloader:
        print(batch)
        break
