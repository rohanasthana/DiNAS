import sys, pathlib
import os, glob, json
import tqdm, hashlib
import networkx as nx 
import numpy as np
import torch 
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj, subgraph
import torch.nn.functional as F

import copy
from src import Settings


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
    from src.Generator import scores_to_adj, adj_to_scores
    from src.nasbenchNLP.nas_environment import Environment
    from src.nasbenchx11.nas_bench_x11.api import load_ensemble
    from src.nasbenchx11.naslib.predictors.utils.encodings_nlp import encode_adj
    from src.nasbenchx11.naslib.search_spaces.nasbenchnlp.conversions import convert_recipe_to_compact, convert_compact_to_recipe
    from src.nasbenchx11.nasbenchnlp.main_one_model_train import main_one_model_train


else:
    from src.Generator import scores_to_adj, adj_to_scores
    from src.nasbenchNLP.nas_environment import Environment
    from src.nasbenchx11.nas_bench_x11.api import load_ensemble
    from src.nasbenchx11.naslib.predictors.utils.encodings_nlp import encode_adj
    from src.nasbenchx11.naslib.search_spaces.nasbenchnlp.conversions import convert_recipe_to_compact, convert_compact_to_recipe
    from src.nasbenchx11.nasbenchnlp.main_one_model_train import main_one_model_train


# Useful constants
NB_NLP_ops = ['output', 'input', 'activation_sigm', 'activation_tanh', 'activation_leaky_relu', \
               'elementwise_sum', 'elementwise_prod', 'linear', 'blend'] ## in compact from naslib no output !!!!!


# Largest Graph with 27 node attrs (inkl. all input and output nodes)
# Only keep architectures with maximal number node atts 12 

precomputed_logs_path = os.path.join('/home/asthana/Documents/DiGress/DiNAS/src/nasbenchNLP/train_logs_single_run/') 
env = Environment(precomputed_logs_path)
search_set = env.get_precomputed_recepies()


nbnlp_surrogate_model = load_ensemble(Settings.PATH_NBNLP)

class Dataset:
            
    ##########################################################################
    def __init__(
            self,
            batch_size,
            sample_size = 50,
            only_prediction = False,
            dataset = 'ImageNet16-120', #'cifar10_valid_converged' ##ImageNet16-120, cifar100, 
            prediction = False, 
        ):



        if __name__ == "__main__":
            path = os.path.join(".","NASBenchNLP") 
        else:
            path = os.path.join(".","datasets","NASBenchNLP") #for debugging
        
        if prediction:
            file_cache = os.path.join(path, "cache_acc")
        else:
            file_cache = os.path.join(path, "cache")
        
        ############################################        

        if not os.path.isfile(file_cache):
            self.data = []
            for index in tqdm.tqdm(range(len(env._logs))):
                #print(index)
                item = search_set[index]
                env.reset()
                epochs_train = 50
                env.simulated_train(item, epochs_train)
                self.data.append(Dataset.map_network(item))
                if env.get_model_status(item) == 'OK':
                    self.data[-1].val_acc = Dataset.map_item(item)[0]
                    self.data[-1].acc = Dataset.map_item(item)[1]
                    self.data[-1].training_time = Dataset.map_item(item)[2]
            #print(self.data)
            # Check for Isomorphisms
            ys = [graph.y.detach().numpy() for graph in self.data]
            ys_np = np.array(ys)
            u, ind = np.unique(ys_np, axis=0, return_index=True)
            
            self.data =[self.data[i] for i in ind]

            # Save only data with max 12 nodes and trained
            #self.data = [i for i in self.data if hasattr(i, 'acc')]
            self.data = [i for i in self.data if i.num_nodes<14]

            if prediction:
                self.data = [i for i in self.data if hasattr(i, 'acc')] 

            print(f"Saving data to cache: {file_cache}")
            torch.save(self.data, file_cache)

        else:
            print(f"Loading data from cache: {file_cache}")
            self.data = torch.load(file_cache)        
        

        ############################################
        self.train_data, self.test_data = Dataset.sample(self.data, sample_size, only_prediction)        
        self.length = len(self.train_data)

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
    
    ##########################################################################
    @staticmethod
    def map_item(item):
        # env.reset()
        epochs_train = 50
        # env.simulated_train(item, epochs_train)
        val_acc = 1 - env.get_model_stats(item, epochs_train - 1)['val_loss']/100.0
        test_acc = 1- env.get_test_loss_of_the_best_validated_architecture()/100.0
        wall_time = env.get_total_time()

        return torch.FloatTensor([val_acc]), torch.FloatTensor([test_acc]), torch.FloatTensor([wall_time]) 
    ##########################################################################
    @staticmethod
    def map_network(item):
        edges, op, hidden = convert_recipe_to_compact(item)

        output_idx = len(op)+1
        adj = torch.zeros(output_idx, output_idx)

        for edge in edges:
            adj[edge[0], edge[1]] = 1

        for edge in hidden:
            adj[edge, output_idx-1] = 1



        edge_index = torch.nonzero(adj).T
        node_attr = torch.tensor([*[i+1 for i in op], NB_NLP_ops.index('output')])
        num_nodes = len(node_attr)

        y_nodes = node_attr
        if len(node_attr) != 26:
            y_nodes = F.pad(y_nodes, pad=(0,26-len(y_nodes)), value=NB_NLP_ops.index('output'))

        x_binary = torch.nn.functional.one_hot(y_nodes, num_classes=len(NB_NLP_ops))

        scores = adj_to_scores(adj, l=325)
        y = torch.cat((x_binary.reshape(-1).float(), scores.float()))


        return Data(edge_index=edge_index.long(), x=node_attr, x_binary = x_binary, num_nodes=num_nodes, y=y, scores=scores)

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
            train_data = copy.deepcopy(dataset)
        
        return train_data, test_data 
    
    ##########################################################################
    @staticmethod
    def generate_nx11_compact(item):
        # Calculates a compact version of our dataset item to a tuple of edges, operations and hidden states
        # Needed for NAS-Bench-X11 Surrogate Model
        
        # edges

        max_ind = torch.max(item.edge_index)
        subset=list(range(max_ind))
        edge_index_sub = subgraph(subset,item.edge_index )[0]
        edge_index_sub, item.edge_index
        edges = tuple(tuple(edge.tolist()) for edge in edge_index_sub.T)

        # hidden:
        adj = to_dense_adj(item.edge_index)[0]
        h = tuple(np.nonzero(adj.numpy()[:,-1])[0])

        # ops:
        op_dict = ['in', 'activation_sigm', 'activation_tanh', 'activation_leaky_relu', \
                       'elementwise_sum', 'elementwise_prod', 'linear', 'blend']

        op = [0,0,0,0]
        ops = [NB_NLP_ops[i] for i in item.x]
        inter_node = [i for i in ops if i not in ['output', 'input']]

        for i in inter_node:
            op.append(op_dict.index(i))
        o = tuple(op)

        
        return (edges, o, h)
    
    ##########################################################################
    @staticmethod
    def get_info_generated_graph(item, dataset=None):
        # NAS-Bench-X11 Surrogate Model
        max_nodes=12
        if isinstance(item , list):
            data = []
            for graph in item:
                if hasattr(graph, "acc"):
                    continue
                compact = Dataset.generate_nx11_compact(graph)
                arch = encode_adj(compact=compact, max_nodes=max_nodes, one_hot=False, accs=None)
                try:
                    learning_curve = nbnlp_surrogate_model.predict(config=arch, representation='compact', with_noise=False,search_space='nlp')
                except ValueError:
                    recipe = convert_compact_to_recipe(compact)
                    # try:
                    #     train_losses, val_losses, test_losses = main_one_model_train(recipe)
                    #     assert len(val_losses) == 3

                    # except Exception as e:
                    val_losses = [6.5, 6.5, 6.5]
                    accs = [100 - loss for loss in val_losses]
                    arch = encode_adj(compact=compact, max_nodes=max_nodes, one_hot=False, accs=accs)
                    learning_curve = nbnlp_surrogate_model.predict(config=arch, representation='compact', with_noise=False,search_space='nlp')

                graph.val_acc = torch.FloatTensor([learning_curve[-1]/100.0])
                data.append(graph) 
                
        else:
            if hasattr(item, "acc"):
                data = item
            else:
                compact = Dataset.generate_nx11_compact(item)
                
                arch = encode_adj(compact=compact, max_nodes=max_nodes, one_hot=False, accs=None)
                
                try:
                    learning_curve = nbnlp_surrogate_model.predict(config=arch, representation='compact', with_noise=False,search_space='nlp')
                except ValueError:
                    recipe = convert_compact_to_recipe(compact)
                    # try:
                    #     train_losses, val_losses, test_losses = main_one_model_train(recipe)
                    #     assert len(val_losses) == 3

                    # except Exception as e:
                    val_losses = [6.5, 6.5, 6.5]
                    accs = [100 - loss for loss in val_losses]
                    arch = encode_adj(compact=compact, max_nodes=max_nodes, one_hot=False, accs=accs)
                    learning_curve = nbnlp_surrogate_model.predict(config=arch, representation='compact', with_noise=False,search_space='nlp')
                item.val_acc = torch.FloatTensor([learning_curve[-1]/100.0])
                data = item
                
            
        return data
    

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
    # print(ds.data)
    # for batch in ds.dataloader:
    #     print(batch)
    #     break
