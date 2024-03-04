import sys, pathlib
import os, glob, json
import torch
import numpy as np 
import itertools
import tqdm
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F

from collections import OrderedDict

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
    from Generator import scores_to_adj, adj_to_scores
    from datasets.nasbench201 import api # nasbench
    from Settings import Settings
    from datasets.HW_NAS_Bench.hw_nas_bench_api import HWNASBenchAPI as HWAPI
else:
    from Generator import scores_to_adj, adj_to_scores
    from datasets.nasbench201 import api
    from datasets.HW_NAS_Bench.hw_nas_bench_api import HWNASBenchAPI as HWAPI
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

OPS_by_IDX_NB201 = {OP_PRIMITIVES_NB201.index(i):i for i in OP_PRIMITIVES_NB201}
OPS_NB201 = {i:OP_PRIMITIVES_NB201.index(i) for i in OP_PRIMITIVES_NB201}

ADJACENCY_NB201 = np.array([[0, 1, 1, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 1 ,0 ,0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0]])

nasbench = api.NASBench201API(os.path.join('/home/asthana/Documents/DiGress/DiGress/data/nasbench201_samples/NAS-Bench-201-v1_0-e61699.pth'))
hw_api = HWAPI(os.path.join(Settings.FOLDER,'/home/asthana/Documents/GenerativeNAS/AG-Net-main/datasets/HW_NAS_Bench/HW-NAS-Bench-v1_0.pickle'), search_space="nasbench201")

devices_NB201 = [
    'edgegpu',
    'raspi4',
    'edgetpu',
    'pixel3',
    'eyeriss',
    'fpga',
]

devices_ONEHOT_HWNB = {i:np.eye(6)[i] for i in range(6)}
devices_ONEHOT_HWNB_by_device ={i:devices_ONEHOT_HWNB[devices_NB201.index(i)] for i in devices_NB201}

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
            path = os.path.join(".","HW_NAS_Bench") 
        else:
            path = os.path.join(".","datasets","HW_NAS_Bench") #for debugging
        
        file_cache = os.path.join(path, "cache_correct_"+dataset)


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
        
        
        if dataset == 'cifar10_valid_converged':
            HW_metrics = hw_api.query_by_index(item, 'cifar10')
        else:
            HW_metrics = hw_api.query_by_index(item, dataset)

        all_latency = []
        for i in devices_NB201:
            dev_one_hot = torch.FloatTensor([devices_ONEHOT_HWNB_by_device[i]])
            latency = torch.FloatTensor([HW_metrics[i+'_latency']])
            dev_lat = torch.cat([dev_one_hot, latency.unsqueeze(0)],1)
            all_latency.append(dev_lat)

        edgegpu_l = all_latency[0]
        raspi4_l = all_latency[1]
        edgetpu_l = all_latency[2]
        pixel3_l = all_latency[3]
        eyeriss_l = all_latency[4]
        fpga_l = all_latency[5]
        
        devices_latency = torch.stack((all_latency)).squeeze(1)


        return Data(val_acc=val_acc_avg, acc=test_acc_avg, training_time=training_time, devices_latency=devices_latency, 
                   edgegpu_l=edgegpu_l, raspi4_l=raspi4_l, edgetpu_l=edgetpu_l, pixel3_l=pixel3_l, eyeriss_l=eyeriss_l, fpga_l=fpga_l)  
        

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
        matrix = np.array(to_dense_adj(data.edge_index)[0].cpu().numpy())
        try: 
            # if (matrix == ADJACENCY_NB201).all():
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
            #print(matrix)
            # data_acc = Data(acc = torch.zeros(1))
        

    ##########################################################################
    @staticmethod
    def get_info_generated_graph(item, dataset):
        data=None
        if isinstance(item , list):
            data = []
            for graph in item:
                try: 
                    data_acc = Dataset.pg_graph_to_nb201(graph, dataset)
                    graph.val_acc = data_acc.val_acc   
                    graph.acc = data_acc.acc  
                    graph.training_time = data_acc.training_time  
                    graph.edgegpu_l = data_acc.edgegpu_l
                    graph.raspi4_l = data_acc.raspi4_l
                    graph.edgetpu_l = data_acc.edgetpu_l
                    graph.pixel3_l = data_acc.pixel3_l
                    graph.eyeriss_l = data_acc.eyeriss_l
                    graph.fpga_l = data_acc.fpga_l
                    graph.devices_latency = data_acc.devices_latency
                    data.append(graph)
                    # data.append(Dataset.pg_graph_to_nb201(graph))
                # data[-1].edge_index =  graph.edge_index
                # data[-1].x = graph.x
                # data[-1].x_binary = graph.x_binary
                # # data[-1].scores = graph.scores
                # data[-1].y = graph.y
                # data[-1].num_nodes = graph.num_nodes
                except:
                    continue

        else:
            try:

                data_acc = Dataset.pg_graph_to_nb201(item, dataset)
                item.val_acc = data_acc.val_acc   
                item.acc = data_acc.acc  
                item.training_time = data_acc.training_time  
                item.edgegpu_l = data_acc.edgegpu_l
                item.raspi4_l = data_acc.raspi4_l
                item.edgetpu_l = data_acc.edgetpu_l
                item.pixel3_l = data_acc.pixel3_l
                item.eyeriss_l = data_acc.eyeriss_l
                item.fpga_l = data_acc.fpga_l
                item.devices_latency = data_acc.devices_latency
                data = item
                # data = Dataset.pg_graph_to_nb201(item)
            except:
                pass



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
                
    # print_keys(json.load(open(os.path.join(".","NASBench301","results_0.json"), "r")))
    ds = Dataset(10)
    print(ds.data)
    for batch in ds.dataloader:
        print(batch)
        break
