import os

import torch
from torch.utils.data import random_split, Dataset
import torch_geometric.utils
from src.datasets.qm9_dataset import QM9Dataset
import pathlib
from typing import Any, Sequence
import torch.nn.functional as F
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos, MolecularDataModule


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


class SelectMuTransform:
    def __call__(self, data):
        data.y = data.y[..., :1]
        return data


class SelectHOMOTransform:
    def __call__(self, data):
        data.y = data.y[..., 1:]
        return data


class NASBenchDataset(Dataset):
    def __init__(self, data_file):
        """ This class can be used to load the comm20, sbm and planar datasets. """
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
        self.filename = os.path.join(base_path, data_file)
        self.data=torch.load(self.filename)
        #print(self.data[0])
        self.adjs=[]
        self.operations=[]
        self.trainable_params=[]
        self.training_time=[]
        self.train_acc=[]
        self.val_acc=[]
        self.test_acc=[]
        for i in self.data:
            
            self.adjs.append(i['module_adjacency'])
            self.operations.append(i['module_operations'])
            #self.trainable_params.append(i['trainable_parameters'])
            #self.training_time.append(i['training_time'])
            self.train_acc.append(i['train_accuracy'])
            self.val_acc.append(i['validation_accuracy'])
            self.test_acc.append(i['test_accuracy'])
            
        print(f'Dataset {self.filename} loaded from file')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        adj = self.adjs[idx]
        types = {'input': 0, 'maxpool3x3': 1, 'conv1x1-bn-relu': 2, 'conv3x3-bn-relu': 3, 'output': 4}
        type_idx=[]
        adj=torch.from_numpy(adj)
        
        #print(adj)
        n = adj.shape[-1]
        for i in self.operations[idx]:
            #print(i)
            type_idx.append(types[i])
            
        X = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
        #y = torch.zeros([1, 0]).float()
        y=torch.tensor([self.val_acc[idx]*100]).unsqueeze(-1)
        #print(y)
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
        edge_attr[:, 1] = 1
        num_nodes = n * torch.ones(1, dtype=torch.long)
        data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                         y=y, idx=idx,n_nodes=num_nodes)
        return data
        
    '''def process(self):
        RDLogger.DisableLog('rdApp.*')

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'I': 4}
        bonds = {BT.SINGLE: 0}

        #target_df = pd.read_csv(self.split_paths[self.file_idx], index_col=0)
        #target_df.drop(columns=['mol_id'], inplace=True)

        #with open(self.raw_paths[-1], 'r') as f:
         #   skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.filename, removeHs=False, sanitize=False)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip or i not in target_df.index:
                continue

            N = mol.GetNumAtoms()

            type_idx = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds)+1).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
            y = torch.zeros((1, 0), dtype=torch.float)

            if self.remove_h:
                type_idx = torch.tensor(type_idx).long()
                to_keep = type_idx > 0
                edge_index, edge_attr = subgraph(to_keep, edge_index, edge_attr, relabel_nodes=True,
                                                 num_nodes=len(to_keep))
                x = x[to_keep]
                # Shift onehot encoding to match atom decoder
                x = x[:, 1:]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])'''
    
    
class NASBench101Dataset(NASBenchDataset):
    def __init__(self):
        super().__init__('/home/asthana/Documents/DiGress/DiGress/data/nasbench101_samples/nasbench101_full.pt')
        
        
class NASBenchDataModule(MolecularDataModule):
    def __init__(self, cfg,regressor:bool=False):
        self.datadir = cfg.dataset.datadir
        self.regressor=regressor
        super().__init__(cfg)
        self.remove_h = cfg.dataset.remove_h
        
    def prepare_dataset(self,cfg):
        target = getattr(self.cfg.general, 'guidance_target', None)
        regressor = getattr(self, 'regressor', None)
        if regressor and target == 'mu':
            transform = SelectMuTransform()
        elif regressor and target == 'homo':
            transform = SelectHOMOTransform()
        elif regressor and target == 'both':
            transform = None
        else:
            transform = RemoveYTransform()
            
        
        graphs = NASBench101Dataset()
        test_len = int(round(len(graphs) * 0.2))
        train_len = int(round((len(graphs) - test_len) * 0.8))
        val_len = len(graphs) - train_len - test_len
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        splits = random_split(graphs, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(1234))
        
        datasets = {'train': splits[0], 'val': splits[1], 'test': splits[2]}
        return super().prepare_datas(datasets)
    
        '''base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': QM9Dataset(stage='train', root=root_path, remove_h=self.cfg.dataset.remove_h,
                                        target_prop=target, transform=RemoveYTransform()),
                    'val': QM9Dataset(stage='val', root=root_path, remove_h=self.cfg.dataset.remove_h,
                                      target_prop=target, transform=RemoveYTransform()),
                    'test': QM9Dataset(stage='test', root=root_path, remove_h=self.cfg.dataset.remove_h,
                                       target_prop=target, transform=transform)}
        super().prepare_datas(datasets)'''
    
    
    
    
'''class QM9DataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        super().__init__(cfg)
        self.remove_h = cfg.dataset.remove_h

    def prepare_data(self) -> None:
        target = getattr(self.cfg.general, 'guidance_target', None)
        regressor = getattr(self, 'regressor', None)
        if regressor and target == 'mu':
            transform = SelectMuTransform()
        elif regressor and target == 'homo':
            transform = SelectHOMOTransform()
        elif regressor and target == 'both':
            transform = None
        else:
            transform = RemoveYTransform()

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': QM9Dataset(stage='train', root=root_path, remove_h=self.cfg.dataset.remove_h,
                                        target_prop=target, transform=RemoveYTransform()),
                    'val': QM9Dataset(stage='val', root=root_path, remove_h=self.cfg.dataset.remove_h,
                                      target_prop=target, transform=RemoveYTransform()),
                    'test': QM9Dataset(stage='test', root=root_path, remove_h=self.cfg.dataset.remove_h,
                                       target_prop=target, transform=transform)}
        super().prepare_data(datasets)

    
    
class NASBenchDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        INPUT = 'input'
        OUTPUT = 'output'
        CONV1X1 = 'conv1x1-bn-relu'
        CONV3X3 = 'conv3x3-bn-relu'
        MAXPOOL3X3 = 'maxpool3x3'
        self.ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT]
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        #print(self.n_nodes)
        self.node_types = self.datamodule.node_types()              # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        #super().compute_input_output_dims(self.datamodule)
        super().complete_infos(self.n_nodes, self.node_types)'''

class NASBenchDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
            #self.atom_encoder = {'INP': 0, '1x1C': 1, '3x3C': 2, '3X3M':3, 'OUT':4}
            self.atom_encoder = {'input': 0, 'maxpool3x3': 1, 'conv1x1-bn-relu': 2, 'conv3x3-bn-relu': 3, 'output': 4}
            
            self.atom_decoder = ['input', 'maxpool3x3', 'conv1x1-bn-relu', 'conv3x3-bn-relu','output']
            self.num_atom_types = 5
            #self.valencies = [4, 3, 2, 1]
            #self.atom_weights = {0: 12, 1: 14, 2: 16, 3: 19}
            self.max_n_nodes = 7
            #self.max_weight = 150
            datamodule.prepare_dataset(cfg)
            
            self.n_nodes = datamodule.node_counts()
            self.node_types = datamodule.node_types()
            self.edge_types = datamodule.edge_counts()###
            #print(self.n_nodes)
            #print(self.node_types)
            #print(self.edge_types)
            

            super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
            #self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            #self.valency_distribution[0: 6] = torch.tensor([2.6071e-06, 0.163, 0.352, 0.320, 0.16313, 0.00073])
       
        
        
