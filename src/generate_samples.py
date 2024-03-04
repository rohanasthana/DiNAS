# These imports are tricky because they use c++, do not move them
from rdkit import Chem
try:
    import graph_tool
except ModuleNotFoundError:
    pass

import os
from torch.utils.data import random_split, Dataset
import pathlib
import warnings
import torch.nn.functional as F
import torch
import wandb
import hydra
import omegaconf
from omegaconf import DictConfig
import time
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos, MolecularDataModule
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import numpy as np
from src import utils
from datasets import guacamol_dataset, qm9_dataset,nasbench101_dataset
from datasets.nasbench101dataset_reg_free import NASBenchDataset,NASBenchDatasetInfos, NASBenchDataModule
from datasets.spectre_dataset import SBMDataModule, Comm20DataModule, PlanarDataModule, SpectreDatasetInfos
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
from metrics.nasbenchmetrics import TrainNASBenchMetricsDiscrete, SamplingNASBenchMetrics
from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
from diffusion_model import LiftedDenoisingDiffusion
from diffusion_model_discrete_reg_free import DiscreteDenoisingDiffusion
from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from analysis.visualization import MolecularVisualization, NonMolecularVisualization,NASBenchVisualization
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.nasbenchregressor_clean import NASBenchRegressorDiscrete
from typing import Any, Sequence
import torch_geometric.utils
import tensorflow as tf
#tf.enable_eager_execution()

'''gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=22000)])
    except RuntimeError as e:
        print(e)
        '''

def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])

def laplacian_positional_encoding(A):
    """
        Graph positional encoding v/ Laplacian eigenvectors
        
    """
    A=csgraph.laplacian(A,normed=True)
    #print(A.shape)
    return A


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


warnings.filterwarnings("ignore", category=PossibleUserWarning)

class SampledData(Dataset):
    def __init__(self, item):
        item=np.array(item).transpose()
        #print(item.shape)
        self.operations = item[:][0]
        #print(self.operations)
        #print(item)
        self.adj_matrix =item[:][1]
        print(f'Dataset loaded from file')

    def __len__(self):
        return len(self.operations)

    def __getitem__(self, idx):
        n=self.adj_matrix[idx].shape[-1]
        X = F.one_hot(torch.tensor(self.operations[idx],device='cuda'), num_classes=5).float()
        edge_index, _ = torch_geometric.utils.dense_to_sparse(torch.tensor(self.adj_matrix[idx],device='cuda'))
        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float,device='cuda')
        edge_attr[:, 1] = 1
        y = torch.ones([1, 1],device='cuda').float()

        num_nodes = n * torch.ones(1, dtype=torch.long)
        data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                             y=y, idx=idx,n_nodes=num_nodes)
        
        return data
    
class SampledDataModule(MolecularDataModule):
    def __init__(self, cfg,item):
        self.datadir = cfg.dataset.datadir
        super().__init__(cfg)
        self.remove_h = cfg.dataset.remove_h
        self.item=item
        
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
            
        
        graphs = SampledData(self.item)
        test_len = len(graphs)-2
        train_len = 1
        val_len = 1
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        splits = random_split(graphs, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(1234))
        
        datasets = {'train': splits[0], 'val': splits[1], 'test': splits[2]}
        return super().prepare_datas(datasets)
        
      
            
            
@hydra.main(version_base='1.1', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    samples_left_to_generate = cfg.general.final_model_samples_to_generate
    samples_left_to_save = cfg.general.final_model_samples_to_save
    chains_left_to_save = cfg.general.final_model_chains_to_save
    number_chain_steps = cfg.general.number_chain_steps
    device = torch.device(f'cuda:{torch.cuda.current_device()}'
                      if torch.cuda.is_available()
                      else 'cpu')
    torch.cuda.set_device(device)

    samples = []
    #regressor_model = NASBenchRegressorDiscrete.load_from_checkpoint(os.path.join(cfg.general.regressor_path))
    diffusion_model=DiscreteDenoisingDiffusion.load_from_checkpoint(os.path.join(cfg.general.diffusion_path))
    #regressor_model.to('cuda')
    diffusion_model.to('cuda')
    num_classes=0
    #print(cfg.dataset)
    if cfg.dataset['name']=='nasbench201' or cfg.dataset['name']=='nasbenchHW':
        num_classes=7
    elif cfg.dataset['name']=='nasbench101':
        num_classes=5


    id = 0
    start = time.time()
    while samples_left_to_generate > 0:
        print(f'Samples left to generate: {samples_left_to_generate}/'
              f'{cfg.general.final_model_samples_to_generate}', end='', flush=True)
        print('\n')
        bs = 192
        #to_generate = min(samples_left_to_generate, bs)
        to_generate=bs
        to_save = min(samples_left_to_save, bs)
        chains_save = min(chains_left_to_save, bs)
        preds=[]
        items=[]
        sample=diffusion_model.sample_batch(id, to_generate, num_nodes=None, save_final=to_save,
                                 keep_chain=chains_save, number_chain_steps=number_chain_steps)
        for i in sample:
            samples.append(i)

        id += 1
        samples_left_to_save -= 1
        samples_left_to_generate -= 1
        chains_left_to_save -= 1
    print("Saving the generated graphs")
    filename = f'generated_samples1.txt'
    for i in range(2, 10):
        if os.path.exists(filename):
            filename = f'generated_samples{i}.txt'
        else:
            break
    #print(samples)
    with open(filename, 'w') as f:
        for item in samples:
            #print(item)
            f.write(f"N={item[0].shape[0]}\n")
            atoms = item[0].tolist()
            f.write("X: \n")
            for at in atoms:
                f.write(f"{at} ")
            f.write("\n")
            f.write("E: \n")
            for bond_list in item[1]:
                for bond in bond_list:
                    f.write(f"{bond} ")
                f.write("\n")
            f.write("\n")
    print("Saved.")
    end=time.time()
    print('Total time taken', end-start)
    
    '''
    #print("Computing sampling metrics...")
    #self.sampling_metrics.reset()
    #self.sampling_metrics(samples, self.name, self.current_epoch, self.val_counter, test=True)
    #self.sampling_metrics.reset()
    print("Done.")'''


if __name__ == '__main__':
    main()
