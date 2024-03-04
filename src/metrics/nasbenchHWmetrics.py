import torch
from torchmetrics import Metric, MetricCollection
from torch import Tensor
import wandb
import torch.nn as nn
from src.metrics.molecular_metrics import GeneratedNDistribution,GeneratedNodesDistribution,GeneratedEdgesDistribution,HistogramsMAE
from src.analysis.rdkit_functionsHW import compute_molecular_metrics

class CEPerClass(Metric):
    full_state_update = False
    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.softmax = torch.nn.Softmax(dim=-1)
        self.binary_cross_entropy = torch.nn.BCELoss(reduction='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model   (bs, n, d) or (bs, n, n, d)
            target: Ground truth values     (bs, n, d) or (bs, n, n, d)
        """
        #target=target[:,:,0]
        #preds=preds[:,:,0]
        target = target.reshape(-1, target.shape[-1])
        #target=target.reshape(preds.shape)
        mask = (target != 0.).any(dim=-1)
        #print(mask.shape)
        #print(self.softmax(preds[0]))
        #print(self.softmax(target[0]))

        #print('print preds',self.softmax(preds).shape)
        #print('print target',self.softmax(target).shape)
        #print('class id', self.class_id)
        
        prob = self.softmax(preds)[..., self.class_id]
        prob = prob.flatten()[mask]
        #print('TARGET',target.shape)

        target = target[:,self.class_id]
        target = target[mask]

        output = self.binary_cross_entropy(prob, target)
        self.total_ce += output
        self.total_samples += prob.numel()

    def compute(self):
        return self.total_ce / self.total_samples
    
class InpCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class C1x1CE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class C3x3CE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class ApCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class OutCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)
        
class SkipCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)
        
class NoneCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)
        
class EdgeCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)
        
        
class NoEdgeCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)

class NodeMetrics(MetricCollection):
    def __init__(self, dataset_infos):
        atom_decoder = dataset_infos.atom_decoder

        class_dict = {'inp': InpCE, '1x1': C1x1CE, '3x3': C3x3CE, 'ap': ApCE,'skip': SkipCE,'none': NoneCE, 'out': OutCE}
        #conv = {'input': 'C', 'conv1x1-bn-relu': 'N', 'conv3x3-bn-relu': 'O', 'maxpool3x3':'F', 'output':'I'}

        metrics_list = []
        for i, atom_type in enumerate(atom_decoder):
            metrics_list.append(class_dict[atom_type](i))
        #print(metrics_list)
        super().__init__(metrics_list)
        
        
class EdgeMetrics(MetricCollection):
    def __init__(self):
        edge1 = EdgeCE(0)
        #edge2=NoEdgeCE(1)
        super().__init__([edge1])
        
class TrainNASBenchHWMetricsDiscrete(nn.Module):
    def __init__(self, dataset_infos):
        super().__init__()
        self.train_atom_metrics = NodeMetrics(dataset_infos=dataset_infos)
        self.train_bond_metrics = EdgeMetrics()

    def forward(self, masked_pred_X, masked_pred_E, true_X, true_E, log: bool):
        self.train_atom_metrics(masked_pred_X, true_X)
        self.train_bond_metrics(masked_pred_E, true_E)
        if log:
            to_log = {}
            for key, val in self.train_atom_metrics.compute().items():
                to_log['train/' + key] = val.item()
            for key, val in self.train_bond_metrics.compute().items():
                to_log['train/' + key] = val.item()

            wandb.log(to_log, commit=False)

    def reset(self):
        for metric in [self.train_atom_metrics, self.train_bond_metrics]:
            metric.reset()

    def log_epoch_metrics(self, current_epoch):
        epoch_atom_metrics = self.train_atom_metrics.compute()
        epoch_bond_metrics = self.train_bond_metrics.compute()

        to_log = {}
        for key, val in epoch_atom_metrics.items():
            to_log['train_epoch/' + key] = val.item()
        for key, val in epoch_bond_metrics.items():
            to_log['train_epoch/' + key] = val.item()
        wandb.log(to_log, commit=False)

        for key, val in epoch_atom_metrics.items():
            epoch_atom_metrics[key] = val.item()
        for key, val in epoch_bond_metrics.items():
            epoch_bond_metrics[key] = val.item()

        print(f"Epoch {current_epoch}: {epoch_atom_metrics} -- {epoch_bond_metrics}")
        
        
class SamplingNASBenchHWMetrics(nn.Module):
    def __init__(self, dataset_infos, train_smiles):
        super().__init__()
        di = dataset_infos
        self.generated_n_dist = GeneratedNDistribution(di.max_n_nodes)
        self.generated_node_dist = GeneratedNodesDistribution(di.output_dims['X'])
        self.generated_edge_dist = GeneratedEdgesDistribution(di.output_dims['E'])
        #self.generated_valency_dist = ValencyDistribution(di.max_n_nodes)

        n_target_dist = di.n_nodes.type_as(self.generated_n_dist.n_dist)
        n_target_dist = n_target_dist / torch.sum(n_target_dist)
        self.register_buffer('n_target_dist', n_target_dist)

        node_target_dist = di.node_types.type_as(self.generated_node_dist.node_dist)
        node_target_dist = node_target_dist / torch.sum(node_target_dist)
        self.register_buffer('node_target_dist', node_target_dist)

        edge_target_dist = di.edge_types.type_as(self.generated_edge_dist.edge_dist)
        edge_target_dist = edge_target_dist / torch.sum(edge_target_dist)
        self.register_buffer('edge_target_dist', edge_target_dist)

        #valency_target_dist = di.valency_distribution.type_as(self.generated_valency_dist.edgepernode_dist)
        #valency_target_dist = valency_target_dist / torch.sum(valency_target_dist)
        #self.register_buffer('valency_target_dist', valency_target_dist)

        self.n_dist_mae = HistogramsMAE(n_target_dist)
        self.node_dist_mae = HistogramsMAE(node_target_dist)
        self.edge_dist_mae = HistogramsMAE(edge_target_dist)
        #self.valency_dist_mae = HistogramsMAE(valency_target_dist)

        self.train_smiles = train_smiles
        self.dataset_info = di

    def forward(self, molecules: list, name, current_epoch, val_counter, test=False):
            stability, rdkit_metrics, all_smiles = compute_molecular_metrics(molecules, self.train_smiles, self.dataset_info)

            if test:
                with open(r'final_smiles.txt', 'w') as fp:
                    for smiles in all_smiles:
                        # write each item on a new line
                        fp.write("%s\n" % smiles)
                    print('All smiles saved')

            self.generated_n_dist(molecules)
            generated_n_dist = self.generated_n_dist.compute()
            self.n_dist_mae(generated_n_dist)

            self.generated_node_dist(molecules)
            generated_node_dist = self.generated_node_dist.compute()
            self.node_dist_mae(generated_node_dist)

            self.generated_edge_dist(molecules)
            generated_edge_dist = self.generated_edge_dist.compute()
            self.edge_dist_mae(generated_edge_dist)

            #self.generated_valency_dist(molecules)
            #generated_valency_dist = self.generated_valency_dist.compute()
            #self.valency_dist_mae(generated_valency_dist)

            to_log = {}
            for i, atom_type in enumerate(self.dataset_info.atom_decoder):
                #print(generated_node_dist)
                generated_probability = generated_node_dist[i]
                print(self.node_target_dist)
                target_probability = self.node_target_dist[i]
                to_log[f'molecular_metrics/{atom_type}_dist'] = (generated_probability - target_probability).item()

            for j, bond_type in enumerate(['Edge']):
                generated_probability = generated_edge_dist[j]
                target_probability = self.edge_target_dist[j]

                to_log[f'molecular_metrics/bond_{bond_type}_dist'] = (generated_probability - target_probability).item()

            #for valency in range(6):
             #   generated_probability = generated_valency_dist[valency]
              #  target_probability = self.valency_target_dist[valency]
               # to_log[f'molecular_metrics/valency_{valency}_dist'] = (generated_probability - target_probability).item()

            wandb.log(to_log, commit=False)

            wandb.run.summary['Gen n distribution'] = generated_n_dist
            wandb.run.summary['Gen node distribution'] = generated_node_dist
            wandb.run.summary['Gen edge distribution'] = generated_edge_dist
            #wandb.run.summary['Gen valency distribution'] = generated_valency_dist

            wandb.log({'basic_metrics/n_mae': self.n_dist_mae.compute(),
                       'basic_metrics/node_mae': self.node_dist_mae.compute(),
                       'basic_metrics/edge_mae': self.edge_dist_mae.compute()}, commit=False)

            valid_unique_molecules = rdkit_metrics[1]
            textfile = open(f'graphs/{name}/valid_unique_molecules_e{current_epoch}_b{val_counter}.txt', "w")
            textfile.writelines(valid_unique_molecules)
            textfile.close()
            print("Stability metrics:", stability, "--", rdkit_metrics[0])
        

    def reset(self):
        for metric in [self.n_dist_mae, self.node_dist_mae, self.edge_dist_mae]:
            metric.reset()
