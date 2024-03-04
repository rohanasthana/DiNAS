import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
from torchmetrics import MeanSquaredError, MeanAbsoluteError

from src.models.transformer_clean import GraphTransformer
from src.diffusion.noise_schedule import PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from src.diffusion import diffusion_utils
from src.metrics.abstract_metrics import NLL, SumExceptBatchKL, SumExceptBatchMetric
from src.metrics.train_metrics import TrainLossDiscrete
import src.utils as utils
import numpy as np
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos, MolecularDataModule

class SampledDatasetInfos(AbstractDatasetInfos):
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

def reset_metrics(metrics):
    for metric in metrics:
        metric.reset()


class NASBenchRegressorDiscrete(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.args = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.num_classes = dataset_infos.num_classes
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_y_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_y_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_y_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_y_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.save_hyperparameters(ignore=[train_metrics, sampling_metrics])
        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features
        cuda=False

        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU(),
                                     )

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        # Marginal transition model
        node_types = self.dataset_info.node_types.float()
        x_marginals = node_types / torch.sum(node_types)

        edge_types = self.dataset_info.edge_types.float()
        e_marginals = edge_types / torch.sum(edge_types)
        print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
        self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                          y_classes=self.ydim_output)

        self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                            y=torch.ones(self.ydim_output) / self.ydim_output)

        self.save_hyperparameters(ignore=[train_metrics, sampling_metrics])

        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

        self.train_loss = MeanSquaredError(squared=True)
        self.val_loss = MeanAbsoluteError()
        self.test_loss = MeanAbsoluteError()
        self.best_val_mae = 1e8

        self.val_loss_each = [MeanAbsoluteError().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) for i
                              in range(2)]
        self.test_loss_each = [MeanAbsoluteError().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) for
                               i in range(2)]
        self.target_dict = {0: "mu", 1: "homo"}

    def training_step(self, data, i):
        # input zero y to generate noised graphs
        target = data.y.clone()
        data.y = torch.zeros(data.y.shape[0], 1).type_as(data.y)
        #print(data.y.shape)

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        #noisy_data = self.apply_noise(X, E, data.y, node_mask)
        #extra_data = self.compute_extra_data(noisy_data)
        #z_t = utils.PlaceHolder(X=X_t, E=E_t, y=data.y).type_as(X).mask(node_mask)
        pred = self.forward(X, E, data.y, node_mask)
        #print(pred)
        mse = self.compute_train_loss(pred, target, log=i % self.log_every_steps == 0)
        return {'loss': mse}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.train.lr, amsgrad=True, weight_decay=1e-12)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                # "monitor": "val_loss",
            },
        }

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        print("Size of the input features", self.Xdim, self.Edim, self.ydim)

    def on_train_epoch_start(self) -> None:
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        train_mse = self.train_loss.compute()

        to_log = {"train_epoch/mse": train_mse}
        print(f"Epoch {self.current_epoch}: train_mse: {train_mse :.3f} -- {time.time() - self.start_epoch_time:.1f}s ")

        wandb.log(to_log)
        self.train_loss.reset()

    def on_validation_epoch_start(self) -> None:
        self.val_loss.reset()
        reset_metrics(self.val_loss_each)

    def validation_step(self, data, i):
        # input zero y to generate noised graphs
        target = data.y.clone()
        #print('Target is',target)
        data.y = torch.zeros(data.y.shape[0], 1).type_as(data.y)

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        #noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        #extra_data = self.compute_extra_data(noisy_data)
        #print(noisy_data.shape)
        #print(extra_data.shape)
        #print(node_mask.shape)
        X, E = dense_data.X, dense_data.E
        #noisy_data = self.apply_noise(X, E, data.y, node_mask)
        #extra_data = self.compute_extra_data(noisy_data)
        #z_t = utils.PlaceHolder(X=X_t, E=E_t, y=data.y).type_as(X).mask(node_mask)
        #pred = self.forward(X, E, y, node_mask)
        #print(pred)
        pred = self.forward(X,E,data.y, node_mask)
        mae = self.compute_val_loss(pred, target)
        # self.log('val_loss', mae, prog_bar=True, on_step=False, on_epoch=True)
        return {'val_loss': mae}
    
    '''def test_step(self,data,i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        architecture_list=[]
        #print(len(data))
        n_nodes = self.node_dist.sample_n(98, self.device)
        pred = self.forward(X, E, data.y, node_mask)
        molecule_list = []
        for i in range(98):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        best_arc=molecule_list[np.argmax(pred.y.cpu())]
        print(best_arc)
        #E_pred=E[np.argmax(pred.y.cpu())]
        X_best,E_best=best_arc.x, best_arc.edge_attr
        #print(data_pred)
        print(X_best,E_best)
        print("Saving the generated graphs")
        filename = f'generated_samples1.txt'
        with open(filename, 'a') as f:
            f.write(f"N={X_best.shape[0]}\n")
            atoms = X_best.tolist()
            f.write("X: \n")
            for at in atoms:
                f.write(f"{at} ")
            f.write("\n")
            f.write("E: \n")
            for bond_list in E_best:
                for bond in bond_list:
                    f.write(f"{bond} ")
                f.write("\n")
            f.write("\n")
        print("Saved.")
        #print("Computing sampling metrics...")
        #self.sampling_metrics.reset()
        #self.sampling_metrics(samples, self.name, self.current_epoch, self.val_counter, test=True)
        #self.sampling_metrics.reset()
        print("Done.")'''
        

    def validation_epoch_end(self, outs) -> None:
        val_mae = self.val_loss.compute()
        to_log = {"val/epoch_mae": val_mae}
        print(f"Epoch {self.current_epoch}: val_mae: {val_mae :.3f}")
        wandb.log(to_log)
        self.log('val/epoch_mae', val_mae, on_epoch=True, on_step=False)

        if val_mae < self.best_val_mae:
            self.best_val_mae = val_mae
        print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_mae, self.best_val_mae))

        if self.args.general.guidance_target == 'both':
            print('Val loss each target:')
            for i in range(2):
                mae_each = self.val_loss_each[i].compute()
                print(f"Target {self.target_dict[i]}: val_mae: {mae_each :.3f}")
                to_log_each = {f"val_epoch/{self.target_dict[i]}_mae": mae_each}
                wandb.log(to_log_each)

        self.val_loss.reset()
        reset_metrics(self.val_loss_each)

    def on_test_epoch_start(self) -> None:
        self.test_loss.reset()
        reset_metrics(self.test_loss_each)

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def compute_val_loss(self, pred, target):
        """Computes MAE.
           pred: (batch_size, n, total_features)
           target: (batch_size, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """

        #print(pred.y.shape)
        #print(target.shape)
        target=target.reshape(target.shape[0],1)
        for i in range(pred.y.shape[1]):
            mae_each = self.val_loss_each[i](pred.y[:, i], target[:, i])

        mae = self.val_loss(pred.y, target)
        return mae

    def forward(self, X, E, y, node_mask):
        #X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        #E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        #y = torch.hstack((y, y)).float()
        #print(X.shape,y.shape)
        #print(E.shape,node_mask.shape)
        
        return self.model(X, E, y, node_mask)

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)

        t = noisy_data['t']

        assert extra_X.shape[-1] == 0, 'The regressor model should not be used with extra features'
        assert extra_E.shape[-1] == 0, 'The regressor model should not be used with extra features'
        return utils.PlaceHolder(X=extra_X, E=extra_E, y=t)

    def compute_train_loss(self, pred, target, log: bool):
        """
           pred: (batch_size, n, total_features)
               pred_epsX: bs, n, dx
               pred_epsy: bs, n, n, dy
               pred_eps_z: bs, dz
           data: dict
           noisy_data: dict
           Output: mse (size 1)
       """
        target=target.reshape(target.shape[0],1)
        mse = self.train_loss(pred.y, target)

        if log:
            wandb.log({"train_loss/batch_mse": mse.item()}, commit=True)
        return mse
