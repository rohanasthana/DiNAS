# These imports are tricky because they use c++, do not move them
from rdkit import Chem
try:
    import graph_tool
except ModuleNotFoundError:
    pass

import os
import pathlib
import warnings

import torch
import wandb
import hydra
import omegaconf
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

import src.utils
from src.datasets import nasbench101_dataset
from src.datasets.nasbench101dataset_reg_free import NASBenchDataset,NASBenchDatasetInfos, NASBenchDataModule
from src.datasets.nasbench201dataset_reg_free import NASBench201Dataset,NASBench201DatasetInfos, NASBench201DataModule
from src.datasets.nasbench301dataset_reg_free import NASBench301Dataset,NASBench301DatasetInfos, NASBench301DataModule
from src.datasets.nasbenchNLPdataset_reg_free import NASBenchNLPDataset,NASBenchNLPDatasetInfos, NASBenchNLPDataModule
from src.datasets.nasbenchHWdataset_reg_free import NASBenchHWDataset,NASBenchHWDatasetInfos, NASBenchHWDataModule
from src.datasets.nasbenchImageNetdataset_reg_free import NASBenchImageNetDataset,NASBenchImageNetDatasetInfos, NASBenchImageNetDataModule
from src.datasets.spectre_dataset import SBMDataModule, Comm20DataModule, PlanarDataModule, SpectreDatasetInfos
from src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
from src.metrics.nasbenchmetrics import TrainNASBenchMetricsDiscrete, SamplingNASBenchMetrics

from src.metrics.nasbench201metrics import TrainNASBench201MetricsDiscrete, SamplingNASBench201Metrics
from src.metrics.nasbench301metrics import TrainNASBench301MetricsDiscrete, SamplingNASBench301Metrics
from src.metrics.nasbenchNLPmetrics import TrainNASBenchNLPMetricsDiscrete, SamplingNASBenchNLPMetrics
from src.metrics.nasbenchHWmetrics import TrainNASBenchHWMetricsDiscrete, SamplingNASBenchHWMetrics
from src.metrics.nasbenchImageNetmetrics import TrainNASBenchImageNetMetricsDiscrete, SamplingNASBenchImageNetMetrics
from src.analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
from src.diffusion_model import LiftedDenoisingDiffusion
from src.diffusion_model_discrete_reg_free import DiscreteDenoisingDiffusion
from src.metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from src.analysis.visualization import MolecularVisualization, NonMolecularVisualization,NASBenchVisualization, NASBench201Visualization,NASBench301Visualization,NASBenchNLPVisualization,NASBenchHWVisualization, NASBenchImageNetVisualization
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures

warnings.filterwarnings("ignore", category=PossibleUserWarning)

import tensorflow as tf
'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=22000)])
    except RuntimeError as e:
        print(e)
'''
        
def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    new_cfg = model.cfg

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'

    new_cfg = cfg.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': f'graph_ddm_{cfg.dataset.name}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg


@hydra.main(version_base='1.1', config_path='./configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    #torch.cuda.set_per_process_memory_fraction(0.5)
    if dataset_config["name"] in ['sbm', 'comm-20', 'planar']:
        if dataset_config['name'] == 'sbm':
            datamodule = SBMDataModule(cfg)
            sampling_metrics = SBMSamplingMetrics(datamodule.dataloaders)
        elif dataset_config['name'] == 'comm-20':
            datamodule = Comm20DataModule(cfg)
            sampling_metrics = Comm20SamplingMetrics(datamodule.dataloaders)
        #elif dataset_config['name']== 'nasbench101':
            #datamodule=NASBenchDataModule(cfg)
            #sampling_metrics = NASBenchSamplingMetrics(datamodule.dataloaders)
            
        else:
            datamodule = PlanarDataModule(cfg)
            sampling_metrics = PlanarSamplingMetrics(datamodule.dataloaders)

        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization()

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}

    elif dataset_config["name"] in ['qm9', 'guacamol', 'moses']:
        if dataset_config["name"] == 'qm9':
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            datamodule.prepare_data()
            train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
        elif dataset_config['name'] == 'guacamol':
            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            datamodule.prepare_data()
            train_smiles = None

        elif dataset_config.name == 'moses':
            datamodule = moses_dataset.MOSESDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            datamodule.prepare_data()
            train_smiles = None
            
        else:
            raise ValueError("Dataset not implemented")

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        if cfg.model.type == 'discrete':
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        else:
            train_metrics = TrainMolecularMetrics(dataset_infos)
            


        # We do not evaluate novelty during training
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
        
                
    elif dataset_config.name=='nasbench101':
        datamodule=NASBenchDataModule(cfg)
        dataset_infos = NASBenchDatasetInfos(datamodule, dataset_config)
        #datamodule.prepare_data()
        train_smiles = None
        #extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()
        dataset_infos.compute_input_output_dims(datamodule, extra_features, domain_features)
        
        train_metrics=TrainNASBenchMetricsDiscrete(dataset_infos)
        sampling_metrics=SamplingNASBenchMetrics(dataset_infos, train_smiles)
        visualization_tools=NASBenchVisualization(dataset_infos,cfg.dataset.remove_h)
        
        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}

    elif dataset_config.name=='nasbench201':
        dataset_choice=dataset_config['dataset']
        print('Training on ', dataset_choice)
        datamodule=NASBench201DataModule(cfg,dataset_choice)
        dataset_infos = NASBench201DatasetInfos(datamodule, dataset_config)
        #datamodule.prepare_data()
        train_smiles = None
        #extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()
        dataset_infos.compute_input_output_dims(datamodule, extra_features, domain_features)
        
        train_metrics=TrainNASBench201MetricsDiscrete(dataset_infos)
        sampling_metrics=SamplingNASBench201Metrics(dataset_infos, train_smiles)
        visualization_tools=NASBench201Visualization(dataset_infos,cfg.dataset.remove_h)
        
        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
        
    elif dataset_config.name=='nasbench301':
        dataset_choice=dataset_config['dataset']
        print('Training on ', dataset_choice)
        datamodule=NASBench301DataModule(cfg,dataset_choice)
        dataset_infos = NASBench301DatasetInfos(datamodule, dataset_config)
        #datamodule.prepare_data()
        train_smiles = None
        #extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()
        dataset_infos.compute_input_output_dims(datamodule, extra_features, domain_features)
        
        train_metrics=TrainNASBench301MetricsDiscrete(dataset_infos)
        sampling_metrics=SamplingNASBench301Metrics(dataset_infos, train_smiles)
        visualization_tools=NASBench301Visualization(dataset_infos,cfg.dataset.remove_h)
        
        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
        
    elif dataset_config.name=='nasbenchNLP':
        dataset_choice=dataset_config['dataset']
        print('Training on ', dataset_choice)
        datamodule=NASBenchNLPDataModule(cfg,dataset_choice)
        dataset_infos = NASBenchNLPDatasetInfos(datamodule, dataset_config)
        #datamodule.prepare_data()
        train_smiles = None
        #extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()
        dataset_infos.compute_input_output_dims(datamodule, extra_features, domain_features)
        
        train_metrics=TrainNASBenchNLPMetricsDiscrete(dataset_infos)
        sampling_metrics=SamplingNASBenchNLPMetrics(dataset_infos, train_smiles)
        visualization_tools=NASBenchNLPVisualization(dataset_infos,cfg.dataset.remove_h)
        
        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
        
    elif dataset_config.name=='nasbenchHW':
        dataset_choice=dataset_config['dataset']
        print('Training on ', dataset_choice)
        datamodule=NASBenchHWDataModule(cfg,dataset_choice)
        dataset_infos = NASBenchHWDatasetInfos(datamodule, dataset_config)
        #datamodule.prepare_data()
        train_smiles = None
        #extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()
        dataset_infos.compute_input_output_dims(datamodule, extra_features, domain_features)
        
        train_metrics=TrainNASBenchHWMetricsDiscrete(dataset_infos)
        sampling_metrics=SamplingNASBenchHWMetrics(dataset_infos, train_smiles)
        visualization_tools=NASBenchHWVisualization(dataset_infos,cfg.dataset.remove_h)
        
        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
        
    elif dataset_config.name=='nasbenchImageNet':
        dataset_choice=dataset_config['dataset']
        print('Training on ', dataset_choice)
        datamodule=NASBenchImageNetDataModule(cfg,dataset_choice)
        dataset_infos = NASBenchImageNetDatasetInfos(datamodule, dataset_config)
        #datamodule.prepare_data()
        train_smiles = None
        #extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()
        dataset_infos.compute_input_output_dims(datamodule, extra_features, domain_features)
        
        train_metrics=TrainNASBenchImageNetMetricsDiscrete(dataset_infos)
        sampling_metrics=SamplingNASBenchImageNetMetrics(dataset_infos, train_smiles)
        visualization_tools=NASBenchImageNetVisualization(dataset_infos,cfg.dataset.remove_h)
        
        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    utils.create_folders(cfg)
    cfg = setup_wandb(cfg)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1,save_weights_only=True)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1,save_weights_only=True)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == 'test':
        print("[WARNING]: Run is called 'test' -- it will run in debug mode on 20 batches. ")
    elif name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      accelerator='gpu' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu',
                      devices=cfg.general.gpus if torch.cuda.is_available() and cfg.general.gpus > 0 else None,
                      limit_train_batches=20 if name == 'test' else None,
                      limit_val_batches=20 if name == 'test' else None,
                      limit_test_batches=20 if name == 'test' else None,
                      val_check_interval=cfg.general.val_check_interval,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      strategy='ddp' if cfg.general.gpus > 1 else None,
                      enable_progress_bar=False,
                      callbacks=callbacks,
                      logger=[])

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule)
    else:
        # Start by evaluating test_only_path
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    setup_wandb(cfg)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
