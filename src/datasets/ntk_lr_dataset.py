from procedure_darts import TENAS
import NASBench301_self
import tqdm
import torch

dataset =  NASBench301_self.Dataset(batch_size=16, sample_size=10, only_prediction=True)
tenas=TENAS(dataset='cifar10', data_path='/home/asthana/Documents/DiGress/DiGress/data/cifar10', config_path='', seed=0)

conditional_train_data = dataset.train_data
for arch in tqdm.tqdm(conditional_train_data):
    genotype = dataset.get_genotype(arch)
    arch.genotype = genotype
    ntk = 0
    lr = 0
    for _ in range(3):
        ntk_i, lr_i = Dataset.get_nb301_ntk_lr(tenas,genotype)
        ntk += ntk_i
        lr += lr_i
    ntk = ntk/3
    lr = lr/3
    arch.ntk = torch.tensor([ntk])
    arch.lr = torch.tensor([lr])

torch.save(conditional_train_data, '/home/asthana/Documents/DiGress/DiGress/data/cifar10/lr_ntl_cifar10_test')
