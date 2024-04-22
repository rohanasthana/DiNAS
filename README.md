# <center> Multi-conditioned Graph Diffusion for Neural Architecture Search </center>
Rohan Asthana, Joschua Conrad, Youssef Dawoud, Maurits Ortmanns, Vasileios Belagiannis

This repository contains the code for the paper titled "Multi-conditioned Graph Diffusion for Neural Architecture Search" [\[link\]](https://openreview.net/forum?id=5VotySkajV).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-conditioned-graph-diffusion-for-neural/neural-architecture-search-on-nas-bench-101)](https://paperswithcode.com/sota/neural-architecture-search-on-nas-bench-101?p=multi-conditioned-graph-diffusion-for-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-conditioned-graph-diffusion-for-neural/neural-architecture-search-on-nas-bench-201-1)](https://paperswithcode.com/sota/neural-architecture-search-on-nas-bench-201-1?p=multi-conditioned-graph-diffusion-for-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-conditioned-graph-diffusion-for-neural/neural-architecture-search-on-nas-bench-201-2)](https://paperswithcode.com/sota/neural-architecture-search-on-nas-bench-201-2?p=multi-conditioned-graph-diffusion-for-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-conditioned-graph-diffusion-for-neural/neural-architecture-search-on-nas-bench-301)](https://paperswithcode.com/sota/neural-architecture-search-on-nas-bench-301?p=multi-conditioned-graph-diffusion-for-neural)


## Abstract
 Neural architecture search automates the design of neural network architectures usually by exploring a large and thus complex architecture search space. To advance the architecture search, we present a graph diffusion-based NAS approach that uses discrete conditional graph diffusion processes to generate high-performing neural network architectures. We then propose a multi-conditioned classifier-free guidance approach applied to graph diffusion networks to jointly impose constraints such as high accuracy and low hardware latency. Unlike the related work, our method is completely differentiable and requires only a single model training. In our evaluations, we show promising results on six standard benchmarks, yielding novel and unique architectures at a fast speed, i.e. less than 0.2 seconds per architecture. Furthermore, we demonstrate the generalisability and efficiency of our method through experiments on ImageNet dataset.


- `nasbench101`: for the NAS-Bench-101 benchmark
- `nasbench201`: for the NAS-Bench-201 benchmark
- `nasbench301`: for the NAS-Bench-301 benchmark
- `nasbenchNLP`: for the NAS-Bench-NLP benchmark
- `nasbenchHW`: for the NAS-Bench-HW benchmark

## Getting Started

To get started with the DiNAS project, follow these steps:

1. Clone the repository: `git clone https://github.com/rohanasthana/DiNAS.git`
2. Load the conda environment 'environment.yml' using the command `conda env create -f environment.yml`
3. Run the training process: `python main_reg_free.py --dataset nasbench101`

## Cite this paper
```
@article{
asthana2024multiconditioned,
title={Multi-conditioned Graph Diffusion for Neural Architecture Search},
author={Rohan Asthana and Joschua Conrad and Youssef Dawoud and Maurits Ortmanns and Vasileios Belagiannis},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=5VotySkajV},
note={}
}
```


## License

This project is licensed under the [MIT License](LICENSE).
