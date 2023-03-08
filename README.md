# Behavior-aware Account De-anonymization on Ethereum Interaction Graph

This is a Python implementation of Ethident, as described in the following:
> Behavior-aware Account De-anonymization on Ethereum Interaction Graph


## Requirements
For hardware configuration, the experiments are conducted at Ubuntu 18.04.5 LTS with the Intel(R) Xeon(R) Gold 5218R CPU @ 2.10GHz, and NVIDIA Tesla V100S GPU (with 40GB memory each).
For software configuration, all model are implemented in
- Python 3.7
- Pytorch-Geometric 2.0.3
- Pytorch 1.8.0
- Scikit-learn 0.24.1
- CUDA 10.2


## Data
Download data from this [page](https://www.notion.so/jjzhou/Ethident-Data-861199675dc7454eb36157eeee09cf5b) and place it under the 'data/' path.


## Usage
Execute the following bash commands in the same directory where the code resides:
  ```bash
  $ python main_ggc.py -l i --hop 2 -ess Volume -layer 2 --pooling max --hidden_dim 128 --batch_size 32 --lr 0.001 --dropout 0.2 -undir 1 --aug edgeRemove+identity --aug_prob1 0.1
  ```
More parameter settings can be found in 'utils/parameters.py'.


## Citation

If you find this work useful, please cite the following:

```bib
@article{zhou2022behavior,
  title={Behavior-aware account de-anonymization on ethereum interaction graph},
  author={Zhou, Jiajun and Hu, Chenkai and Chi, Jianlei and Wu, Jiajing and Shen, Meng and Xuan, Qi},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={17},
  pages={3433--3448},
  year={2022},
  publisher={IEEE}
}
```

