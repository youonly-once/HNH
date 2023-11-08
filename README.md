# HNH
***********************************************************************************************************

This repository is for ["High-order nonlocal Hashing for unsupervised
cross-modal retrieval"](https://sci-hub.se/https://doi.org/10.1007/s11280-020-00859-y) 


***********************************************************************************************************

## Usage
### Requirements
- python == 3.11.5
- pytorch == 2.1.0
- torchvision
- CV2
- PIL
- h5py

### Datasets
For datasets, we follow [Deep Cross-Modal Hashing's Github (Jiang, CVPR 2017)](https://github.com/jiangqy/DCMH-CVPR2017/tree/master/DCMH_matlab/DCMH_matlab). You can download these datasets from:
- Wikipedia articles, [Link](http://www.svcl.ucsd.edu/projects/crossmodal/)
- MIRFLICKR25K, [[OneDrive](https://pkueducn-my.sharepoint.com/:f:/g/personal/zszhong_pku_edu_cn/EpLD8yNN2lhIpBgQ7Kl8LKABzM68icvJJahchO7pYNPV1g?e=IYoeqn)], [[Baidu Pan](https://pan.baidu.com/s/1o5jSliFjAezBavyBOiJxew), password: 8dub]
- NUS-WIDE (top-10 concept), [[OneDrive](https://pkueducn-my.sharepoint.com/:f:/g/personal/zszhong_pku_edu_cn/EoPpgpDlPR1OqK-ywrrYiN0By6fdnBvY4YoyaBV5i5IvFQ?e=kja8Kj)], [[Baidu Pan](https://pan.baidu.com/s/1GFljcAtWDQFDVhgx6Jv_nQ), password: ml4y]


### Process

__The following experiment results are the average values, if you demand for better results, please run the experiment a few more times (2~5).__

- Clone this repo: `git clone https://github.com/youonly-once/HNH.git`.
- Change the 'dataPath' in `default_config.yml` to where you place the datasets.
- An example to train a model:
```bash
python main.py
```
- Modify the parameter `eval = True` in `default_config.yml` for validation.
- Ablation studies (__optional__): if you want to evaluate other components of our HNH, please refer to our paper and `default_config.yml`.

***********************************************************************************************************


All rights are reserved by the authors.
***********************************************************************************************************
