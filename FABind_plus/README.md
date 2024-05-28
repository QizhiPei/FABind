<h1 align="center">
FABind+: Enhancing Molecular Docking through Improved Pocket Prediction and Pose Generation 🔥
</h1>

<div align="center">


[![](https://img.shields.io/badge/FABind+-arxiv2403.20261-red?style=plastic&logo=GitBook)](https://arxiv.org/abs/2403.20261)



[![](https://img.shields.io/badge/model-pink?style=plastic&logo=themodelsresource)](https://huggingface.co/KyGao/FABind_plus_model) 
[![](https://img.shields.io/badge/dataset-zenodo-orange?style=plastic&logo=zenodo)](https://zenodo.org/records/11352521)
[![](https://img.shields.io/badge/awesome-docking-orange?style=plastic&logo=awesomelists)](https://github.com/KyGao/awesome-docking/tree/main)
[![](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

</div>

## Overview
This repository contains the source code for the paper "[FABind+: Enhancing Molecular Docking through Improved Pocket Prediction and Pose Generation](https://arxiv.org/abs/2403.20261)". FABind+ achieves accurate docking performance with high speed compared to recent baselines. If you have questions, don't hesitate to open an issue or ask me via <im_kai@hust.edu.cn>, Qizhi Pei via <qizhipei@ruc.edu.cn>, or Lijun Wu via <lijun_wu@outlook.com>. We are happy to hear from you!

![](./imgs/pipeline.jpg)


## Setup Environment
This is an example of how to set up a working conda environment to run the code. In this example, we have cuda version==11.3, torch==1.12.0, and rdkit==2021.03.4. To make sure the pyg packages are installed correctly, we directly install them from whl.

**As the trained model checkpoint is included in the HuggingFace repository with git-lfs, you need to install git-lfs to pull the data correctly.**

```shell
sudo apt-get install git-lfs # run this if you have not installed git-lfs
git lfs install
git clone https://github.com/QizhiPei/FABind.git --recursive
conda create --name fabind python=3.8
conda activate fabind
conda install -c conda-forge graph-tool -y
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_cluster-1.6.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.1.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_sparse-0.6.15%2Bpt112cu113-cp38-cp38-linux_x86_64.whl 
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_spline_conv-1.2.1%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/pyg_lib-0.2.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install torch-geometric==2.4.0
pip install torchdrug==0.1.2 torchmetrics==0.10.2 tqdm mlcrate pyarrow accelerate Bio lmdb fair-esm tensorboard
pip install wandb spyrmsd
pip install rdkit-pypi==2021.03.4
conda install -c conda-forge openbabel # install openbabel to save .mol2 file and .sdf file at the same time
cd FABind_plus
```

## Data
Compared to FABind, we additionally add isomorphism features and construct `data_new.pt` using scripts in `fabind/tools/inject_isomorphism_to_data.py`. Everything else remains the same. We provide the processed dataset on [zenodo](https://zenodo.org/records/11352521).

If you want to train FABind+ from scratch, or reproduce the FABind+ results, you can:
If you want to train FABind from scratch, or reproduce the FABind results, you can:
1. download dataset from [zenodo](https://zenodo.org/records/11352521)
2. unzip the `zip` file and place it into `data_path` such that `data_path=pdbbind2020`

### Generate the ESM2 embeddings for the proteins
Before training or evaluation, you need to first generate the ESM2 embeddings for the proteins based on the preprocessed data above.
```shell
data_path=../data/pdbbind2020

python fabind/tools/generate_esm2_t33.py ${data_path}
```
Then the ESM2 embedings will be saved at `${data_path}/dataset/processed/esm2_t33_650M_UR50D.lmdb`.

## Model
The pre-trained regression-based model is placed at `ckpt/fabind_plus_best_ckpt.bin`, and the sampling-based model is at `ckpt/confidence_model.bin`, which will be automatically downloaded when cloning this reporsitory with `--recursive`.

You can also manually download the pre-trained model from [Hugging Face](https://huggingface.co/KyGao/FABind_plus_model)

## Regression-based FABind+

### Evaluation Results
```shell
ckpt_path=ckpt/fabind_plus_best_ckpt.bin
data_path=pdbbind2020
python fabind/test_regression_fabind.py \
    --batch_size 4 \
    --data-path ${data_path} \
    --resultFolder ./results \
    --exp-name test_exp \
    --symmetric-rmsd ${data_path}/renumber_atom_index_same_as_smiles \
    --ckpt ${ckpt_path}
```
### Inference on Custom Complexes
Here are the scripts available for inference with smiles and according pdb files.

The following script iteratively runs:
- Given smiles in `index_csv`, preprocess molecules with `num_threads` multiprocessing and save each processed molecule to `{save_pt_dir}/mol`.
- Given protein pdb files in `pdb_file_dir`, preprocess protein information and save it to `{save_pt_dir}/processed_protein.pt`.
- Load model checkpoint in `ckpt_path`, save the predicted molecule conformation in `output_dir`. Another csv file in `output_dir` indicates the smiles and according filename.

```shell
index_csv=../inference_examples/example.csv
pdb_file_dir=../inference_examples/pdb_files
num_threads=10
save_pt_dir=../inference_examples/temp_files
save_mols_dir=${save_pt_dir}/mol
ckpt_path=../ckpt/fabind_plus_best_ckpt.bin
output_dir=../inference_examples/inference_output

cd fabind

echo "======  preprocess molecules  ======"
python inference_preprocess_mol_confs.py --index_csv ${index_csv} --save_mols_dir ${save_mols_dir} --num_threads ${num_threads}

echo "======  preprocess proteins  ======"
python inference_preprocess_protein.py --pdb_file_dir ${pdb_file_dir} --save_pt_dir ${save_pt_dir}

echo "======  inference begins  ======"
python inference_fabind.py \
    --ckpt ${ckpt_path} \
    --batch_size 4 \
    --test-gumbel-soft \
    --post-optim \
    --write-mol-to-file \
    --sdf-output-path-post-optim ${output_dir} \
    --index-csv ${index_csv} \
    --preprocess-dir ${save_pt_dir}
```

### Re-training
```shell
data_path=pdbbind2020

python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"
accelerate launch fabind/main_fabind.py \
    --data-path ${data_path} --resultFolder ./results --exp-name train_fabind_plus_regression \
    --batch_size 16 --addNoise 5 --seed 224 --total-epochs 1500 --warmup-epochs 15 \
    --lr 5e-5 --lr-scheduler poly_decay --clip-grad --optim adam \
    --coord-loss-weight 1.5 --pair-distance-loss-weight 1.0 --pair-distance-distill-loss-weight 1.0 \
    --pocket-cls-loss-weight 1.0 --pocket-distance-loss-weight 0.05 --pocket-radius-loss-weight 0.05 \
    --pocket-radius-buffer 5 --min-pocket-radius 20 --use-for-radius-pred ligand --permutation-invariant \
    --distmap-pred mlp --dismap-choice npair --use-esm2-feat --dis-map-thres 15 \
    --pocket-pred-layers 1 --pocket-pred-n-iter 1 --n-iter 8 --mean-layers 5 \
    --rm-layernorm --add-attn-pair-bias --explicit-pair-embed --add-cross-attn-layer \
    --expand-clength-set --cut-train-set --random-n-iter --pocket-idx-no-noise \
    --use-ln-mlp --dropout 0.1 --mlp-hidden-scale 1 \
    --test-interval 3 --num-workers 0 --wandb
```

## Sampling-based FABind+
### Evaluation Results
```shell
data_path=pdbbind2020
ckpt_path=ckpt/confidence_model.bin
sample_size=40

python fabind/test_sampling_fabind.py \
    --batch_size 8 \
    --data-path ${data_path} \
    --resultFolder ./results \
    --exp-name test_exp \
    --ckpt ${ckpt_path} --use-clustering --infer-dropout \
    --sample-size ${sample_size} \
    --symmetric-rmsd ${data_path}/renumber_atom_index_same_as_smiles \
    --save-rmsd-dir ./rmsd_results
```

### Re-training Confidence Model
```shell
ckpt_path=ckpt/fabind_plus_best_ckpt.bin
data_path=pdbbind2020
python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"
accelerate launch fabind/train_confidence.py \
    --reload ${ckpt_path} --data-path ${data_path} --resultFolder ./results --exp-name train_confidence \
    --seed 3407 --batch_size 1 --num-copies 5 --warmup-epochs 5 --total-epochs 100 \
    --optim adamw --lr 1e-4 --lr-scheduler poly_decay \
    --ranking-loss logsigmoid --keep-cls-2A  \
    --use-clustering --dbscan-eps 9.0 --dbscan-min-samples 2 --choose-cluster-prob 0.5 --infer-dropout \
    --confidence-training  \
    --stack-mlp --confidence-dropout 0.2 --confidence-use-ln-mlp --confidence-mlp-hidden-scale 1 \
    --wandb
```


## About
### Citations
```
@article{gao2024fabind+,
  title={FABind+: Enhancing Molecular Docking through Improved Pocket Prediction and Pose Generation},
  author={Gao, Kaiyuan and Pei, Qizhi and Zhu, Jinhua and Qin, Tao and He, Kun and Liu, Tie-Yan and Wu, Lijun},
  journal={arXiv preprint arXiv:2403.20261},
  year={2024}
}
```

### Acknowledegments
We appreciate [EquiBind](https://github.com/HannesStark/EquiBind), [TankBind](https://github.com/luwei0917/TankBind), [E3Bind](https://openreview.net/forum?id=sO1QiAftQFv), [DiffDock](https://github.com/gcorso/DiffDock) and other related works for their open-sourced contributions.
