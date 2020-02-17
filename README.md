# Preliminaries

Install [Anaconda](https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh)!

Pull the submodules:
```
git submodule init
git submodule update
```

# Chemprop

Publication: [Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/10.1021/acs.jcim.9b01076)

## Instructions

Make conda environment:

```
cd chemprop/
conda env create -f environment.yml
conda activate chemprop
conda install cudatoolkit=10.1 -c pytorch
```

Decompress data:
```
tar xvf data.tar.gz
```

Run microbenchmark:

```
python train.py --data_path data/tox21.csv --dataset_type classification --save_dir tox21_checkpoints
```

## Results

* GPU activity: ~30%
* CPU activity: ~100%

# n2nmn

Publication: [Learning to Reason: End-to-End Module Networks for Visual Question Answering](https://arxiv.org/abs/1704.05526)

## Instructions

Make conda environment:

```
conda create --name n2nmn
conda activate n2nmn
conda install pip
python -m pip install tensorflow-gpu==1.0.0
python -m pip install https://storage.googleapis.com/tensorflow_fold/tensorflow_fold-0.0.1-py3-none-linux_x86_64.whl
```

Get and process the data, per the README:

```
# Warning: the data is big (~18 GB)!
# Place it somewhere appropriate and change the symlink target path.
wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
unzip CLEVR_v1.0

cd n2nmn
ln -f -s CLEVR_v1.0 exp_clevr/clevr-dataset

# Preprocessing.
# This part will also consume some 30 GB of disk space.
./exp_clevr/tfmodel/vgg_net/download_vgg_net.sh  # VGG-16 converted to TF

cd ./exp_clevr/data/
python extract_visual_features_vgg_pool5.py  # feature extraction
python get_ground_truth_layout.py  # construct expert policy
python build_clevr_imdb.py  # build image collections
cd ../../
```

Run microbenchmark:

```
python exp_clevr/train_clevr_scratch.py
```

## Results

* GPU activity: ~50%

It prints messages about slow I/O, which it does when its prefetch queue is empty.

# PINN

An implementation of [RobustFill](https://arxiv.org/abs/1703.07469).

## Instructions

Make conda environment:

```
conda create --name pinn
conda activate pinn
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

Run microbenchmark:

```
cd pinn
python test_robustfill.py
```

## Results:

* GPU activity: ~70%
* CPU activity: ~100%

# Attention Is All You Need

Publication: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Instructions

Make conda environment:

```
conda create --name t2t
conda activate t2t
conda install pip
python -m pip install tensorflow-gpu==1.15.0

cd tensor2tensor
python setup.py install
```

Generate data:

```
mkdir -p t2t_data t2t_datagen t2t_train
t2t-datagen \
  --data_dir=t2t_data \
  --tmp_dir=t2t_datagen \
  --problem=translate_ende_wmt32k
```

Run microbenchmark:

```
t2t-trainer \
  --data_dir=t2t_data \
  --problem=translate_ende_wmt32k \
  --model=transformer \
  --hparams_set=transformer_base_single_gpu \
  --output_dir=t2t_train
```

## Results:

* GPU activity: ~90%
* CPU activity: ~110%
