# Word2Box: Capturing Set-Theoretic Semantics of Words using Box Embeddings
> Using boxes for word representations

## Dev Instructions
```
conda create -n word2box python=3.8
conda activate word2box
conda install pytorch==1.7.1 -c pytorch
```
```
git clone {repo-url} --recurse-submodules
cd word2box
pip install -e lib/*
pip install -e .
```
## Usage
This package will install a command accessible as follows:
```
language-modeling-with-boxes --help
```
(You can also access this via the script in `/bin/language-modeling-with-boxes`, which can be useful to provide a "handle" for debuggers, for instance.)

## Training
To get description of all the hyper paramer options run the following. 
```
language-modeling-with-boxes train --help
```
One example command to run training - 
```
bin/language-modeling-with-boxes train \
 --batch_size=4096 --box_type=BoxTensor \
 --data_device=gpu \
 --dataset= ptb `Please change this to your dataset` \
 --embedding_dim=64 \
 --eval_file=./data/similarity_datasets/ \
 --int_temp=1.9678289474987882 \
 --log_frequency=10 \
 --loss_fn=max_margin \
 --lr=0.004204091643267762 \
 --margin=5 \
 --model_type=Word2BoxConjunction \
 --n_gram=5 \
 --negative_samples=2 \
 --num_epochs=10 \
 --subsample_thresh=0.001 \
 --vol_temp=0.33243242379830407 \
 --save_model \
 --add_pad \
 -- save_dir `Please change this to your dataset` 
```
