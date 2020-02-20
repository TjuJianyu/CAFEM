# Source code for "Cross-data Automatic Feature Engineering viaMeta-learning and Reinforcement Learning"

## Introduction 

## Packages 
CAFEM relays on several common packages: 

tensorflow 

sklearn

numpy

multiprocessing 

tqdm 

## Training
### Train FeL with Order-1 Operators on 20 datasets
```
for val in 620 589 586 618 616 607 795 770 1480 913 735 1067 917 4154 1049 806 1566 954 1038 825
do
    python single_afem.py --num_epochs 100 --dataset $val --cuda 0 --n_jobs 1 --buffer_size 1000 --num_episodes 1 --out_dir ../out/skip_$val 
done
```

### Train FeL with Order-1 & Order-2 operators on 20 datasets


### Train CAFEM on 100 datasets and apply on each of other 20 datasets


```
python cafem.py --multiprocessing 6 --out_dir ../out/ml --num_episodes 5
```

```
python single_afem.py --load_weight ../out/ml/model/model_5.ckpt --dataset 1049 --out_dir ../out/o2_5_1049 --num_epochs 50 --buffer_size 1000 --num_episodes 1
```


