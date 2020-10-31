import argparse

parm = argparse.ArgumentParser()


#global
parm.add_argument('--multiprocessing', type=int,default=0,help='multiprocessing or not')

parm.add_argument('--cuda',type=str,default='0',help='cuda id')
#meta learning train
parm.add_argument('--meta_batch_size',type=int,default=30,help='number of task sampled per meta-update')
parm.add_argument('--meta_lr',type=float,default=0.001,help='the base learning rate of the generator')
parm.add_argument('--update_batch_size',type=int,default=15*5,help='number of examples used for inner gradient update (K for K-shot learning).')
parm.add_argument('--update_lr', type=float,default=1e-3, help='step size alpha for inner gradient update.')
parm.add_argument('--num_updates',type=int,default=1, help='number of inner gradient updates during training.')
parm.add_argument('--out_dir',type=str,default='../out/795/', help='out dir')
parm.add_argument('--save_interval',type=int,default=100,help='round to save model')
#fe
parm.add_argument('--opt_type',type=str,default='o1',help='o1 / o2')
parm.add_argument('--qsa_size',type=int,default=100,help='number of QSA bins for each class')
parm.add_argument('--history_size',type=int,default=5,help='number of history steps')
parm.add_argument('--seed',type=int,default=666,help='seed for classification/regression cross validation')
parm.add_argument('--depth',type=int,default=5,help='depth of transformation operators')
parm.add_argument('--budget',type=int,default=10,help='max features to generate for each feature')
parm.add_argument('--n_jobs',type=int,default=1,help='n_jobs')
parm.add_argument('--evaluatertype',type=str,default='rf',help='rf or lr')
#train
parm.add_argument('--buffer_size',type=int,default=1000,help="buffer_size")
parm.add_argument('--gamestep',type=int,default=15,help='max search step for each feature')
parm.add_argument('--num_epochs',type=int,default=100,help='train epochs for the whole dataset')
parm.add_argument('--num_local_epochs',type=int,default=1,help='train epochs for each feature')
parm.add_argument('--num_episodes',type=int,default=10,help='game episodes')
parm.add_argument('--tau',type=float,default=0.95,help='update target net parameter rate')
parm.add_argument('--gamma',type=float,default=0.99,help='reward discount rate')
parm.add_argument('--epsilon',type=float,default=1,help='init epsilon')
parm.add_argument('--batch_size',type=int,default=32,help='batch size for update modelnetwork')
parm.add_argument('--num_optimize_steps',type=int,default=10,help="number of update steps")
parm.add_argument('--eval_interval',type=int,default=1,help='round to evaluate')
parm.add_argument('--min_epsilon',type=float,default=0.15,help='min epsilon')
parm.add_argument('--epsilon_decay',type=float,default=0.95,help='decay')
parm.add_argument('--dataset',type=int,default=1480,help='dataset')
parm.add_argument('--load_weight',type=str,default='',help='pretrain weight')
parm.add_argument('--sample',type=int,default=100,help='number o2 sample')
parm.add_argument('--o2ono1',type=int,default=1,help='number o2 sample')
args = parm.parse_args()