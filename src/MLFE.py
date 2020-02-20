
import tensorflow as tf
import numpy as np
import random
import os
import math
from multiprocessing import Pool, cpu_count, Process
import multiprocessing

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.linear_model import LogisticRegression,Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, f1_score, log_loss, roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score
from scipy.io.arff import loadarff
from scipy import stats
import numpy.ma as ma
from sklearn.metrics import make_scorer

import os
from args import args

from utils import *
def one_mse_func():
    def one_relative_abs(y_true,y_pred):
        mae = mean_absolute_error(y_true,y_pred)
        one_mae = 1 - mae/np.mean(np.abs(y_true - np.mean(y_true)))
        #print(one_mae,np.abs(one_mae))
        return np.abs(one_mae)
        
    scorefunc = make_scorer(one_relative_abs, greater_is_better=False)
    return scorefunc

class Evaluater(object):
    """docstring for Evaluater"""
    def __init__(self, cv=5, stratified=True, n_jobs=1, tasktype="C", evaluatertype="rf", n_estimators=20,
                 random_state=np.random.randint(100000)):
        # tasktype = "C" or "R" for classification or regression
        # evaluatertype = 'rf', 'svm', 'lr' for random forest, SVM, logisticregression
        self.random_state = random_state
        self.cv = cv
        self.stratified = stratified
        self.n_jobs = n_jobs
        self.tasktype = tasktype
        if self.tasktype == "C":
            self.kf = StratifiedKFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
        else:
            self.kf = KFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)

        if evaluatertype == 'rf':
            if tasktype == "C":
                self.clf = RandomForestClassifier(n_estimators=n_estimators,
                                                  random_state=self.random_state)
            elif tasktype == "R":
                self.clf = RandomForestRegressor(n_estimators=n_estimators,
                                                 random_state=self.random_state)
        elif evaluatertype == "lr":
            if tasktype == "C":
                self.clf = LogisticRegression(solver='liblinear',random_state = self.random_state)
            elif tasktype =="R":
                self.clf = Lasso(random_state = self.random_state)
        #print(evaluatertype)
    # @profile
    def CV(self, X, y):
        X = np.nan_to_num(X)
        X = np.clip(X,-3e38,3e38)
        scoring = 'f1' if self.tasktype == "C" else one_mse_func()
        score = cross_val_score(self.clf,X,y,scoring=scoring,cv = self.kf,n_jobs=self.n_jobs)
        
  
        #print(X,y)
        #print(X.shape)
        #print("cv score",score,score.mean())
        return abs(score.mean())

    def CV2(self, X, y):
        res = []
        feature_importance = []

        # Parallel(n_jobs=1)(delayed()() )

        for train_index, test_index in self.kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.clf.fit(X_train, y_train)
            y_test_hat = self.clf.predict(X_test)
            #feature_importance.append(self.clf.feature_importances_)
            res.append(self.metrics(y_test, y_test_hat))

        #self.feature_importance = np.array(feature_importance).mean(axis=0)
        return np.array(res).mean(axis=0)

    def metrics(self, y_true, y_pred):
        if self.tasktype == "C":
            f_score = f1_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            logloss = log_loss(y_true, y_pred)
            return f_score, auc, logloss
        else:
            rel_MAE = 1 - mean_absolute_error(y_true, y_pred) / np.mean(np.abs(y_true - np.mean(y_true)))
            rel_MSE = 1 - mean_squared_error(y_true, y_pred) / np.mean(np.square((y_true - np.mean(y_true))))
            # print(mean_absolute_error(y_true,y_pred))

            return rel_MAE, rel_MSE

def load(f_path):
    le = LabelEncoder()
    tasktype=''
    if f_path[-4:] =='arff':

        dataset,meta = loadarff(f_path)
        dataset = np.array(dataset.tolist())

        meta_names = meta.names()
        meta_types = meta.types()
        if meta_types[-1] == "nominal":
            tasktype = "C"
        else:
            tasktype = "R"
    for i,val in enumerate(meta_types):
        if val == "nominal":

            target =  le.fit_transform(dataset[:,i]).astype(int)
            dataset[:, i] = target
    dataset = dataset.astype(float)
    return dataset,meta_types,tasktype

# Bit flipping environment
class Env():
    def __init__(self, dataset,feature,globalreward=True,maxdepth=5,evalcount=10,binsize=100,opt_type='o1',tasktype="C",evaluatertype='rf',\
                 random_state=np.random.randint(100000),historysize=5,pretransform=None,n_jobs=1):
        if  opt_type=='o2':
            maxdepth = 1
        #print(feature)
        self.opt_type = opt_type
        self.historysize = historysize
        self.maxdepth=maxdepth
        self.globalreward = globalreward
        self.action= ['fs','square','tanh','round','log','sqrt','mmn','sigmoid','zscore'] \
            if opt_type == 'o1' else ['fs','sum','diff','product','divide']
        self.one_hot = np.array([0] * len(self.action))
        self.action_size = len(self.action)
        self.tasktype=tasktype
        self.evaluatertype=evaluatertype
        self.evalcount=evalcount
        self.random_state = random_state

        self.origin_dataset = dataset
        self.origin_feat = feature

        self._pretrf_mapper = [i for i in range(self.origin_dataset.shape[1])]
        if pretransform is not None:
            #print(pretransform)

            for act in pretransform:
                #print(self.origin_dataset.shape)
                print(act)
                feat_id = act[0]
                actions = act[1].split("_")
                self.fe(actions,feat_id)
                #print(act)
            #print(self.origin_dataset)
            #print(self.origin_dataset.shape)
        if self.opt_type == 'o1':
            self.origin_feat = self._pretrf_mapper[self.origin_feat]
        elif self.opt_type == 'o2':
            value = []
            for val in self.origin_feat:
                value.append(self._pretrf_mapper[val])
            self.origin_feat = value

        self.evaluater = Evaluater(random_state=random_state,tasktype=tasktype,evaluatertype=evaluatertype,n_jobs=n_jobs)
        #print(self.origin_dataset[:,:-1].shape)
        #print(self.origin_dataset[0])

        self._init_pfm =self.evaluater.CV(self.origin_dataset[:,:-1],self.origin_dataset[:,-1])
        self.init_pfm = self._init_pfm
        self.y = np.copy(self.origin_dataset[:,-1])
        self.binsize = binsize
        self._init()
        #print("init performance",self._init_pfm)

    def _init(self):
        self.dataset = np.copy(self.origin_dataset)
        #print(self.origin_feat)
        self.feature = np.copy(self.dataset[:,self.origin_feat])
        self.transform = [0]*((len(self.action) + 1) * self.historysize)
        self.now_pfm = self._init_pfm
        self.stop=False
        qsa_rep = self._QSA()
        #[qsa_rep, np.copy(self.transform), \
        # np.copy(act_node_visit), np.copy(self.action_count), \
        # np.array(gain_each), np.array(action_count_root), \
        # np.array([gain_last, gain_last_last, budget, depth])
        self.state = np.concatenate([qsa_rep,np.copy(self.transform),\
                                     [0]*len(self.action),[0]*len(self.action),\
                                     [0]*len(self.action),[0]*len(self.action),\
                                     [0,0,1,0]],axis=None)

        #self.perform = [self._init_pfm]
        self.tg = {0:{'p':self._init_pfm,'d':0}}
        self.nodeid = 0
        self.countnode = 1
        self.action_mask = np.array([0]*len(self.action))
        self.best_seq = []
        self.action_count = [0]*len(self.action)
        self.action_gain = [0.0]*len(self.action)
        self.node_visit = [0]*self.evalcount
        self.node_visit[0] = 1
        self.current_f = self.origin_feat
        self.tg[self.nodeid]['fid'] = self.current_f
    def node2root(self,adict,node):
        current_node = node
        apath = [node]
        #print(node)
        while 'father' in adict[current_node]:
            current_node = adict[current_node]['father']
            apath.append(current_node)
        return [apath[i] for i in range(len(apath)-1,-1,-1)][1:]

    def step(self, action):

        operator = self.action[action]

        if self.stop:
            return
        if operator == 'stop':
            self.stop = True
        elif operator == 'fs':

            if "father" in self.tg[self.nodeid]:
                self.nodeid = self.tg[self.nodeid]['father']
                performance = self.tg[self.nodeid]['p']
                self.current_f = self.tg[self.nodeid]['fid']

            else:

                self.tg[self.nodeid]['father'] = -1
                self.nodeid = -1
                performance = self.evaluater.CV(np.delete(self.origin_dataset,self.origin_feat,axis=1)[:,:-1], self.y)
                self.tg[self.nodeid] = {'p':performance,'d':1}
                self.stop=True

            reward = performance - self.now_pfm
            self.now_pfm = performance

        else:

            # feature was generated alreadly
            if operator in self.tg[self.nodeid]:
                newnode = self.tg[self.nodeid][operator]
                performance = self.tg[newnode]['p']
                self.nodeid = newnode
                self.current_f = self.tg[self.nodeid]['fid']
                reward = performance - self.now_pfm
                self.now_pfm = performance
            else:
                newfeature = feature = self.dataset[:,self.current_f]
                if self.opt_type == 'o1':
                    if operator in set(['square', 'tanh', 'round']):
                        newfeature = getattr(np, operator)(feature)
                    elif operator == "log":
                        vmin = feature.min()
                        newfeature = np.log(feature - vmin + 1) if vmin < 1 else np.log(feature)

                    elif operator == "sqrt":
                        vmin = feature.min()
                        newfeature = np.sqrt(feature - vmin) if vmin <0 else np.sqrt(feature)

                    elif operator == "mmn":
                        mmn = MinMaxScaler()
                        newfeature = mmn.fit_transform(feature[:,np.newaxis]).flatten()

                    elif operator == "sigmoid":
                        newfeature = (1 + getattr(np, 'tanh')(feature / 2)) / 2

                    elif operator == 'zscore':
                        if np.var(feature) != 0:

                            newfeature = stats.zscore(feature)
                elif self.opt_type == 'o2':
                    if operator == "sum":
                        newfeature =feature.sum(axis=1)
                    elif operator == "diff":
                        newfeature = feature[:,0] * feature[:,1]
                    elif operator =="product":
                        newfeature = feature[:, 0] * feature[:, 1]
                    elif operator == 'divide':
                        over = feature[:,1]
                        while (np.any(over == 0)):
                            over = over + 1e-5
                        newfeature = feature[:, 0] / over


                if newfeature is not None:

                    newfeature = np.nan_to_num(newfeature)
                    newfeature = np.clip(newfeature, -math.sqrt(3.4e38), math.sqrt(3.4e38))
                    self.dataset = np.insert(self.dataset,self.dataset.shape[1]-1,newfeature,axis=1)
                    self.current_f = self.dataset.shape[1]-2
                else: #TODO
                    pass


                #X = np.concatenate([np.delete(self.dataset[:, :self.count_feat], self.origin_feat, 1), \
                #                    self.dataset[:, -2][:, np.newaxis]], axis=1)

                apath = self.node2root(self.tg,self.nodeid)

                #X = np.concatenate([self.origin_dataset[:,:-1],\
                #                    self.dataset[:,[self.tg[v]['fid'] for v in apath]+[self.current_f]]],axis=1)
                X = np.concatenate([self.origin_dataset[:, :-1], \
                                   self.dataset[:, [self.current_f]]], axis=1)
                #X = np.concatenate([np.delete(self.origin_dataset[:, :-1],self.origin_feat,axis=1), \
                #                    self.dataset[:, [self.current_f]]], axis=1)

                performance = self.evaluater.CV(X, self.y)

                self.tg[self.nodeid][operator] = self.countnode
                newnode = self.countnode
                self.tg[newnode] = {'p': performance, 'd': self.tg[self.nodeid]['d'] + 1,'fid':self.current_f}
                self.tg[newnode]['father'] = self.nodeid
                self.countnode += 1
                self.nodeid = newnode
                if self.countnode >= self.evalcount:
                    self.stop = True
                reward = performance - self.now_pfm
                self.now_pfm = performance

        if self.stop:
            reward = 0


        if self.tg[self.nodeid]['d'] >= self.maxdepth:
            self.action_mask = [0] + [1] * (len(self.action) - 1)
        else:
            self.action_mask = [0] * len(self.action)

        if self.dataset.shape[1] <= 2:
            self.action_mask[0] = 1
        if self.countnode >= self.evalcount:
            self.action_mask = [1] * (len(self.action))

        # history seq
        onehot = np.copy(self.one_hot)
        onehot[action] = 1
        self.transform.extend(onehot)
        self.transform.append(reward)
        self.transform = self.transform[-self.historysize*(len(self.action)+1):]
        # ExQSA
        qsa_rep = self._QSA()
        #action node visit
        self.node_visit[self.nodeid]+=1
        act_node_visit = [0]*len(self.action)
        for key in self.tg[self.nodeid]:
            if key in self.action:
                act_node_visit[self.action.index(key)] = self.node_visit[self.tg[self.nodeid][key]]

        # count of action
        self.action_count[action] += 1
        if self.node_visit[self.nodeid] == 1:
            self.action_gain[action] += reward
        # gain each action
        gain_each = [0 if self.action_count[i]==0 else self.action_gain[i]/self.action_count[i] for i in range(len(self.action))]

        # count action from root
        action_count_root = [0]*len(self.action)
        startnode = self.nodeid
        while "father" in self.tg[startnode] and startnode > 0:
            lastnode = self.tg[startnode]['father']
            for key in self.tg[lastnode]:
                if key in self.action and key != 'fid' and self.tg[lastnode][key] == startnode:
                    self.best_seq.insert(0,key)
                    action_count_root[self.action.index(key)] += 1
                    break
            startnode = lastnode
        if self.nodeid == -1:
            action_count_root[1] = 1
        # gain last and last last
        gain_last = reward
        gain_lastlast = 0
        if 'father' in self.tg[self.nodeid]:
            if 'father' in self.tg[self.tg[self.nodeid]['father']]:
                gain_lastlast = self.tg[self.nodeid]['p'] - self.tg[self.tg[self.tg[self.nodeid]['father']]['father']]['p']

        # budget
        budget = 1- self.countnode*1.0 / self.evalcount
        #depth
        depth = self.tg[self.nodeid]['d']
        depth = abs(depth)



        self.state = np.concatenate([qsa_rep,np.copy(self.transform),\
                                     np.copy(act_node_visit),np.copy(self.action_count),\
                                     np.array(gain_each),np.array(action_count_root),\
                                     np.array([gain_last,gain_lastlast,budget,depth])],axis=None)



        allperf = np.array([self.tg[i]['p'] for i in range(self.countnode)])
        startnode = allperf.argmax()
        self.best_pfm = allperf.max()

        #print('best-----------',self.best_pfm)
        if self.globalreward:
            reward = allperf.max() - self._init_pfm

        self.best_seq = []
        while "father" in self.tg[startnode] and startnode >0:
            lastnode = self.tg[startnode]['father']
            for key in self.tg[lastnode] :
                if key != 'fid' and self.tg[lastnode][key] == startnode:
                    self.best_seq.insert(0,key)
                    break
            startnode = lastnode
        #print(self.best_seq, np.array([self.tg[i]['p'] for i in range(self.countnode) ]).max())

        return self.state,reward

    def _QSA(self):
        if self.opt_type == 'o1':
            if self.tasktype == "C":
                feat_0 = self.feature[self.y == 0]
                feat_1 = self.feature[self.y == 1]
            elif self.tasktype == 'R':
                median = np.median(self.y)
                feat_0 = self.feature[self.y <  median]
                feat_1 = self.feature[self.y >= median]

            minval,maxval = feat_0.min(),feat_0.max()
            if abs(maxval - minval) < 1e-8:
                QSA0 = [0] * self.binsize
            else:
                bins = np.arange(minval,maxval,(maxval-minval) * 1.0 / self.binsize)[1:self.binsize]
                QSA0 = np.bincount(np.digitize(feat_0,bins)).astype(float) / len(feat_0)

            minval,maxval = feat_1.min(),feat_1.max()
            if abs(maxval - minval) < 1e-8:
                QSA1 = [0] * self.binsize
            else:
                bins = np.arange(minval,maxval,(maxval-minval) * 1.0 / self.binsize)[1:self.binsize]
                QSA1 = np.bincount(np.digitize(feat_1,bins)).astype(float) / len(feat_1)
            QSA = np.concatenate([QSA0,QSA1])
        elif self.opt_type == 'o2':
            QSA =   []
            #print(self.feature)
            for i in range(2):
                if self.tasktype == "C":
                    feat_0 = self.feature[:,i][self.y == 0]
                    feat_1 = self.feature[:,i][self.y == 1]
                elif self.tasktype == 'R':
                    median = np.median(self.y)
                    feat_0 = self.feature[:,i][self.y < median]
                    feat_1 = self.feature[:,i][self.y >= median]

                minval, maxval = feat_0.min(), feat_0.max()
                if abs(maxval - minval) < 1e-8:
                    QSA0 = [0] * self.binsize
                else:
                    bins = np.arange(minval, maxval, (maxval - minval) * 1.0 / self.binsize)[1:self.binsize]
                    QSA0 = np.bincount(np.digitize(feat_0, bins)).astype(float) / len(feat_0)

                minval, maxval = feat_1.min(), feat_1.max()
                if abs(maxval - minval) < 1e-8:
                    QSA1 = [0] * self.binsize
                else:
                    bins = np.arange(minval, maxval, (maxval - minval) * 1.0 / self.binsize)[1:self.binsize]
                    QSA1 = np.bincount(np.digitize(feat_1, bins)).astype(float) / len(feat_1)
                QSA.append(QSA0)
                QSA.append(QSA1)
            QSA = np.concatenate(QSA)
        return QSA

    def reset(self):
        self.dataset = self.origin_dataset
        self.feature = self.origin_feat
        self._init()
    def fe(self,operators,feat_id):
        #target = self.dataset[:,-1]
        #self.dataset = pd.DataFrame(np.copy(self.dataset[:, :-1]))
        #print(operators)
        #print('fe',feat_id)
        if  type(feat_id) is int:
            new_feat_id = self._pretrf_mapper[feat_id]

            if new_feat_id != -1:
                feature = self.origin_dataset[:,new_feat_id]
            else:
                feature = None
        else:
            new_feat_id_a = self._pretrf_mapper[feat_id[0]]
            new_feat_id_b = self._pretrf_mapper[feat_id[1]]
            new_feat_id = [new_feat_id_a,new_feat_id_b]
            if new_feat_id_a != -1 and new_feat_id_b != -1:
                feature = self.origin_dataset[:,new_feat_id]
            else:
                feature = None

        for operator in operators:
            #print(operator)
            if type(feat_id) is int:
                if operator in set(['square', 'tanh', 'round']):
                    feature = getattr(np, operator)(feature)
                elif operator == "log":
                    vmin = feature.min()
                    feature = np.log(feature - vmin + 1) if vmin < 1 else np.log(feature)

                elif operator == "sqrt":
                    vmin = feature.min()
                    feature = np.sqrt(feature - vmin) if vmin <0 else np.sqrt(feature)

                elif operator == "mmn":
                    mmn = MinMaxScaler()
                    feature = mmn.fit_transform(feature[:,np.newaxis]).flatten()

                elif operator == "sigmoid":
                    feature = (1 + getattr(np, 'tanh')(feature / 2)) / 2

                elif operator == 'zscore':
                    if np.var(feature) != 0:
                        feature = stats.zscore(feature)
                    else:
                        feature = None
                else:
                    feature =None
            else:
                if operator == "sum":
                    feature =feature.sum(axis=1)
                elif operator == "diff":
                    feature = feature[:,0] * feature[:,1]
                elif operator =="product":
                    feature = feature[:, 0] * feature[:, 1]
                elif operator == 'divide':
                    over = feature[:,1]
                    while (np.any(over == 0)):
                        over = over + 1e-5
                    feature = feature[:, 0] / over
                else:
                    feature = None
        #print('fff',operator,'cc')
        #print(feature)
        #print(self.origin_dataset)
        if len(operators) > 0 and feature is not None and operators[0] != 'fs':
            feature = np.nan_to_num(feature)
            feature = np.clip(feature, -math.sqrt(3.4e38), math.sqrt(3.4e38))
            self.origin_dataset = np.insert(self.origin_dataset,-1,feature,axis=1)
        if len(operators) > 0 and operators[0] == "fs":
            self.origin_dataset = np.delete(self.origin_dataset,new_feat_id,axis=1)
            if type(feat_id) is int:
                self._pretrf_mapper[feat_id] = -1
                for i in range(feat_id, len(self._pretrf_mapper)):
                    if self.pretrf_mapper[i] >=1:
                        self._pretrf_mapper[i] -= 1

            else:
                for feat in feat_id:
                    self._pretrf_mapper[feat] = -1
                for i in range(min(feat_id),len(self._pretrf_mapper)):
                    if self.pretrf_mapper[i] >=1:
                        self._pretrf_mapper[i] -= 1
                for i in range(max(feat_id), len(self._pretrf_mapper)):
                    if self.pretrf_mapper[i] >= 1:
                        self._pretrf_mapper[i] -= 1

        return feature




# Experience replay buffer
class Buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[max(1,int(0.0001 * self.buffer_size)):]

    def sample(self,size):
        if len(self.buffer) >= size:
            experience_buffer = self.buffer
        else:
            experience_buffer = self.buffer * size
        return np.copy(np.reshape(np.array(random.sample(experience_buffer,size)),[size,5]))

# Simple feed forward neural network

class Model():
    def __init__(self, opt_size, input_size, name, meta=False,update_lr=1e-3,meta_lr=0.001,num_updates=1,maml=True,qsasize=200):
        self.input_size = input_size
        self.opt_size = self.dim_output =  opt_size
        self.dim_hidden = [128,128,64]
        self.skip=1
        self.qsasize = qsasize
        self.inputs = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)
        self.Q_next = tf.placeholder(shape=None, dtype=tf.float32)
        self.action = tf.placeholder(shape=None, dtype=tf.int32)
        #print(self.input_size)
        #print(self.qsasize)
        self.inputsa = tf.placeholder(shape=[None,None, self.input_size], dtype=tf.float32)
        self.inputsb = tf.placeholder(shape=[None,None, self.input_size], dtype=tf.float32)
        self.Q_nexta = tf.placeholder(shape=[None,None], dtype=tf.float32)
        self.Q_nextb = tf.placeholder(shape=[None,None], dtype=tf.float32)
        self.actiona = tf.placeholder(shape=[None,None], dtype=tf.int32)
        self.actionb = tf.placeholder(shape=[None,None], dtype=tf.int32)
        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.num_updates = num_updates
        self.size = opt_size
        self.input_size = input_size
        self.loss_func = self.mse
        self.weights = self.construct_fc_weights()
        self.network()
        if maml:
            self.construct_model()
        self.init_op = tf.global_variables_initializer()

    def mse(self, y_pred, y_true):
        return tf.reduce_sum(tf.square(y_pred - y_true))

    def construct_fc_weights(self):
        factor = 1
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.input_size, self.dim_hidden[0]], \
                                                        stddev=math.sqrt(  factor/((self.input_size+self.dim_hidden[0])/2)  )))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1, len(self.dim_hidden)):
            weights['w' + str(i + 1)] = tf.Variable(
                tf.truncated_normal([self.dim_hidden[i - 1], self.dim_hidden[i]], \
                                    stddev=math.sqrt( factor/((self.dim_hidden[i - 1]+ self.dim_hidden[i])/2 ) )))
            weights['b' + str(i + 1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))

        if self.skip == 1:
            weights['skip' +  str(len(self.dim_hidden) + 1 -1)] = tf.Variable(
                tf.truncated_normal([self.input_size - self.qsasize, self.dim_hidden[-1]], \
                                    stddev=math.sqrt(factor / ((self.input_size - self.qsasize + self.dim_hidden[-1])))))
        elif self.skip == 0:
            weights['skip' + str(len(self.dim_hidden) + 1)] = tf.Variable(
                tf.truncated_normal([self.input_size - self.qsasize, self.dim_output], \
                                    stddev=math.sqrt(factor / ((self.input_size - self.qsasize + self.dim_output)))))

        else:
            pass
        weights['w' + str(len(self.dim_hidden) + 1)] = tf.Variable(
            tf.truncated_normal([self.dim_hidden[-1], self.dim_output], \
                                stddev=math.sqrt( factor/((self.dim_hidden[-1]+ self.dim_output)/2 ) )))
        #weights['b' + str(len(self.dim_hidden) + 1)] = tf.Variable(tf.zeros([self.dim_output]))

        return weights

    def forward(self, inp,  weights, reuse=False):
        hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse,
                           scope='0',norm="None")

        if self.skip == 1:

            for i in range(1, len(self.dim_hidden)-1):
                hidden = normalize(tf.matmul(hidden, weights['w' + str(i + 1)]) + weights['b' + str(i + 1)],
                                   activation=tf.nn.relu, reuse=reuse, scope=str(i + 1), norm='None')

            hiddenp1 = tf.matmul(hidden, weights['w' + str(i + 1+1)]) + weights['b' + str(i +1+ 1)]

            hiddenp2 = tf.matmul(inp[:, self.qsasize:], weights['skip' + str(len(self.dim_hidden) + 1 - 1)])
            hidden = normalize(hiddenp1+hiddenp2, \
                               reuse=reuse, activation=tf.nn.relu, scope='skip' + str(i + 1), norm='None')
            Q_ = tf.matmul(hidden, weights['w' + str(len(self.dim_hidden) + 1)])



        elif self.skip == 0:
            for i in range(1, len(self.dim_hidden)):
                hidden = normalize(tf.matmul(hidden, weights['w' + str(i + 1)]) + weights['b' + str(i + 1)],
                                   activation=tf.nn.relu, reuse=reuse, scope=str(i + 1), norm='None')

            Q_ = tf.matmul(hidden, weights['w' + str(len(self.dim_hidden) + 1)]) + \
                 tf.matmul(inp[:,self.qsasize:],weights['skip'+ str(len(self.dim_hidden) + 1)])
        else:
            for i in range(1, len(self.dim_hidden)):
                hidden = normalize(tf.matmul(hidden, weights['w' + str(i + 1)]) + weights['b' + str(i + 1)],
                                   activation=tf.nn.relu, reuse=reuse, scope=str(i + 1), norm='None')

            Q_ = tf.matmul(hidden, weights['w' + str(len(self.dim_hidden) + 1)])

        return Q_
    def L2loss(self,weights,reg):
         loss_reg = 0.0
         for key in weights:
             loss_reg += reg*tf.reduce_sum(tf.square(weights[key]))
         return loss_reg

    def network(self):

        self.Q_ = self.forward(self.inputs, self.weights)
        self.action_onehot = tf.one_hot(self.action, self.size, dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.Q_, self.action_onehot), axis=1)
        self.loss = self.loss_func(self.Q_next, self.Q) +self.L2loss(self.weights,1e-5)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.meta_lr)
        self.train_op = self.optimizer.minimize(self.loss)
        #TODO optimizer can be only one

    def construct_model(self):

        with tf.variable_scope('mamlmodel', reuse=None) as training_scope:

            #if 'weight' in dir(self):
            training_scope.reuse_variables()
            weights = self.weights
            #else:
            #    self.weights = weights = self.construct_fc_weights()
            def forward_Q(output,action):
                action_onehot = tf.one_hot(action, self.size, dtype=tf.float32)
                Q = tf.reduce_sum(tf.multiply(output, action_onehot), axis=1)
                return Q

            def task_metalearn(inp, reuse=True):
                inputa, inputb, labela, labelb, actiona, actionb = inp
                task_outputbs, task_lossesb = [], []
                task_outputa = forward_Q(self.forward(inputa, weights, reuse=reuse),actiona)  # only reuse on the first iter


                task_lossa = self.loss_func(task_outputa, labela)
                grads = tf.gradients(task_lossa, list(weights.values()))

                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(
                    zip(weights.keys(), [weights[key] - self.update_lr * gradients[key] for key in weights.keys()]))
                output = forward_Q(self.forward(inputb, fast_weights, reuse=True),actionb)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))

                for j in range(self.num_updates - 1):
                    loss = self.loss_func(forward_Q(self.forward(inputa, fast_weights, reuse=True),actiona), labela)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(),
                                            [fast_weights[key] - self.update_lr * gradients[key] for key in
                                             fast_weights.keys()]))
                    output = forward_Q(self.forward(inputb, fast_weights, reuse=True),actionb)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

                return task_output

            out_dtype = [tf.float32, [tf.float32] * self.num_updates, tf.float32, [tf.float32] * self.num_updates]
            result = tf.map_fn(task_metalearn, elems=(self.inputsa, self.inputsb, self.Q_nexta, self.Q_nextb,\
                                                      self.actiona,self.actionb),\
                               dtype=out_dtype, parallel_iterations=args.meta_batch_size)
            outputas, outputbs, lossesa, lossesb = result

            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(args.meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(args.meta_batch_size) for
                                                  j in range(self.num_updates)]
            # after the map_fn
            self.outputas, self.outputbs = outputas, outputbs
            #TODO same optimizer problem
            #optimizer = tf.train.AdamOptimizer(self.meta_lr)
            optimizer = self.optimizer
            self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[self.num_updates - 1])
            self.metatrain_op = optimizer.apply_gradients(gvs)
            # tf.summary.scalar( 'Pre-update loss', total_loss1)
            # for j in range(self.num_updates):
            #     tf.summary.scalar( 'Post-update loss, step ' + str(j + 1), total_losses2[j])

        #
        # def network(name,inputs,opt_size,input_size):
        #     with tf.variable_scope(name):
        #         # self.input_size
        #         self.hidden1 = tf.layers.dense(inputs, 128, activation=tf.nn.relu, name="fc1")
        #         self.hidden2 = tf.layers.dense(self.hidden1, 128, activation=tf.nn.relu, name="fc2")
        #         self.hidden3 = tf.layers.dense(self.hidden2, 64, activation=tf.nn.relu, name='fc3')
        #         self.Q_ = tf.layers.dense(self.hidden3, self.size, activation=None, use_bias=False, name='Q')
        #         # self.predict = tf.argmax(tf.argmax(self.Q_,axis=-1)
        #         self.action = tf.placeholder(shape=None, dtype=tf.int32)
        #         self.action_onehot = tf.one_hot(self.action, self.size, dtype=tf.float32)
        #         self.Q = tf.reduce_sum(tf.multiply(self.Q_, self.action_onehot), axis=1)
        #
        #
        #         self.Q_next = tf.placeholder(shape=None, dtype=tf.float32)
        #         self.loss = tf.reduce_sum(tf.square(self.Q_next - self.Q))
        #         self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        #         self.train_op = self.optimizer.minimize(self.loss)
        #         self.init_op = tf.global_variables_initializer()



def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * (1. - tau)) + (tau * tfVars[idx + total_vars // 2].value())))
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)



# def main():
#
#     use_multiprocessing = True
#     num_process = 4
#     if use_multiprocessing:
#         pool = Pool(num_process)
#         np.array(pool.map(traineach, [(1,i) for i in range(21)]))
#         pool.close()
#         pool.join()
#     else:
#         for val in [(1,i) for i in range(21)]:
#             traineach(val)


# def main2():
#     HER = True
#     shaped_reward = False
#     size = 15
#     num_epochs = 5
#     num_cycles = 50
#     num_episodes = 16
#     optimisation_steps = 40
#     K = 4
#     buffer_size = 1e6
#     tau = 0.95
#     gamma = 0.98
#     epsilon = 0.0
#     batch_size = 128
#     add_final = False
#
#     total_rewards = []
#     total_loss = []
#     success_rate = []
#     succeed = 0
#
#     save_model = True
#     model_dir = "./train"
#     train = True
#
#     if not os.path.isdir(model_dir):
#         os.mkdir(model_dir)
#
#     modelNetwork = Model(size = size, name = "model")
#     targetNetwork = Model(size = size, name = "target")
#     trainables = tf.trainable_variables()
#     updateOps = updateTargetGraph(trainables, tau)
#     env = Env(size = size, shaped_reward = shaped_reward)
#     buff = Buffer(buffer_size)
#
#     if train:
#         plt.ion()
#         fig = plt.figure()
#         ax = fig.add_subplot(211)
#         plt.title("Success Rate")
#         ax.set_ylim([0,1.])
#         ax2 = fig.add_subplot(212)
#         plt.title("Q Loss")
#         line = ax.plot(np.zeros(1), np.zeros(1), 'b-')[0]
#         line2 = ax2.plot(np.zeros(1), np.zeros(1), 'b-')[0]
#         fig.canvas.draw()
#         with tf.Session() as sess:
#             sess.run(modelNetwork.init_op)
#             sess.run(targetNetwork.init_op)
#             for i in tqdm(range(num_epochs), total = num_epochs):
#                 for j in range(num_cycles):
#                     total_reward = 0.0
#                     successes = []
#                     for n in range(num_episodes):
#                         env.reset()
#                         episode_experience = []
#                         episode_succeeded = False
#                         for t in range(size):
#                             s = np.copy(env.state)
#                             g = np.copy(env.target)
#                             inputs = np.concatenate([s,g],axis = -1)
#                             action = sess.run(modelNetwork.predict,feed_dict = {modelNetwork.inputs:[inputs]})
#                             action = action[0]
#                             if np.random.rand(1) < epsilon:
#                                 action = np.random.randint(size)
#                             s_next, reward = env.step(action)
#                             episode_experience.append((s,action,reward,s_next,g))
#                             total_reward += reward
#                             if reward == 0:
#                                 if episode_succeeded:
#                                     continue
#                                 else:
#                                     episode_succeeded = True
#                                     succeed += 1
#                         successes.append(episode_succeeded)
#                         for t in range(size):
#                             s, a, r, s_n, g = episode_experience[t]
#                             inputs = np.concatenate([s,g],axis = -1)
#                             new_inputs = np.concatenate([s_n,g],axis = -1)
#                             buff.add(np.reshape(np.array([inputs,a,r,new_inputs]),[1,4]))
#                             if HER:
#                                 for k in range(K):
#                                     future = np.random.randint(t, size)
#                                     _, _, _, g_n, _ = episode_experience[future]
#                                     inputs = np.concatenate([s,g_n],axis = -1)
#                                     new_inputs = np.concatenate([s_n, g_n],axis = -1)
#                                     final = np.sum(np.array(s_n) == np.array(g_n)) == size
#                                     if shaped_reward:
#                                         r_n = 0 if final else -np.sum(np.square(np.array(s_n) == np.array(g_n)))
#                                     else:
#                                         r_n = 0 if final else -1
#                                     buff.add(np.reshape(np.array([inputs,a,r_n,new_inputs]),[1,4]))
#
#                     mean_loss = []
#                     for k in range(optimisation_steps):
#                         experience = buff.sample(batch_size)
#                         s, a, r, s_next = [np.squeeze(elem, axis = 1) for elem in np.split(experience, 4, 1)]
#                         s = np.array([ss for ss in s])
#                         s = np.reshape(s, (batch_size, size * 2))
#                         s_next = np.array([ss for ss in s_next])
#                         s_next = np.reshape(s_next, (batch_size, size * 2))
#                         Q1 = sess.run(modelNetwork.Q_, feed_dict = {modelNetwork.inputs: s_next})
#                         Q2 = sess.run(targetNetwork.Q_, feed_dict = {targetNetwork.inputs: s_next})
#                         doubleQ = Q2[:, np.argmax(Q1, axis = -1)]
#                         Q_target = np.clip(r + gamma * doubleQ,  -1. / (1 - gamma), 0)
#                         _, loss = sess.run([modelNetwork.train_op, modelNetwork.loss], feed_dict = {modelNetwork.inputs: s, modelNetwork.Q_next: Q_target, modelNetwork.action: a})
#                         mean_loss.append(loss)
#
#                     success_rate.append(np.mean(successes))
#                     total_loss.append(np.mean(mean_loss))
#                     updateTarget(updateOps,sess)
#                     total_rewards.append(total_reward)
#                     ax.relim()
#                     ax.autoscale_view()
#                     ax2.relim()
#                     ax2.autoscale_view()
#                     line.set_data(np.arange(len(success_rate)), np.array(success_rate))
#                     # line.set_data(np.arange(len(total_rewards)), np.array(total_rewards))
#                     line2.set_data(np.arange(len(total_loss)), np.array(total_loss))
#                     fig.canvas.draw()
#                     fig.canvas.flush_events()
#                     plt.pause(1e-7)
#             if save_model:
#                 saver = tf.train.Saver()
#                 saver.save(sess, os.path.join(model_dir, "model.ckpt"))
#         print("Number of episodes succeeded: {}".format(succeed))
#         raw_input("Press enter...")
#     with tf.Session() as sess:
#         saver = tf.train.Saver()
#         saver.restore(sess, os.path.join(model_dir, "model.ckpt"))
#         while True:
#             env.reset()
#             print("Initial State:\t{}".format(env.state))
#             print("Goal:\t{}".format(env.target))
#             for t in range(size):
#                 s = np.copy(env.state)
#                 g = np.copy(env.target)
#                 inputs = np.concatenate([s,g],axis = -1)
#                 action = sess.run(targetNetwork.predict,feed_dict = {targetNetwork.inputs:[inputs]})
#                 action = action[0]
#                 s_next, reward = env.step(action)
#                 print("State at step {}: {}".format(t, env.state))
#                 if reward == 0:
#                     print("Success!")
#                     break
#             raw_input("Press enter...")
import pandas as pd
def performance(did,maxfeat,step):
    d_path = "../data/%d/%d.arff" % (did,did)
    dataset,meta = load(d_path)

    env = Env(dataset, random_state=666,opt_type='o1',feature=0)

    f_path = "../out/ddqn_single/%d_%d.csv"
    records = {}
    for i in range(maxfeat):
        try:
            record = pd.read_csv(f_path % (did,i),header=None).fillna('').values
        except:
            pass
        records[i] = record
    trfs = []
    for i in range(step):
        actions = []
        for key in records:

            action = records[key][i][1].split("_")
            actions.append((key,action))
        trfs.append(actions)
    #trfs = [trfs[i] for i in range(len(trfs)-1,-1,-1)]
    env.batch_perform(trfs)


if __name__ == "__main__":
    #traineach((1,0))
    main()
#     act = [0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,2,1,1,3,3,3,0,3,3,3,3,0,0
# ,0,0,0,0,0,0,0,0,2,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,0,0,0,0
# ,0,3,3,3,3,3,2,1,1,1,1,2,2,2,2,2,3,3,3,3,3,1,0,0,0,0,2,2,2,2,2]
#
#
#     originreward = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.033271092797494184, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0,
#      0.01609692793235651,
#      0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0.0044764209194717575, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, -0.023262244252483855, 0.0,
#      0.0, 0.0
#         , 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02774759379366304, 0.0, 0.0, 0.0, 0.0, -0.01347452835261076, 0.0, 0.0,
#      0.0, 0.0, 0, 0, 0, 0, 0, -0.01908813158420708, 0, 0, 0, 0, 0.03522684408349441, 0.0, 0.0, 0.0, 0.0, 0.007427125036177817, 0.0,
#      0.0, 0.0, 0.0, -0.027512723068076173, 0.027521080864861214, -0.05218683031829169, 0.036759477202605306, -0.01776815616807259,
#      0.0037391272566771327, 0.0, 0, 0, 0, -0.004505403385712203, 0.0, 0.0, 0.0, 0.0]
#      #print(performance(1,20,10))
#
#     d_path = "../data/%d/%d.arff" % (1,1)
#     dataset,meta = load(d_path)
#     env = Env(dataset, random_state=666,opt_type='o1',feature=0)
#     target = dataset[:,-1]
#     dataset = dataset[:,:-1]
#     dataset = pd.DataFrame(dataset)
#     initreward = 0.4039313
#     for i,actions in enumerate( np.array(act).reshape((int(len(act)/5),5))):
#         #dataset = pd.DataFrame(np.copy(dataset))
#
#         copy = np.copy(dataset[i])
#         featname = ''
#         print('--------',i)
#         print(actions)
#         print(np.array(originreward).reshape((int(len(act)/5),5))[i])
#         for action in actions:
#             featname += str(action)
#             if action == 0:
#                 break
#             elif action == 1:
#                 copy = np.squeeze(np.sqrt(abs(copy)))
#             elif action == 2:
#                 # print('mmn')
#                 scaler = MinMaxScaler()
#                 copy = np.squeeze(scaler.fit_transform(np.reshape(copy, [-1, 1])))
#             elif action == 3:
#                 # print('log')
#                 while (np.any(copy == 0)):
#                     copy = copy + 1e-5
#                 copy = np.squeeze(np.log(abs(np.array(copy))))
#         if actions[0] != 0:
#             dataset.insert(0,'new_%d_%s' % (i,featname), copy)
#
#     print(dataset.columns)
#
#     evaluater = Evaluater(random_state=666, \
#                                        tasktype='C', evaluatertype='rf')
#
#
#     score = evaluater.CV(dataset.values, target)
#     print(score)
#     reward = score - initreward
#     initreward = score
#     print(reward)