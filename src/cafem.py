from MLFE import *
from utils import *
from args import args

opt_type = args.opt_type
opt_size = 9 if opt_type == 'o1' else 5
qsa_size = args.qsa_size
history_size = args.history_size
input_size = qsa_size * 2 + opt_size * history_size + 1 * history_size + opt_size * 4 + 4
buffer_size = args.buffer_size
seed = args.seed
depth = args.depth
budget = args.budget
gamestep = args.gamestep
num_epochs = args.num_epochs
num_local_epochs = args.num_local_epochs
num_episodes = args.num_episodes
optimisation_steps = args.num_optimize_steps
n_jobs = args.n_jobs
tau = args.tau
gamma = args.gamma
epsilon = args.epsilon
batch_size = args.batch_size

num_process=args.multiprocessing
multiprocessing=True if args.multiprocessing>0 else False

save_model = True
# train = True
# test = True
out_dir = os.path.join(args.out_dir, 'cafem')
model_dir = os.path.join(args.out_dir, 'cafem_model')
if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
class Tasks():
    def __init__(self,datasetids,buffer_size=100):
        self.datasets = {}
        self.tasks = []
        if os.path.isfile('../data/log.csv'):
            log = pd.read_csv('../data/log.csv',header=None)
            for val in log.values:
                self.tasks.append([val[0],val[1],Buffer(buffer_size)])
        else:
            f = open('../data/log.csv','a')
            for did in tqdm(datasetids):
                f_dataset = "../data/%d/%d.arff" % (did, did)
                dataset, meta,tasktype = load(f_path=f_dataset)
                self.datasets[did] = (dataset,meta)
                for i,v in enumerate(meta[:-1]):
                    if v == "numeric":
                        f.write("%d,%d\n" % (did,i))
                        self.tasks.append([did,i,Buffer(buffer_size)])
            f.close()            
    def sample(self,n):
        tasks = random.sample(self.tasks,n)
        return tasks


def generate_trajectories(task):
    #env = envfunc(task[0],feature=task[1])
    #print('loading')
    
    f_dataset = "../data/%d/%d.arff" % (task[0], task[0])
    weights = task[2]
    dataset, meta,tasktype = load(f_path=f_dataset) 
    #print('loaded')
    env = Env(dataset, feature=task[1], maxdepth=depth, evalcount=budget,
              opt_type=opt_type, \
              random_state=seed, n_jobs=1)
    tmp_buffer = []
    #print('env done')
    localmodel = Model(opt_size=opt_size, input_size=input_size, name="model",maml=False)
    #print('model done')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    localsess = tf.Session(config=config)
    
    localsess = tf.Session()
    for key in localmodel.weights:
        localsess.run(localmodel.weights[key].assign(weights[key]))
        
    saver = tf.train.Saver()


    for n in range(num_episodes):
        print(n)
        total_reward = 0
        env.reset()
        for j in range(gamestep):
            s = np.copy(env.state)
            act_mask = np.copy(env.action_mask)
            Q = localsess.run(localmodel.Q_, feed_dict={localmodel.inputs: [s]})[0]

            action = ma.masked_array(Q, mask=act_mask).argmax()

            if np.random.rand(1) < epsilon:
                action = ma.masked_array(np.random.rand(opt_size), mask=act_mask).argmax()
            #print(action)
            s_next, reward = env.step(action)
            total_reward += reward
            tmp_buffer.append([s, action, reward, s_next,act_mask])
            if env.stop:
                break
    del env
    return tmp_buffer


def main():


    modelNetwork = Model(opt_size=opt_size, input_size=input_size, name="model")
    targetNetwork = Model(opt_size=opt_size, input_size=input_size, name="target")
    #model.summ_op = tf.summary.merge_all()

    tasks = Tasks(TRAINSETID,buffer_size)
    #tasks = Tasks([795,1067],buffer_size)
    saver = tf.train.Saver()


    sess = tf.Session()
    sess.run(modelNetwork.init_op)
    sess.run(targetNetwork.init_op)
    saver.save(sess,os.path.join(model_dir, "model_0.ckpt"))
    trainables = tf.trainable_variables()
    updateOps = updateTargetGraph(trainables, tau)
    
    for g in tqdm(range(num_epochs), total = num_epochs):

        sample_tasks = tasks.sample(args.meta_batch_size)
        weights = {}
        for key in modelNetwork.weights:
            weights[key] = modelNetwork.weights[key].eval(sess)
            #print(weights[key])
            

        if multiprocessing:
            pool = Pool(num_process)
            
            all_tmp_buffer = np.array(pool.map(generate_trajectories,[(val[0],1,weights) for val in sample_tasks]))
            pool.close()
            pool.join()
            for i,tmp_buffer in enumerate( all_tmp_buffer):
                for val in tmp_buffer:
                    sample_tasks[i][2].add(val)
        else:
            for task in sample_tasks:
                #tmp_buffer = generate_trajectories((tasks.datasets[task[0]][0],task[1]))
                tmp_buffer = generate_trajectories((task[0],task[1],weights))
                
                for val in tmp_buffer:
                    task[2].add(val)

        inputsa,labela,inputsb,labelb,actiona,actionb=[],[],[],[],[],[]


        for task in sample_tasks:
            task_buff = task[2]
            task_sample = task_buff.sample(args.update_batch_size*2)
            s, a, r, s_next,act_mask = [np.squeeze(elem, axis=1) for elem in np.split(task_sample, 5, 1)]
            s = np.array([ss for ss in s])
            s = np.reshape(s, (args.update_batch_size*2, input_size))
            s_next = np.array([ss for ss in s_next])
            s_next = np.reshape(s_next, (args.update_batch_size*2, input_size))
            act_mask = np.array([am for am in act_mask])
            act_mask = np.reshape(act_mask,(args.update_batch_size*2,opt_size))

            Q1 = sess.run(modelNetwork.Q_, feed_dict={modelNetwork.inputs: s_next})
            Q2 = sess.run(targetNetwork.Q_, feed_dict={targetNetwork.inputs: s_next})
            
            #doubleQ = Q2[:, np.argmax(ma.masked_array(Q1, mask=act_mask), axis=-1)]
            doubleQ =  np.array([Q2[i][ss] for i,ss in \
                                 enumerate(np.argmax(ma.masked_array(Q1, mask=act_mask), axis=-1))])
            Q_target = np.clip(r + gamma * doubleQ, -1. / (1 - gamma), 0)
            inputsa.append(s[:args.update_batch_size])
            labela.append(Q_target[:args.update_batch_size])
            inputsb.append(s[args.update_batch_size:])
            labelb.append(Q_target[args.update_batch_size:])
            actiona.append(a[:args.update_batch_size])
            actionb.append(a[args.update_batch_size:])
        feed_dict = {modelNetwork.inputsa: inputsa, \
                     modelNetwork.inputsb: inputsb, \
                     modelNetwork.Q_nexta: labela, \
                     modelNetwork.Q_nextb: labelb, \
                     modelNetwork.actiona: actiona,\
                     modelNetwork.actionb: actionb
                     }

        result = sess.run([modelNetwork.metatrain_op,modelNetwork.total_loss1,modelNetwork.total_losses2[-1]],\
                          feed_dict)
        print(result)
        #print(modelNetwork.total_loss1)
        #print(modelNetwork.total_losses2)
        updateTarget(updateOps, sess)
        try:
            f = open(os.path.join(out_dir,'loss.csv'),'a')
            f.write("%.8f,%.8f\n" % (result[1],result[2]))
            f.close()
            #if  (g+1) % SAVE_INTERVAL == 0:
            path = saver.save(sess, os.path.join(model_dir, "model_%d.ckpt" % (g+1) ))
            path = saver.save(sess, os.path.join(out_dir, "model_%d.ckpt" % (g + 1)))
            print(path)
        except:
            pass
    #saver.save(sess,os.path.join(model_dir, "model.ckpt" ))


if __name__ == "__main__":
    main()
# python cafem.py --multiprocessing 6 --out_dir ../out/ml --num_episodes 5