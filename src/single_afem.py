from MLFE import *
from utils import *
from args import args
import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"#args.cuda
                       
                        
def simulate(inp):

    env,g,epsilon,opt_size,input_size,gamestep,load,localsess,localmodelNetwork,model_dir,fid = inp

    if load:
        localmodelNetwork = Model(opt_size=opt_size, input_size=input_size, name="model",maml=False)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        localsess = tf.Session(config=config)
        saver = tf.train.Saver()
        #print('model_%d_%d.ckpt' % (fid,g))
        saver.restore(localsess,os.path.join(model_dir,'model_%d_%d.ckpt' % (fid,g)))


    tmp_buffer = []
    #for n in range(num_episodes):
    total_reward = 0
    env.reset()
    for j in range(gamestep):
        #print(j)
        s = np.copy(env.state)
        act_mask = np.copy(env.action_mask)


        Q = localsess.run(localmodelNetwork.Q_,feed_dict = { localmodelNetwork.inputs:[s]})[0]
        action = ma.masked_array(Q,mask=act_mask).argmax()

        if np.random.rand(1) < epsilon:
            action = ma.masked_array(np.random.rand(opt_size),mask = act_mask).argmax()

        s_next, reward = env.step(action)
        total_reward += reward

        tmp_buffer.append(np.copy([s,action,reward,s_next,act_mask]))
        if env.stop:
            break
    #if env.best_pfm > best_pfm:
    #    best_pfm = env.best_pfm
    #    best_seq = env.best_seq
        #print(best_seq, best_pfm)
    #print('done')
    if load:
        localsess.close()
    return tmp_buffer,np.copy(env.best_seq),np.copy(env.best_pfm)
    #return 1,1,1

def main():
    opt_type= args.opt_type
    opt_size = 9 if opt_type =='o1' else 5
    qsa_size = args.qsa_size
    history_size = args.history_size
    input_size = qsa_size*2 + opt_size*history_size + 1*history_size + opt_size*4 + 4
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
    multiprocessing = args.multiprocessing

    save_model = True
    train = True
    test = True
    out_dir = os.path.join(args.out_dir,'safem')
    model_dir = os.path.join(args.out_dir,'safem_model')
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    did = args.dataset
    f_dataset = "../data/%d/%d.arff" % (did,did)
    dataset, meta, tasktype = load(f_path=f_dataset)


    n_feats = dataset.shape[1] - 1
    modelNetwork = Model(opt_size=opt_size, input_size=input_size, name="model",maml=False)
    targetNetwork = Model(opt_size=opt_size, input_size=input_size, name="target",maml=False)

    globalbuff = [None] * n_feats
    bst = []
    if train:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            init_pfm = 0
            test_pfm = []
            for g in tqdm(range(num_epochs), total = num_epochs):
                pretransform =[]
                globalbuff = [None] * n_feats
                perf = 0

                for fid in tqdm(range(dataset.shape[1]-1),total = dataset.shape[1]-1):

                    if globalbuff[fid] is None:
                        globalbuff[fid] = Buffer(buffer_size)
                    buff = globalbuff[fid]

                    saver = tf.train.Saver()
                    if len(args.load_weight)> 0 and g==0:
                        print("loading weight")
                        saver = tf.train.Saver()
                        saver.restore(sess, args.load_weight)
                        if multiprocessing >0:
                            saver.save(sess, os.path.join(model_dir, 'model_%d.ckpt' % (fid)))
                    elif g ==0:
                        sess.run(modelNetwork.init_op)
                        sess.run(targetNetwork.init_op)
                        if multiprocessing>0:
                            saver.save(sess,os.path.join(model_dir,'model_%d.ckpt' % (fid)))
                    else:
                        saver.restore(sess,os.path.join(model_dir, "model_%d.ckpt" % (fid)))

                    print(tasktype)
                    if multiprocessing > 0:
                        n_jobs = 1
                    env = Env(dataset, feature=fid,maxdepth=depth,evalcount=budget,opt_type=opt_type,random_state=seed,\
                              tasktype=tasktype,pretransform=pretransform,n_jobs = n_jobs,evaluatertype=args.evaluatertype)
                    if g == 0 and fid == 0:
                        print(fid,"init perform",env.init_pfm)
                        f = open(os.path.join(out_dir, "test_succeed.csv"), 'a')
                        f.write("%d,%.6f\n" % (g-1, env.init_pfm))
                        f.close()

                    if init_pfm is None:
                        init_pfm == env.init_pfm
                    #print(env.dataset.shape)



                    trainables = tf.trainable_variables()
                    updateOps = updateTargetGraph(trainables, tau)
                    best_seq = []
                    best_pfm = []
                    best_tst = 0
                    for i in tqdm(range(num_local_epochs), total = num_local_epochs):


                        if multiprocessing>0:
                            pool = Pool(multiprocessing)


                            result = pool.map(simulate, [[env,g-1,epsilon,opt_size,input_size,gamestep,True,None,None,\
                                                          model_dir,fid] \
                                                         for n in range(num_episodes)])

                            pool.close()
                            pool.join()

                            for ep in result:
                                tmp_buffer, localbest_seq, localbest_pfm = ep
                                for val in tmp_buffer:
                                    buff.add(val)
                                best_seq.append(localbest_seq)
                                best_pfm.append(localbest_pfm)

                        # for n in range(num_episodes):
                        else:
                            for n in range(num_episodes):
                                #print(n)
                                tmp_buffer,localbest_seq,localbest_pfm = \
                                    simulate((env, g, epsilon, opt_size, input_size, gamestep,False, sess,\
                                              modelNetwork,model_dir,fid))
                                for val in tmp_buffer:
                                    buff.add(val)
                                best_seq.append(localbest_seq)
                                best_pfm.append(localbest_pfm)

                            #print(total_reward)
                        mean_loss=[]
                        for k in range(optimisation_steps):
                            experience = buff.sample(batch_size)
                            #print(experience.shape)
                            s, a, r, s_next,act_mask = [np.squeeze(elem, axis=1) for elem in np.split(experience, 5, 1)]
                            s = np.array([ss for ss in s])
                            s = np.reshape(s, (batch_size, input_size))
                            #print(s_next.shape)
                            s_next = np.array([ss for ss in s_next])
                            #print(s_next.shape)
                            s_next = np.reshape(s_next, (batch_size, input_size))
                            act_mask = np.array([am for am in act_mask])
                            act_mask = np.reshape(act_mask, (batch_size, opt_size))

                            Q1 = sess.run(modelNetwork.Q_, feed_dict={modelNetwork.inputs: s_next})
                            Q2 = sess.run(targetNetwork.Q_, feed_dict={targetNetwork.inputs: s_next})
                            #doubleQ = Q2[:, np.argmax(ma.masked_array(Q1, mask=act_mask), axis=-1)]
                            doubleQ = np.array([Q2[i][ss] for i, ss in
                                       enumerate(np.argmax(ma.masked_array(Q1, mask=act_mask), axis=-1))])

                            Q_target = np.clip(r + gamma * doubleQ, -1. / (1 - gamma), 0)
                            _, loss = sess.run([modelNetwork.train_op, modelNetwork.loss],
                                               feed_dict={modelNetwork.inputs: s, modelNetwork.Q_next: Q_target,
                                                          modelNetwork.action: a})
                            print(loss)
                            mean_loss.append(loss)
                        f = open(os.path.join(out_dir,"loss_%d.csv" % fid),'a')
                        f.write("%.6f\n" % (sum(mean_loss) / len(mean_loss)))
                        f.close()

                        updateTarget(updateOps,sess)


                    saver = tf.train.Saver()
                    saver.save(sess, os.path.join(model_dir, "model_%d.ckpt" % (fid)))
                    round_best_pfm = np.around(np.array(best_pfm),decimals=4)
                    bestindex = ma.masked_array([len(val) for val in best_seq],\
                                                ~(round_best_pfm==round_best_pfm.argmax())).argmin()
                    seq = best_seq[bestindex]
                    f = open(os.path.join(out_dir,'%d_%d_%d.csv' % (g, did, fid)), 'a')
                    f.write('%.6f,%s\n' % (max(best_pfm), '_'.join(seq)))
                    f.close()
                    pretransform.append((fid, '_'.join(seq)))
                f = open(os.path.join(out_dir, "succeed.csv"), 'a')
                f.write("%.6f\n" % max(best_pfm))
                f.close()
                
                testdecay = 1
                if (g+1) % args.eval_interval == 0:

                    pretransform_test = []
                    for fid in tqdm(range(dataset.shape[1] - 1), total=dataset.shape[1] - 1):
                        env = Env(dataset, feature=fid, maxdepth=depth, evalcount=budget, opt_type=opt_type,tasktype=tasktype,
                                  random_state=seed, pretransform=pretransform_test, n_jobs=n_jobs,evaluatertype=args.evaluatertype)
                        saver = tf.train.Saver()
                        saver.restore(sess, os.path.join(model_dir, "model_%d.ckpt" % (fid)))

                        for j in range(gamestep):
                            s = np.copy(env.state)
                            act_mask = np.copy(env.action_mask)
                            Q = sess.run(modelNetwork.Q_, feed_dict={modelNetwork.inputs: [s]})
                            action = ma.masked_array(Q, mask=act_mask).argmax()
                            s_next, reward = env.step(action)
                            if env.stop:
                                break
                        pretransform_test.append((fid, '_'.join(env.best_seq)))

                    f = open(os.path.join(out_dir,"test_succeed_feat.csv"),'a')
                    for val in pretransform_test:
                        f.write("%d,%s\n" % (val[0],val[1]))
                    f.close()
                    test_pfm.append(env.best_pfm)
                    print("test pfm %.6f" % env.best_pfm)
                    f = open(os.path.join(out_dir,"test_succeed.csv"),'a')
                    f.write("%d,%.6f\n" % (g,env.best_pfm))
                    f.close()
                    #testdecay = init_pfm / (sum(test_pfm[-2:]) / len(test_pfm[-2:]))
                epsilon = max(args.min_epsilon,epsilon*args.epsilon_decay * testdecay)
                #epsilon = max(args.min_epsilon,epsilon*args.epsilon_decay*(init_pfm/test_pfm[]))

if __name__ == "__main__":
    main()

#    python3 single_afem.py --load_weight ../out/ml/model/model_5.ckpt --dataset
#python single_afem.py --load_weight ../out/ml/model/model_5.ckpt --dataset 931 --out_dir ../out/o2_5_931_r2
#python single_afem.py --load_weight ../out/ml/model/model_5.ckpt --dataset 1049 --out_dir ../out/o2_5_1049 --num_epochs 50 --buffer_size 1000 --num_episodes 1
#python single_afem.py --load_weight ../out/ml/model/model_5.ckpt --dataset 589 --out_dir ../out/o2_5_589 --num_epochs 50 --buffer_size 1000 --num_episodes 1
#python single_afem.py --load_weight ../out/ml/model/model_5.ckpt --dataset 917 --out_dir ../out/o2_5_917 --num_epochs 50 --buffer_size 1000 --num_episodes 1
#python single_afem.py --load_weight ../out/ml/model/model_5.ckpt --dataset 1480 --out_dir ../out/o2_5_1480 --num_epochs 50 --buffer_size 1000 --num_episodes 1
#python single_afem.py --load_weight ../out/ml/model/model_5.ckpt --dataset 4154 --out_dir ../out/o2_5_4154 --num_epochs 50 --buffer_size 1000 --num_episodes 1
#python single_afem.py --load_weight ../out/ml/model/model_5.ckpt --dataset 1050 --out_dir ../out/o2_5_1050 --num_epochs 50 --buffer_size 1000 --num_episodes 1
#python single_afem.py --load_weight ../out/ml/model/model_5.ckpt --dataset 806 --out_dir ../out/o2_5_806 --num_epochs 50 --buffer_size 1000 --num_episodes 1
#python single_afem.py --load_weight ../out/ml/model/model_5.ckpt --dataset 1566 --out_dir ../out/o2_5_1566 --num_epochs 50 --buffer_size 1000 --num_episodes 1
