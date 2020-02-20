
import numpy as np
import os
import random
import tensorflow as tf
import pandas as pd

TRAINSETID=[833, 1004, 1494, 971, 727, 847, 807, 1570, 734, 912,
          40701, 827, 1487, 717, 1056, 312, 1464, 979, 951, 978,
          1454, 251, 1069, 904, 871, 1020, 740, 40910, 728, 1479,
          841, 1444, 1547, 37, 1452, 1507, 40704, 40900, 949,
          1019, 849, 819, 1068, 845, 1471, 1496, 1462, 1453,
          751, 772, 881, 40705, 903, 853, 947, 1443, 40983, 1046,
          923, 950, 1510, 761, 843, 1120, 718, 1116, 40589, 970,
          151, 1042, 962, 737, 1016, 1485, 40994, 722, 723, 743,
          40666, 1063, 866, 958, 901, 799, 823, 1451, 846, 821,
          976, 1504, 977, 797, 837, 994, 715, 1040, 725, 803, 995, 948]
TESTSETID = [620,589,586,618,616,607,795,770,825,1480,913,735,
             1067,917,4154,1049,806,1566,954,1038,]


def normalize(inp, activation, reuse, scope,norm):
    return activation(inp)

    #if norm == 'batch_norm':
    #    return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    #elif norm == 'layer_norm':
    #    return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    #elif norm == 'None':
    #    if activation is not None:
    #        return activation(inp)
    #    else:
    #        return inp
def load_pretransform(fdir):
    pfm = pd.read_csv(os.path.join(fdir,'safem/test_succeed.csv'),header=None)
    transform = pd.read_csv(os.path.join(fdir,'safem/test_succeed_feat.csv'),header=None)
    transform = transform.fillna('')
    index = np.argmax(pfm[1].values)
    print(index)
    index = int(pfm.values[index,0])

    print(index,max(pfm[1].values))
    countfeatures = transform[0].max()+1
    bsttransform = transform.values[index*(countfeatures):(index+1)*countfeatures]
    print(bsttransform)
    return bsttransform
def get_result(mark,did,plot=True):
    f_path_te = '../out/%s%d/safem/test_succeed.csv'
    f_path_tr = '../out/%s%d/safem/succeed.csv'
    f_loss = '../out/%s%d/safem/loss_%s.csv'
    res = []

    for tid in did:
        try:
            score_te = pd.read_csv(f_path_te % (mark,tid),header=None)
            score_tr = pd.read_csv(f_path_tr % (mark,tid),header=None)
            score_te = score_te.drop_duplicates(0,keep='last')

            res.append([tid,score_tr[0].values.max(),score_tr[0].values.mean(),score_te[1].max(),score_te[0].max(),score_te.values[:,0][np.argmax(score_te.values[:,1])]])
            if plot:
                plt.plot(range(len(score_tr)),score_tr[0])
                plt.plot(score_te[0],score_te[1])
                plt.show()
        except:
            res.append([tid,np.nan,np.nan,np.nan,np.nan,np.nan])

            print("error",tid)
    res = pd.DataFrame(res,columns=['tid','bstrandom','random','train','round','bstround'])
    return res
def plot(fpath1,fpath2,size=30,name=''):
    import matplotlib.pyplot as plt
    fontsize=17
    safem = pd.read_csv(fpath1,header=None)
    cafem = pd.read_csv(fpath2,header=None)
    plt.figure(figsize=(5,3))
    plt.title("%s" % name,fontsize=fontsize)
    plt.xlabel('epoch')
    plt.ylabel('performance',fontsize=fontsize)
    plt.xticks((0, 5, 10, 15, 20))
    plt.plot(range(min(size,len(safem[1]))),safem[1].cummax().values[:min(size,len(safem[1]))],"r--",label = 'SAFEM')
    if len(cafem) < size:
        for i in range(len(cafem),size):
            cafem.loc[i,1]=0
    plt.plot(range(min(size,len(cafem[1]))),cafem[1].cummax().values[:min(size,len(cafem[1]))],label = 'CAFEM')
    plt.legend(loc='lower right',fontsize=fontsize)
    #plt.savefig('../out/%s.png' % name, dpi=255)
    plt.show()


def plot2():
    #plt.bar()
    fontsize=17

    import numpy as np
    import matplotlib.pyplot as plt

    men_means = (10.76, 10.06, 25.83, 0.19)
    women_means = (4.17, 3.62, 18.17, 0.03)

    ind = np.arange(len(men_means))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10,2.8))
    rects1 = ax.bar(ind - width / 2, men_means, width, color='SkyBlue', label='Random Forest')
    rects2 = ax.bar(ind + width / 2, women_means, width, color='Green', label='Logistic Regression')
    plt.yticks((0,5,10,15,20,25,30))
    for x,y in zip(ind-width/2,men_means):
        plt.text(x,y-0.05,'%.1f' % y +'%',ha='center',va='bottom',fontsize=fontsize)
    for x,y in zip(ind+width/2,women_means):
        plt.text(x,y-0.05,'%.1f' % y+ '%',ha='center',va='bottom',fontsize=fontsize)



    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('% improvement\n from Baseline',fontsize=fontsize)
    ax.set_title('Performance of SAFEM with different learning algorithms on 20 datasets',fontsize=fontsize)
    plt.xticks(ind, ('Average', 'Median', 'Maximum', 'Minimum'),fontsize=fontsize)
    ax.legend(fontsize=fontsize,loc='upper left')

    plt.show()



if __name__=='__main__':
    # #load_pretransform('../out/tmp/')
    #plot('../doc/ijcai19/ml/589.csv', '../out/o2_5_589/safem/test_succeed.csv', size=21, name='openml_589')
    #plot('../doc/ijcai19/ml/806.csv', '../out/o2_5_806/safem/test_succeed.csv', size=21, name='fri_c3_1000_50')
     #plot('../doc/ijcai19/ml/931.csv','../out/o2_5_931/safem/test_succeed.csv',size=21,name='Disclosure_z')
     #plot('../doc/ijcai19/ml/917.csv','../out/o2_5_917/safem/test_succeed.csv',size=21,name='fri_c1_1000_25')
     #plot('../doc/ijcai19/ml/1480.csv', '../out/o2_5_1480/safem/test_succeed.csv', size=21, name='llpd')
     #plot('../doc/ijcai19/ml/1566.csv', '../out/o2_5_1566/safem/test_succeed.csv', size=21, name='Hill-valley')
     #plot('../doc/ijcai19/ml/4154.csv', '../out/o2_5_4154/safem/test_succeed.csv', size=21, name='Credit Card')
    #

    plot2()
