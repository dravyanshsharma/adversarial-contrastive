import matplotlib.pyplot as plt
import numpy as np


taus = [3.0]
tau1s = [0.0,2.7,3.0]
## load feature vectors (test and training)
## type = self or sup
def load_data(type,tau):
    taustr = ''
    if tau and tau > 0.0:
        taustr = '_'+str(tau)
    trainx = np.load('features/'+type+'con_feature_train'+taustr+'.npy')
    trainy = np.load('features/'+type+'con_label_train'+taustr+'.npy')
    testx = np.load('features/'+type+'con_feature_test.npy')
    testy = np.load('features/'+type+'con_label_test.npy')
    return trainx, trainy, testx, testy

def distance(a,b):
    return np.linalg.norm(a-b)

## compute fraction of test points not within tau for given tau
for tau1 in tau1s:
    for typ in ['self','sup']:
        print('tau1='+str(tau1)+', type='+typ)
    
        trainx, trainy, testx, testy = load_data(typ,tau1)
        for tau in taus:
            print('tau='+str(tau))
            error = 0
            num_iters=10000
            notdk = 0
            for i in range(num_iters):
                x = testx[i]
                y = testy[i]
                found = False
                mind = float("inf")
                mindx = -1
                for j in range(len(trainx)):
                    x1 = trainx[j]
                    d = distance(x,x1)
                    if d<mind:
                        mind = d
                        mindx = j
                if mind<tau:
                    notdk += 1
                    if trainy[mindx]!=y:
                        error += 1
            dk = (num_iters-notdk)*100.0/num_iters
            # print('dk='+str(dk)+'%')
            natacc = (num_iters-error)*100.0/num_iters
            # print('natacc='+str(natacc)+'%')
