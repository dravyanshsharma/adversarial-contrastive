
import numpy as np

## filter training points within tau of a differently labeled point


taus = [1.8]
#taus = [0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0]

def load_data(type):
    trainx = np.load('features/'+type+'con_feature_train.npy')
    trainy = np.load('features/'+type+'con_label_train.npy')
    return trainx,trainy

def distance(a,b):
    return np.linalg.norm(a-b)

type = 'sup'
trainx,trainy = load_data(type)

print(len(trainx))
print(np.size(trainy))

## Uncomment the following to find the nearest neighbors
#
#nearests = np.array([])
#num_iters=len(trainx)
#for i in range(num_iters):
#    if i % 500 == 0:
#        print(i)
#    x = trainx[i]
#    mind = float("inf")
#    mindx = -1
#    # find nearest point of a different label
#    # we store this too for different thresholds algo
#    for j in range(len(trainx)):
#        if trainy[i] == trainy[j]:
#            continue
#        d = distance(x,trainx[j])
#        if d < mind:
#            mind = d
#            mindx = j
#    nearests=np.append(nearests,mind)
    
## Loads pre-computed nearest neighbors
nearests = np.load('features/nearests.npy')


## Uncomment to pre-process features
#for tau in taus:
#    print('tau='+str(tau))
#    to_delete = []
#    for i in range(np.size(nearests)):
#        if nearests[i]<tau:
#            to_delete.append(i)
#    trainx = np.delete(trainx,to_delete,0)
#    nearests = np.delete(nearests,to_delete)
#    print(len(trainx))
#    np.save('features/'+type+'con_feature_train_'+str(tau)+'.npy', trainx)


## Pre-processes labels
for tau in taus:
    print('tau='+str(tau))
    to_delete = []
    for i in range(np.size(nearests)):
        if nearests[i]<tau:
            to_delete.append(i)
    trainy = np.delete(trainy,to_delete,0)
    nearests = np.delete(nearests,to_delete)
    print(len(trainy))
    np.save('features/'+type+'con_label_train_'+str(tau)+'.npy', trainy)
        