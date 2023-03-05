

from random import gauss
import numpy as np


n2 = 512
taus = [0.75,1.25,1.75,2.5,3.5,5.0]
tau1s = [1.0,1.5,2.0,2.5,3.0]



## load feature vectors (test and training)
## type = self or sup
def load_data(type,tau):
    taustr = ''
    if tau:
        taustr = '_'+str(tau)
    trainx = np.load('features/'+type+'con_feature_train'+taustr+'.npy')
    trainy = np.load('features/'+type+'con_label_train'+taustr+'.npy')
    testx = np.load('features/'+type+'con_feature_test.npy')
    testy = np.load('features/'+type+'con_label_test.npy')
    return trainx, trainy, testx, testy

## uniformly random vector in R^dims
def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]


def rotation_matrix(a,b):
    norm = np.matmul(np.transpose(a+b),a+b)
    return 2*(np.matmul(a+b,np.transpose(a+b)))/norm - np.identity(len(a))

def intersect_intervals(a, b):
    if a[0]<b[0]:
        if a[1]<b[0]:
            return None
        elif b[1]<a[1]:
            return b
        else:
            return (b[0],a[1])
    else:
        if b[1]<a[0]:
            return None
        elif a[1]<b[1]:
            return a
        else:
            return (a[0],b[1])
        
## distance of point x from line vector d through the origin
def distance(d, x):
    return np.linalg.norm(np.dot(x,d)*d-x)
        
for tau1 in tau1s:
    for typ in ['sup','self']:
        print('tau1='+str(tau1)+', type='+typ)

        trainx, trainy, testx, testy = load_data(typ,tau1)
        for tau in taus:
            print('tau='+str(tau))
            
        
            # one random adversary for each test point
            num_iters = 100#len(testx)
            fails = 0
            for itr in range(num_iters):
                x = testx[itr]
                y = testy[itr]
                
                adv = np.array(make_rand_vector(n2))
                # center points at given test point
                trainx_fixed = [tx - x for tx in trainx]
                
                train_same = []
                train_diff = []
                for i in range(len(trainx_fixed)):
                    if trainy[i] == y:
                        train_same.append(trainx_fixed[i])
                    else:
                        train_diff.append((trainx_fixed[i], i))
                
                train_same_filtered = []
                for j in range(len(train_same)):
                    dist = distance(adv, train_same[j])
                    # for all training points of the same label as x
                    # filter training points to those within tau of line
                    if dist <= tau:
                        train_same_filtered.append(train_same[j])
            
                fc = 0
                adversarial_point = np.zeros(n2)
                adversarial_point_found = False
                for i in range(len(train_diff)):
                    # for all training points of a different label
                    # filter training points to those within tau of line
                    dist = distance(adv, train_diff[i][0])
                    if dist > tau:
                        fc +=1
                        continue
                    
                    found = False
                    nearest = np.dot(train_diff[i][0],adv)*adv
                    # first check the closest point to x'
                    # if closer than all from train_same_filtered, then output
                    for j in range(len(train_same_filtered)):
                        d = np.linalg.norm(train_same_filtered[j]-nearest)
                        if d < dist:
                            found = True
                            break
                         
                    if not found:
                        adversarial_point = nearest
                        adversarial_point_found = True
                        break
                
                    # for each point with same label as x, determine interval of x-axis where 
                    # x' is closer to axis. Intersect all these intervals. Check if end points
                    # are within tau.
                    # we use that the point p equidistant from x and x' satisfies that point
                    # joining midpoint of xx' to p is perpendicular to xx'.
                    interval = (-float("inf"), float("inf"))
                    for j in range(len(train_same_filtered)):
                        diff = train_same_filtered[j]-train_diff[i][0]
                        dp = np.dot(diff,adv)
                        # if diff is perpendicular to adv axis (highly unlikely) we can ignore x
                        if dp == 0:
                            continue
                        mid = 0.5*(train_same_filtered[j]+train_diff[i][0])
                        point = np.dot(mid,diff)/dp
                        # interval opens to right if point+delta is closer to x'
                        pd = (point+0.1)*adv
                        dist_x = np.linalg.norm(pd-train_same_filtered[j])
                        dist_x1 = np.linalg.norm(pd-train_diff[i][0])
                        if (dist_x1<dist_x):
                            interval = intersect_intervals(interval, (point, float("inf")))
                        else:
                            interval = intersect_intervals(interval, (float("inf"), point))
                        if not interval:
                            break
                    if interval:
                        adversarial_point[0] = (interval[0]+interval[1])/2
                        adversarial_point_found = True
                        break
                if adversarial_point_found:
                    fails+=1
            
            print('acc='+str((num_iters-fails)*100/num_iters)+'%')
            
