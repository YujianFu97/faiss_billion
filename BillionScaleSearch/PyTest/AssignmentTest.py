from sklearn.cluster import KMeans
import numpy as np
from numpy import random
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.colors as mcolors
import math

def fetchsearchcost(TargetK, assignment, trainindices, NeighborNum, clusterindices):

    trainset = set()
    for i in range(TargetK):
        NNLabel = assignment[trainindices[i]]
        for j in range(NeighborNum):
            if (clusterindices[j] not in trainset):
                trainset.add(clusterindices[j])
            if (NNLabel == clusterindices[j]):
                break;

    return trainset


nb = 400
nt = 400
D = 2
nc = 8
TargetK = 15
NeighborNum = 5

np.random.seed(10)
base = random.rand(nb, D)


train = base[:nt, :]


kmeans = KMeans(n_clusters=nc).fit(train)
clusternbrs = NearestNeighbors(n_neighbors = NeighborNum, algorithm = 'brute').fit(kmeans.cluster_centers_)
clusterdistances, clusterindices = clusternbrs.kneighbors(train)

assignment = clusterindices[[i for i in range(nt)], 0]
clustersize = []
for i in range(nc):
    clustersize.append(0)
for i in range(nt):
    clustersize[assignment[i]] += 1

#print(clustersize)

trainnbrs = NearestNeighbors(n_neighbors=TargetK + 1, algorithm = 'brute').fit(train)
traindistances, trainindices = trainnbrs.kneighbors(train)
traindistances = traindistances[:, 1:TargetK+1]
trainindices = trainindices[:, 1:TargetK+1]
#print(trainindices)

TrainBeNNs = []
for i in range(nt):
    IthNNs = []

    for j in range(nt):
        for k in range(TargetK):
            if (trainindices[j][k] == i):
                IthNNs.append(j)

                break;
    TrainBeNNs.append(IthNNs)

#print(TrainBeNNs)



# Get the search cost of each train vector

OriginSearchCost = []
for i in range(nt):
    OriginSearchCost.append(fetchsearchcost(TargetK, assignment, trainindices[i], NeighborNum, clusterindices[i]))

print(OriginSearchCost)
colorlist = [item for item in mcolors.CSS4_COLORS]
colorlist = ['aqua', 'black', 'blue', 'yellow', 'brown', 'darkcyan', 'darkgreen', 'darkmagenta', 'darkorchid', 'darkred', 'darkslategray', 'deeppink', 'fuchsia', 'indigo', 'lime', 'maroon', 'navy', 'orangered', 'darkseagreen',
 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet']

continueFlag = True

shifttimes = 0
for i in range(nt):
    for j in range(TargetK):
        NN = trainindices[i][j]
        NNLabel = assignment[NN]

        origincost = 0
        for k in range(len(TrainBeNNs[NN])):
            NNRNN = TrainBeNNs[NN][k]
            for m in (OriginSearchCost[NNRNN]):
                origincost += clustersize[m]
        # Shift the NN
        assignment[NN] = assignment[i]
        clustersize[NNLabel] -= 1
        clustersize[assignment[i]] += 1

        shiftcost = 0
        NewSearchCost = []
        for k in range(len(TrainBeNNs[NN])):
            NNRNN = TrainBeNNs[NN][k]
            NewSearchCost.append(fetchsearchcost(TargetK, assignment, trainindices[NNRNN], NeighborNum, clusterindices[NNRNN]))
            
            for m in (NewSearchCost[-1]):
                shiftcost += clustersize[m]
        
        if (shiftcost < origincost):
            shifttimes += 1
            for k in range(len(TrainBeNNs[NN])):
                NNRNN = TrainBeNNs[NN][k]
                OriginSearchCost[NNRNN] = NewSearchCost[k]
        else:
            assignment[NN] = NNLabel
            clustersize[NNLabel] += 1
            clustersize[assignment[i]] -= 1

originassignment = clusterindices[[i for i in range(nt)], 0]

conflictID = [i for i in range(nt) if originassignment[i] != assignment[i]]

outvectors = []
for i in range(nc):
    outvectors.append([])

for i in conflictID:
    outvectors[originassignment[i]].append(i)

baseclusterdistances, baseclusterindices = clusternbrs.kneighbors(base)
baseassignment  = baseclusterindices[[i for i in range(nb)], 0]

#print(baseassignment)

for i in range(nb):
    mindist = 1e9
    targetclusterID = baseassignment[i]
    if (len(outvectors[baseassignment[i]]) > 0):
        for NN in outvectors[baseassignment[i]]:
            vectordist = math.dist(train[NN], base[i])
            if (vectordist < mindist):
                baseassignment[i] = assignment[NN]
                mindist = vectordist

        for j in range(nt):
            if (assignment[j] == targetclusterID):
                vectordist = math.dist(base[i], train[j])
                if (vectordist < mindist):
                    baseassignment[i] = targetclusterID
                    break;

originbaseassignment = baseclusterindices[[i for i in range(nb)], 0]
baseconflictID = [i for i in range(nt) if originbaseassignment[i] != baseassignment[i]]

#print(assignment)
#print(originassignment)
#print(conflictID)
#print("The total number of conflict: ", len(conflictID))

plt.figure()
vor = Voronoi(kmeans.cluster_centers_)
voronoi_plot_2d(vor, line_colors = 'black', show_vertices = False, point_size = 15)
for i in range(nc):
    #plt.scatter(kmeans.cluster_centers_[i, 0], kmeans.cluster_centers_[i, 1], marker = "D", color = colorlist[2*i%len(colorlist)])
    #plt.text(kmeans.cluster_centers_[i][0], kmeans.cluster_centers_[i][1], str(np.round(kmeans.cluster_centers_[i], 2)))

    ConflictCluster = [id for id in (conflictID) if originassignment[id] == i]
    filter_label = train[ConflictCluster]
    #plt.scatter(filter_label[:, 0], filter_label[:, 1], marker = '*', color = colorlist[2*i%len(colorlist)])

    ConflictCluster = [id for id in conflictID if assignment[id] == i]
    filter_label = train[ConflictCluster]
    #plt.scatter(filter_label[:, 0], filter_label[:, 1], marker = 'X', color = colorlist[2*i%len(colorlist)])

    ConflictCluster = [id for id in baseconflictID if baseassignment[id] == i]
    filter_label = base[ConflictCluster]
    #plt.scatter(filter_label[:, 0], filter_label[:, 1], color = colorlist[2*i%len(colorlist)], marker = '+')

    filter_label = base[baseassignment == i]
    #plt.scatter(filter_label[:, 0], filter_label[:, 1], color = colorlist[2*i%len(colorlist)], s = 1, marker = ',')

    filter_label = train[assignment == i]
    #plt.scatter(filter_label[:, 0], filter_label[:, 1], color = colorlist[2*i%len(colorlist)], s = 10)

    filter_label = base[originassignment == i]
    plt.scatter(filter_label[:, 0], filter_label[:, 1], color = colorlist[2*i%len(colorlist)], s = 10)

ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.show()
exit()

for i in range(len(conflictID)):
    for j in range(len(TrainBeNNs[conflictID[i]])):
        RNN = TrainBeNNs[conflictID[i]][j]
        plt.plot([train[RNN][0], train[conflictID[i]][0]],[train[RNN][1], train[conflictID[i]][1]], color = 'black', linewidth = 0.2)

print("The shift times: ", shifttimes)



