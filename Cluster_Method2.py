import random
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import easygui
import cv2

path = easygui.fileopenbox()



def slope(x, y):
    num = (len(x) * sum(x * y) - sum(x) * sum(y))
    den = (len(x) * sum(x ** 2) - (sum(x)) ** 2)
    if den != 0:
        m =  num / den
        theta = math.degrees(math.atan(m))
    else:
        theta = 90

    if theta < 0:
        theta = 360 + theta
    if 90 < theta <= 180:
        theta = 180 - theta
    elif 180 < theta <= 270:
        theta = theta - 180
    elif 270 < theta <= 360:
        theta = 360 - theta

    return theta




def clustering_function(X, DBSCAN_eps, DBSCAN_minsample,  min_data_cluster, DBSCAN_eps1, DBSCAN_minsample1, min_data_cluster1):

    db = DBSCAN(eps=DBSCAN_eps, min_samples=DBSCAN_minsample).fit(X)  #150 400

    data = np.array(list(zip(X[:, 0], X[:, 1], db.labels_)))  # keep all  X,Y,Cluster
    data = data[data[:, 2] != -1]
    index = set(data[:, 2])

    data_cluster = []
    data_cluster_reduced = np.empty(shape=[0, 4])
    data_cluster_reduced_ = []
    len_cluster = []
    slope_cluster = []
    for i in index:
        dataAux = data[data[:, 2] == i]
        data_cluster.append(dataAux)
        print(i,len(dataAux))
        # len_cluster.append([i, len(dataAux)])
        if len(dataAux) > min_data_cluster:                                                                                         #min data cluster

            X = dataAux[:, 0:2]
            db = DBSCAN(eps=DBSCAN_eps1, min_samples=DBSCAN_minsample1).fit(X) #12 100
            labels = set(db.labels_)

            data_1 = list(zip(dataAux[:, 0], dataAux[:, 1], dataAux[:, 2], db.labels_))
            data_1 = np.array(data_1)

            for j in labels:
                if j != -1:
                    a = data_1[data_1[:, 3] == j]
                    print(i, j, len(a))
                    if len(a) > min_data_cluster1:                                                                                    #second min data cluster
                        len_cluster.append([i, j, len(a)])
                        slope_cluster.append([a[0, 2], a[0, 3],
                                              slope(a[:, 0], a[:, 1]),
                                              [np.mean(a[:, 0]), np.mean(a[:, 1])]])

                        data_cluster_reduced = np.concatenate((data_cluster_reduced, a),
                                                              axis=0)  # data_cluster_reduced.append(a)
                        data_cluster_reduced_.append(a)

    return data_cluster, data_cluster_reduced, slope_cluster, data_cluster_reduced_, len_cluster




def check_cluster(data_cluster_reduced, total_data, len_cluster, min_samples_final_clusters):

    if len(total_data) != 0:
        total_data = np.array(total_data)
        clusters_total_data = set(total_data[:, 0])
        len_cluster = np.array(len_cluster)
        total_data_cluster = np.empty(shape=[0, 4])
        for i in clusters_total_data:
            aux = np.concatenate((np.unique(total_data[total_data[:, 0] == i][:, 1]),
                                  np.unique(total_data[total_data[:, 0] == i][:, 2])))
            aux = np.unique(aux)

            acum = 0
            for j in aux:
                acum = acum + len_cluster[(len_cluster[:, 0] == i) & (len_cluster[:, 1] == j)][0, 2]
            for j in aux:
                len_cluster[np.where((len_cluster[:, 0] == i) & (len_cluster[:, 1] == j))] = [i, j, acum]

        clusters_data_reduced = np.unique(len_cluster[:, 0])
        for i in clusters_data_reduced:
            aux = np.unique(len_cluster[len_cluster[:, 0] == i][:, 1])
            for j in aux:
                if len_cluster[(len_cluster[:, 0] == i) & (len_cluster[:, 1] == j)][0, 2] > min_samples_final_clusters:
                    # print(i, j, len_cluster[(len_cluster[:, 0] == i) & (len_cluster[:, 1] == j)][0, 2])

                    total_data_cluster = np.concatenate((total_data_cluster, data_cluster_reduced[
                        (data_cluster_reduced[:, 2] == i) & (data_cluster_reduced[:, 3] == j)]),
                                                        axis=0)  # data_cluster_reduced.append(a)

    else:
        if len(len_cluster)>0:
            total_data_cluster = np.empty(shape=[0, 4])
            len_cluster = np.array(len_cluster)

            clusters_data_reduced = set(len_cluster[:, 0])

            for i in clusters_data_reduced:
                aux = np.unique(len_cluster[len_cluster[:, 0] == i][:, 1])
                for j in aux:
                    if len_cluster[(len_cluster[:, 0] == i) & (len_cluster[:, 1] == j)][
                        0, 2] > min_samples_final_clusters:
                        # print(i, j, len_cluster[(len_cluster[:, 0] == i) & (len_cluster[:, 1] == j)][0, 2])

                        total_data_cluster = np.concatenate((total_data_cluster, data_cluster_reduced[
                            (data_cluster_reduced[:, 2] == i) & (data_cluster_reduced[:, 3] == j)]),
                                                            axis=0)  # data_cluster_reduced.append(a)
        else:
            total_data_cluster = data_cluster_reduced


    return total_data_cluster






def completing_function(gray_image, DBSCAN_eps, DBSCAN_minsample, min_data_cluster, min_distance):
    data_cluster = []
    data_cluster_reduced = []
    total_data = []
    img = []
    data_angle = []
    stime = time.time()
    w, h = tuple(gray_image.shape)
    x = np.argwhere(gray_image > 0)

    x1 = np.zeros(2)
    y1 = np.zeros(2)
    if len(x) > 0:

        data_cluster, data_cluster_reduced, info_cluster, data_cluster_reduced_, len_cluster = clustering_function(x, DBSCAN_eps,
                                                                                                      DBSCAN_minsample,
                                                                                                      min_data_cluster, DBSCAN_eps1, DBSCAN_minsample1, min_data_cluster1)

        clusters = set(data_cluster_reduced[:, 2])

        for i in clusters:
            sub_clusters = np.unique(data_cluster_reduced[data_cluster_reduced[:, 2] == i][:, 3])

            data_sub_custer = data_cluster_reduced[data_cluster_reduced[:, 2] == i]

            for j in sub_clusters:
                data_angle.append([i, j, slope(data_sub_custer[data_sub_custer[:, 3] == j][:, 0],
                                               data_sub_custer[data_sub_custer[:, 3] == j][:, 1])])

            for j in range(len(sub_clusters)):
                data_c1 = data_sub_custer[data_sub_custer[:, 3] == sub_clusters[j]]
                for m in range(len(sub_clusters) - j - 1):
                    n = m + j + 1
                    data_c2 = data_sub_custer[data_sub_custer[:, 3] == sub_clusters[n]]
                    btree = cKDTree(data_c1[:, 0:2])  # two fist columns from dataAux
                    dist, idx = btree.query(data_c2[:, 0:2])



                    if dist.min() < min_distance:
                        dist_data = np.argwhere(dist == dist.min())
                        idx_data = idx[np.where(dist == dist.min())]
                        p1 = data_c1[idx_data[len(idx_data) - 1]]
                        slope_p1 = slope(data_c2[:, 0], data_c2[:, 1])
                        p2 = data_c2[int(dist_data[len(dist_data) - 1])]
                        slope_p2 = slope(data_c1[:, 0], data_c1[:, 1])
                        distance_p1_p2 = dist.min()
                        x1[0] = p1[0]
                        x1[1] = p2[0]
                        y1[0] = p1[1]
                        y1[1] = p2[1]
                        s = slope(x1, y1)
                        print(p1, p2, distance_p1_p2, slope_p1, slope_p2, slope(x1, y1))
                        #30 30
                        if (abs(s - slope_p2) < 30) or abs(s - slope_p1) < 30:


                            total_data.append([i, p1[3], p2[3], p1[:2], p2[:2]])

        total_data_cluster = check_cluster(data_cluster_reduced, total_data, len_cluster, min_samples_final_clusters)

    clusters = set(total_data_cluster[:, 3])
    completed_img = np.zeros((w, h, 3), np.uint8)
    color =(195, 242, 74)
    for i in clusters:
        data_x = total_data_cluster[total_data_cluster[:, 3] == i][:, 0]
        data_y = total_data_cluster[total_data_cluster[:, 3] == i][:, 1]
        for j in range(len(data_x)):
            cv2.circle(completed_img, (int(data_y[j]), int(data_x[j])), 10, color, 5)
    for i in total_data:
        cv2.line(completed_img, (int(i[4][1]), int(i[4][0])), (int(i[3][1]), int(i[3][0])), color, 50)

    return data_cluster, data_cluster_reduced, total_data_cluster, total_data, completed_img, data_cluster_reduced_

# # EDITED 64x 64
DBSCAN_eps = 150
DBSCAN_minsample = 500
min_data_cluster = 1500 #2000
DBSCAN_eps1= 20 #10#12
DBSCAN_minsample1 = 100 #50#100
min_data_cluster1 = 700 #600
# TESTING
tresh = 160#150 140
min_samples_final_clusters = 1100 #2500
min_distance = 500  # min distance among two clusters

# EDITED 64x 64
# DBSCAN_eps = 150
# DBSCAN_minsample = 500
# min_data_cluster = 4000#6062 #2000
# DBSCAN_eps1= 20 #10#12
# DBSCAN_minsample1 = 100 #50#100
# min_data_cluster1 = 2785 #600
# # TESTING
# tresh = 160#150 140
# min_samples_final_clusters = 4000#4423 #2500
# min_distance = 1966  # min distance among two clusters

imgColor = cv2.imread(path)
img = cv2.cvtColor(imgColor, cv2.COLOR_BGR2GRAY)
w, h = tuple(img.shape)
ret1, th3 = cv2.threshold(img, tresh, 255, cv2.THRESH_BINARY)

Start = time.time()
# countours_img= np.zeros((w, h, 3), np.uint8)
# im2, contours, hierarchy = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
# print(len(contours))


data_cluster, data_cluster_reduced,total_data_cluster, total_data, img_r, data_cluster_reduced_ = completing_function(gray_image=th3,
                                                                                                   DBSCAN_eps=DBSCAN_eps,
                                                                                                   DBSCAN_minsample=DBSCAN_minsample,
                                                                                                   min_data_cluster=min_data_cluster,
                                                                                                   min_distance=min_distance)
finish = time.time() - Start


data_cluster = np.asarray(data_cluster)
data_cluster_reduced = np.asarray(data_cluster_reduced)

data_cluster_img = np.zeros((w, h, 3), np.uint8)
data_cluster_reduced_img = np.zeros((w, h, 3), np.uint8)
data_cluster_completed_img = np.zeros((w, h, 3), np.uint8)
completed_img = np.zeros((w, h, 3), np.uint8)

figb, ax = plt.subplots(1, 3)
figb.set_size_inches(18, 8)
ax[0].set_title('CLUSTERING ORIGINAL FIRST STAGE ' + str(len(data_cluster)) + ' CLUSTERS')
for k in range(len(data_cluster)):
    x = [-m[0] for m in data_cluster[k]]
    y = [m[1] for m in data_cluster[k]]
    op = [m[2] for m in data_cluster[k]]
    ax[0].plot(y, x, 'bo', c=np.random.random(3), label=str(op[0]))
ax[0].legend(loc=0)

ax[1].set_title('CLUSTERING SECOND STAGE ' + str(len(data_cluster_reduced_)) + ' - SUBCLUSTERs -')
for k in range(len(data_cluster_reduced_)):
    x = [-m[0] for m in data_cluster_reduced_[k]]
    y = [m[1] for m in data_cluster_reduced_[k]]
    op = [[m[2], m[3]] for m in data_cluster_reduced_[k]]
    ax[1].plot(y, x, 'bo', c=np.random.random(3), label=str(op[0]))
ax[1].legend(loc=0)

ax[2].set_title('MERGING '+str(finish)+' SECONDS')
ax[2].imshow(img_r)





fig2, ax2 = plt.subplots(2,2)
fig2.set_size_inches(18, 8)

for k in data_cluster:
    color=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
    for m in k:
        cv2.circle(data_cluster_completed_img, (m[1], m[0]), 4, color, 4)



clusters = set(data_cluster_reduced[:, 3])
for i in clusters:
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    data_x = data_cluster_reduced[data_cluster_reduced[:, 3] == i][:, 0]
    data_y = data_cluster_reduced[data_cluster_reduced[:, 3] == i][:, 1]
    for j in range(len(data_x)):
        cv2.circle(data_cluster_reduced_img, (int(data_y[j]), int(data_x[j])), 10, color, 5)


clusters = set(total_data_cluster[:, 3])
for i in clusters:
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    data_x = total_data_cluster[total_data_cluster[:, 3] == i][:, 0]
    data_y = total_data_cluster[total_data_cluster[:, 3] == i][:, 1]
    for j in range(len(data_x)):
        cv2.circle(completed_img, (int(data_y[j]), int(data_x[j])), 10, color, 5)
for i in total_data:
    cv2.line(completed_img, (int(i[4][1]), int(i[4][0])), (int(i[3][1]), int(i[3][0])), (195, 242, 74), 50)


ax2[0, 0].set_title('THRESHOLD IMAGE '+str(tresh))
ax2[0, 0].imshow(th3, cmap='gray', vmin=0, vmax=255)
ax2[0, 1].set_title('CLUSTERING ORIGINAL FIRST STAGE ' + str(len(data_cluster)) + ' CLUSTERS')
ax2[0, 1].imshow(data_cluster_completed_img)
ax2[1, 0].set_title('CLUSTERING SECOND STAGE ' + str(len(data_cluster_reduced_)) + ' - SUBCLUSTERs -')
ax2[1, 0].imshow(data_cluster_reduced_img)
ax2[1, 1].set_title('FINAL RESULT')
ax2[1, 1].imshow(completed_img)

plt.tight_layout()



# # ploting
# path_color_org_img = 'C:/Users/LAB01-PC/Desktop/20190601/20190601_128/val/'
#
# img_name = 'C2-3-20190516124511-Processed'
# img_color_org =  cv2.imread(path_color_org_img+img_name+'.bmp')
# img_color_org = cv2.addWeighted(img_color_org, 0.8, img_r, 0.5, 0.0 )
#
#
# img_name_feature = img_name + '_featuremap'
#
# pathComplete = 'C:/Users/LAB01-PC/Desktop/20190601/20190601_128/val/result/2/'
# figb.savefig(pathComplete +img_name+'2.png', dpi=300)
# fig2.savefig(pathComplete +img_name+'FinalResult2.png', dpi=300)
# #figb.savefig(pathComplete +img_name+'CP.png')
#
# cv2.imwrite(pathComplete +img_name + 'T.png',img_color_org)

plt.show()
