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

def centroid_function(data_cluster, number_cluster):
    center_x = np.mean(data_cluster[:, 0])
    center_y = np.mean(data_cluster[:, 1])
    return center_x, center_y, number_cluster

def angle_function(point_1, point_2):
    numerator = point_2[1] - point_1[1]
    denominator = point_2[0] - point_1[0]
    if denominator != 0:
        slope = numerator / denominator
        alpha = math.degrees(math.atan(slope))
        if alpha < 0:
            alpha = 360 + alpha
    else:
        alpha = 90
    # Measure of Reference Angle
    if 90 < alpha <= 180:
        alpha = 180 - alpha
    elif 180 < alpha <= 270:
        alpha = alpha - 180
    elif 270 < alpha <= 360:
        alpha = 360 - alpha

    return alpha

def clustering_function(X, DBSCAN_eps, DBSCAN_minsample, min_data_cluster):
    # DBSCAN

    db = DBSCAN(eps=DBSCAN_eps, min_samples=DBSCAN_minsample).fit(X)

    data = np.array(list(zip(X[:, 0], X[:, 1], db.labels_)))  # keep all  X,Y,Cluster
    data = data[data[:, 2]!= -1]
    index = set(data[:, 2])

    data_cluster =[]
    data_cluster_reduced = []
    len_cluster = []
    for i in index:
        dataAux=data[data[:, 2] == i]
        data_cluster.append(dataAux)

        # len_cluster.append([i, len(dataAux)])
        if len(dataAux) > min_data_cluster:
            data_cluster_reduced.append(dataAux)
            len_cluster.append([i, len(dataAux)])

    # data_cluster_reduced = data_cluster
    # print('Estimated number of clusters %d' % len(data_cluster))
    # print('Estimated number of clusters %d' % len(data_cluster_reduced))
    return data_cluster, data_cluster_reduced, len_cluster

def clusters_analysis(data_cluster, min_distance):
    total_data = []
    #CALCULATE THE CENTERS OF THE EACH CLUSTER
    centroids = [centroid_function(k[:, 0:2], k[0, 2]) for k in data_cluster]
    centroids = np.asarray(centroids)
    x1 = np.zeros(2)
    y1 = np.zeros(2)
    for k in range(len(data_cluster)):
        a = data_cluster[k][:, 0:2]

        for m in range(len(data_cluster) - k - 1):
            n = m + k + 1
            b = data_cluster[n][:, 0:2]
            btree = cKDTree(a)  # two fist columns from dataAux
            dist, idx = btree.query(b)

            if dist.min() < min_distance:
                dist_data = np.argwhere(dist == dist.min())
                idx_data = idx[np.where(dist == dist.min())]
                p1 = a[idx_data[len(idx_data) - 1]]
                p2 = b[int(dist_data[len(dist_data) - 1])]
                n_cluster_1 = data_cluster[k][0, 2]
                n_cluster_2 = data_cluster[n][0, 2]

                slope_p1 = slope(a[: ,0], a[: ,1])
                slope_p2 = slope(b[: ,0], b[: ,1])

                x1[0] = p1[0]
                x1[1] = p2[0]
                y1[0] = p1[1]
                y1[1] = p2[1]
                s = slope(x1, y1)
                print(n_cluster_1, p1, n_cluster_2, p2,dist.min(), slope_p1, slope_p2,s )
                if (abs(s - slope_p2) < 30) or abs(s - slope_p1) < 30:
                    total_data.append([n_cluster_1, p1, n_cluster_2, p2])

                # centroid_1 = centroids[centroids[:, 2] == data_cluster[k][0, 2]]
                # centroid_1 = centroid_1[0][0:2]
                # angle_p1_centroid1 = angle_function(p1, centroid_1)
                # # p2 = b[d_index[0][0]]
                #
                # centroid_2 = centroids[centroids[:, 2] == data_cluster[n][0, 2]]
                # centroid_2 = centroid_2[0][0:2]
                # angle_p1_centroid2 = angle_function(p2, centroid_2)
                # distance_p1_p2 = dist.min()
                # angle_p1_p2 = angle_function(p1, p2)
                #
                # aa = 15 #25
                # ao = 20#30

                # if abs(angle_p1_p2 - angle_p1_centroid1) < aa and abs(angle_p1_p2 - angle_p1_centroid2) < aa and abs(
                #         angle_p1_centroid2 - angle_p1_centroid1) < ao:
                    total_data.append([n_cluster_1, p1, n_cluster_2, p2])

    return np.array(total_data)


def check_cluster(data_cluster_reduced, total_data, len_cluster):
    print('CLUSTERS')
    for i in len_cluster:
        print(i)

    if len(len_cluster) > 0 and len(total_data) > 0:

        len_cluster = np.asarray(len_cluster)
        data_cluster = [[i[0], i[2]] for i in total_data]


        cols = 0
        sum_aux = 0
        d = []
        c = []
        m = []
        m_aux = []

        while len(data_cluster) != 0:

            data_aux = np.array(data_cluster.copy())
            aux = np.argwhere(data_aux == data_aux[cols, 0])
            [d.append(int(j[0])) for j in aux]
            aux = np.argwhere(data_aux == data_aux[cols, 1])
            [d.append(int(j[0])) for j in aux]

            d_aux = sorted(set(d), key=d.index)

            actual_data_1 = data_aux[cols, 0]
            actual_data_2 = data_aux[cols, 1]
            if len(d_aux) > 1:
                for i in d_aux:
                    [c.append(j) for j in data_cluster[i]]
                for i in range(len(d_aux)):
                    data_cluster.pop(0)
                while len(m_aux) != 0:
                    data_aux = np.array(data_cluster.copy())
                    c_aux = sorted(set(c), key=c.index)
                    for i in c_aux:
                        if i != actual_data_1 and i != actual_data_2:
                            # print("data ===========> ", i)
                            aux = np.argwhere(data_aux == i)
                            [m.append(int(j[0])) for j in aux]
                    m_aux = sorted(set(m), key=m.index)
                    if len(m_aux) > 1:
                        for i in m_aux:
                            [c.append(j) for j in data_cluster[i]]
                        for i in range(len(d_aux)):
                            data_cluster.pop(0)
                        #data_aux = np.array(data_cluster.copy())
                    elif len(m_aux) == 1:
                        [c.append(i) for i in data_cluster[d_aux[0]]]
                        data_cluster.pop(d_aux[0])
                    m.clear()
                    d.clear()
                    d_aux.clear()
            elif len(d_aux) == 1:
                [c.append(i) for i in data_cluster[d_aux[0]]]
                data_cluster.pop(d_aux[0])
            d.clear()
            d_aux.clear()
            c_aux = sorted(set(c), key=c.index)
            for i in c_aux:
                aux = np.argwhere(len_cluster == i)
                sum_aux = sum_aux + len_cluster[aux[0, 0], 1]
            for i in c_aux:
                aux = np.argwhere(len_cluster == i)
                len_cluster[aux[0, 0], 1] = sum_aux
            c.clear()
            c_aux.clear()

        # print('FINAL')
        # print(len_cluster)

        data_aux_reduced = []


        reducer = len_cluster[len_cluster[:, 1] < min_data_cluster_2]#2000]#1500] #limiting data cluster
        #reducer = len_cluster[len_cluster[:, 1] < 3000]  # limiting data cluster 128 x 128 2700

        cls = np.unique(reducer[:,0])
        total_data_mod = []
        for i in cls:

           delete_data = np.where((total_data[:,0]==i) | (total_data[:, 2]==i))
           delete_data= delete_data[0]
           total_data= np.delete(total_data, delete_data, axis=0)


        for i in data_cluster_reduced:
            for j in i:
                data_aux_reduced.append(j)
        data_aux_reduced = np.asarray(data_aux_reduced)
        data_cluster_reduced.clear()

        for i in reducer:
            data_aux_reduced = data_aux_reduced[data_aux_reduced[:, 2] != i[0]]

        index = set(data_aux_reduced[:, 2])
        data_cluster_reduced = []
        for i in index:
            dataAux = data_aux_reduced[data_aux_reduced[:, 2] == i]
            data_cluster_reduced.append(dataAux)
    else:

        data_cluster_reduced1 = []
        for i in data_cluster_reduced:
            if len(i) > min_data_cluster_2:
            # if len(i) > 3000:
                data_cluster_reduced1.append(i)
        data_cluster_reduced.clear()
        data_cluster_reduced = data_cluster_reduced1

        # data_cluster = []
        # data_cluster_reduced1 = []
        # len_cluster = []
        # total_data = []
        # for i in index:
        #     dataAux = data_cluster_reduced[data_cluster_reduced[:, 2] == i]
        #     data_cluster.append(dataAux)
        #
        #     # len_cluster.append([i, len(dataAux)])
        #     if len(dataAux) > 1000:
        #         data_cluster_reduced1.append(dataAux)
        #         len_cluster.append([i, len(dataAux)])


    return data_cluster_reduced, total_data

def plotting(data_cluster_reduced, total_data, w, h):
    img = np.zeros((w, h, 1), np.uint8)
    for k in data_cluster_reduced:
        for m in k:
            cv2.circle(img, (m[1], m[0]), 4, 255, 4)
    for k in total_data:
        px1 = k[1][0]
        py1 = k[1][1]
        px2 = k[3][0]
        py2 = k[3][1]
        cv2.line(img, (py1, px1), (py2, px2), 255, 30)
    return img

def completing_function(gray_image, DBSCAN_eps, DBSCAN_minsample, min_data_cluster, min_distance):
    stime=time.time()
    w, h = tuple(gray_image.shape)
    x = np.argwhere(gray_image > 0)
    if len(x) > 0 :
        print('CLUSTERING PROCESS : eps = ', DBSCAN_eps, " Minimun sample = ", DBSCAN_minsample)
        data_cluster, data_cluster_reduced, len_cluster = clustering_function(x, DBSCAN_eps, DBSCAN_minsample, min_data_cluster)
        total_data = clusters_analysis(data_cluster_reduced, min_distance)

        data_cluster_reduced, total_data = check_cluster(data_cluster_reduced, total_data, len_cluster)
        print("Time before building the plot : ", time.time() - stime)
        img = plotting(data_cluster_reduced, total_data, w, h)
        print("Time after building the plot : ", time.time() - stime)
    else:
        data_cluster = []
        data_cluster_reduced = []
        total_data = []
        img = []
    return data_cluster, data_cluster_reduced, total_data, img

# #ORIGINAL
# DBSCAN_eps =12
# DBSCAN_minsample =50
# #TESTING
# tresh = 160
# #First clustering data
# alpha_angle = 22
# min_data_cluster = 500#400#min data of cluster
# min_distance = 200 #230# min distance among two clusters

#EDITED 64x 64
method_size = 64

DBSCAN_eps =12
DBSCAN_minsample =50 #100
#TESTING
# DBSCAN_eps =12
# DBSCAN_minsample =50#100

tresh = 160
# First clustering data
alpha_angle = 22
percentage_image = 10
min_data_cluster = 636#percentage_image * method_size ** 2 / 100 + 10
min_data_cluster_2 = 4 * min_data_cluster
min_distance = 181  # 200# min distance among two clusters

# # #128x128
# #TESTING
# tresh = 160
# #First clustering data
# alpha_angle = 22
# min_data_cluster = 2000#500#400#min data of cluster
# min_distance = 200# min distance among two clusters

imgColor = cv2.imread(path)
img = cv2.cvtColor(imgColor, cv2.COLOR_BGR2GRAY)
w, h = tuple(img.shape)
ret1, th3 = cv2.threshold(img, tresh, 255, cv2.THRESH_BINARY)

Start = time.time()
data_cluster, data_cluster_reduced, total_data, img = completing_function(gray_image = th3, DBSCAN_eps = DBSCAN_eps, DBSCAN_minsample = DBSCAN_minsample, min_data_cluster = min_data_cluster, min_distance = min_distance)
finish = time.time() - Start


data_cluster = np.asarray(data_cluster)
data_cluster_reduced = np.asarray(data_cluster_reduced)

data_cluster_img = np.zeros((w, h, 3), np.uint8)
data_cluster_reduced_img = np.zeros((w, h, 3), np.uint8)
data_cluster_completed_img = np.zeros((w, h, 3), np.uint8)
completed_img = np.zeros((w, h, 3), np.uint8)

final = img
for k in data_cluster_reduced:
    color=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
    for m in k:
        cv2.circle(data_cluster_completed_img, (m[1], m[0]), 4, color, 4)
        cv2.circle(data_cluster_reduced_img, (m[1], m[0]), 4, color, 4)
        cv2.circle(completed_img, (m[1], m[0]), 4, (0, 242, 74), 4)
    for k in total_data:

        px1 = k[1][0]
        py1 = k[1][1]
        px2 = k[3][0]
        py2 = k[3][1]
        cv2.line(data_cluster_completed_img, (py1, px1), (py2, px2), (255, 0, 0), 30)
        cv2.line(completed_img, (py1, px1), (py2, px2), (0, 242, 74), 30)

#ploting



figa, axarr = plt.subplots(1, 3)
figa.suptitle('RESULTS ' +str(finish))
figa.set_size_inches(18, 8)
axarr[0].set_title('ORIGINAL IMAGE ')
axarr[0].imshow(imgColor)
axarr[1].set_title('THRESHOLD ' + str(tresh))
axarr[1].imshow(th3, cmap='gray', vmin=0, vmax=255)
axarr[2].set_title('CLUSTERING PROCESS')
axarr[2].imshow(data_cluster_completed_img)



figb, ax = plt.subplots(1,3)
figb.set_size_inches(18, 8)
ax[0].set_title('CLUSTERING ORIGINAL IMAGE ' + str(len(data_cluster)) + ' CLUSTERS')
for k in range(len(data_cluster)):
    x = [m[1] for m in data_cluster[k]]
    y = [-m[0] for m in data_cluster[k]]
    op = [m[2] for m in data_cluster[k]]
    ax[0].plot(x, y, 'bo', c=np.random.random(3), label=str(op[0]))
ax[0].legend(loc=0)

ax[1].set_title('CLUSTERING REDUCED ' + str(len(data_cluster_reduced)) + ' CLUSTERS')
for k in range(len(data_cluster_reduced)):
    x = [m[1] for m in data_cluster_reduced[k]]
    y = [-m[0] for m in data_cluster_reduced[k]]
    op = [m[2] for m in data_cluster_reduced[k]]
    ax[1].plot(x, y, 'bo', c=np.random.random(3), label=str(op[0]))
ax[1].legend(loc=0)

ax[2].set_title("FINAL")
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
ax[2] = plt.imshow(img)


# path_color_org_img = 'C:/Users/LAB01-PC/Desktop/20190601/20190601_128/val/'
#
# img_name = 'C2-3-20190516124511-Processed'
# img_color_org= cv2.imread(path_color_org_img+img_name+'.bmp')
# img_color_org = cv2.addWeighted(img_color_org, 0.8, completed_img, 0.5, 0.0 )
#
#
# img_name_feature = img_name + '_featuremap'
# pathComplete = 'C:/Users/LAB01-PC/Desktop/20190601/20190601_128/val/result/1/'
# figa.savefig(pathComplete +img_name+'1.png', dpi=300)
# figb.savefig(pathComplete +img_name+'FinalResult1.png', dpi=300)
# #figb.savefig(pathComplete +img_name+'CP.png')
#
# cv2.imwrite(pathComplete +img_name + 'T.png',img_color_org)

plt.show()

