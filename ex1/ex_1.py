import math
import sys
import scipy.io.wavfile
# import matplotlib.pyplot as plt

import numpy as np


def oclid_dist(p0, p1):
    dist = 0.0
    for i in range(0, len(p0)):
        dist += (p0[i] - p1[i]) ** 2
    return math.sqrt(dist)


def update(points, cluster_centers, cluster, loss):
    for p in range(0, len(points)):
        min_dist = float("inf")
        min_index = 0
        for c in range(0, len(cluster_centers)):
            dist = oclid_dist(points[p], cluster_centers[c])
            if dist < min_dist:
                min_dist = dist
                min_index = c

        loss += min_dist
        cluster[min_index].append(points[p])

    return loss/len(points)

def define_centroids(cluster_centers, cluster):
    for i in range(0, len(cluster_centers)):
        if not cluster[i]:
            continue
        cluster_centers[i] = np.round(np.mean(cluster[i], axis=0))


def kmeans(cluster_center, points):
    max_iterations = 30
    prev_lost = 0
    f = open("output2.txt", "w")
    for i in range(0, max_iterations):
        loss = 0
        cluster = [[] for _ in range(len(cluster_center))]
        current_loss = update(points, cluster_center, cluster, loss)
        define_centroids(cluster_center, cluster)
        if current_loss == prev_lost:
            break
        prev_lost = current_loss
        f.write(f"[iter {i}]:{','.join([str(i) for i in cluster_center])}\n")

    f.close()
    restult = []
    for p in points:
        min_dist = float("inf")
        min_dist_index = None
        for c in range(0, len(cluster_center)):
            dist = oclid_dist(p, cluster_center[c])
            if dist < min_dist:
                min_dist = dist
                min_dist_index = c
        restult.append(cluster_center[min_dist_index])

    return restult

def main():

    sample, centroids = sys.argv[1], sys.argv[2]
    fs, y = scipy.io.wavfile.read(sample)
    x = np.array(y.copy())
    centroids = np.loadtxt(centroids)
    if centroids.shape == (2,):
        centroids = centroids.reshape((1,2))
    new_values = kmeans(centroids, x)
    scipy.io.wavfile.write("compressed.wav", fs, np.array(new_values, dtype=np.int16))

if __name__ == '__main__':
    main()