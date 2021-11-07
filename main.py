import math

import matplotlib.pyplot as plt
import numpy as np
import sys


def pixels_distance(pixel1, pixel2):
    return math.sqrt(pow(pixel1[0] - pixel2[0], 2) + pow(pixel1[1] - pixel2[1], 2) + pow(pixel1[2] - pixel2[2], 2))


image_fname, centroids_fname, out_fname = sys.argv[1], sys.argv[2], sys.argv[3]
z = np.loadtxt(centroids_fname)  # load centroids

orig_pixels = plt.imread(image_fname)
pixels = orig_pixels.astype(float) / 255.

clusters = [[centroid, []] for centroid in z]

out = open(out_fname, "w")

for i in range(20):
    for row in pixels:
        for pixel in row:
            min_dist = 2
            cluster = None
            for entry in clusters:
                dist = pixels_distance(pixel, entry[0])
                if dist < min_dist:
                    min_dist = dist
                    cluster = entry[1]
            cluster.append(pixel)
    new_centroid_list = []
    for cluster in clusters:
        new_cent = np.mean(cluster[1], axis=0).round(4)     # normalize cluster's centroid

        # if cluster[0] == new_cent:
        #     break
        new_centroid_list.append(new_cent)
        cluster[0] = new_cent

    out.write(f"[iter {i}]:{','.join([str(j) for j in new_centroid_list])}\n")
print("Done.")

# Reshape the image(128x128x3) into an Nx3 matrix where N = number of pixels.
pixels = pixels.reshape(-1, 3)

out.close()
