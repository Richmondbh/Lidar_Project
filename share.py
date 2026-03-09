'''
point cloud data is stored as a 2D matrix
each row has 3 values i.e. the x, y, z value for a point

Project has to be submitted to github in the private folder assigned to you
Readme file should have the numerical values as described in each task
Create a folder to store the images as described in the tasks.

Try to create commits and version for each task.

'''
#%%
import matplotlib
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

from sklearn.neighbors import NearestNeighbors

os.makedirs('images', exist_ok=True)  # folder to store all output plots
#%% utility functions
def show_cloud(points_plt):
    ax = plt.axes(projection='3d')
    ax.scatter(points_plt[:,0], points_plt[:,1], points_plt[:,2], s=0.01)
    plt.show()

def show_scatter(x,y):
    plt.scatter(x, y)
    plt.show()
# ════════════════════════════════════════════════════════════════════════════
# TASK 1 –  updated ground level function
# ════════════════════════════════════════════════════════════════════════════
def get_ground_level(pcd):
    counts, bin_edges = np.histogram(pcd[:, 2], bins=200)
    peak_bin = np.argmax(counts)
    ground_z = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0
    return ground_z


#%% read file containing point cloud data
pcd = np.load("dataset1.npy")
print("Dataset1 shape:", pcd.shape)



#%% show downsampled data in external window
# %matplotlib qt  
#show_cloud(pcd)
#show_cloud(pcd[::10]) # keep every 10th point

# ════════════════════════════════════════════════════════════════════════════
#  Ground level detection
# ════════════════════════════════════════════════════════════════════════════
#%%
'''
Task 1 (3)
find the best value for the ground level
One way to do it is useing a histogram 
np.histogram

update the function get_ground_level() with your changes

For both the datasets
Report the ground level in the readme file in your github project
Add the histogram plots to your project readme
'''
est_ground_level = get_ground_level(pcd)
print(est_ground_level)

# Ploting the Z histogram so the ground peak is clearly visible
plt.figure(figsize=(8, 4))
plt.hist(pcd[:, 2], bins=200, color='steelblue', edgecolor='none')
plt.axvline(est_ground_level, color='red', linestyle='--', linewidth=1.5,
            label=f'Ground level = {est_ground_level:.3f}')
plt.xlabel('Z (height)')
plt.ylabel('Number of points')
plt.title('Dataset1 – Z-value Histogram (Ground Detection)')
plt.legend()
plt.tight_layout()
plt.savefig('images/dataset1_task1_histogram.png', dpi=150)
plt.show()

#%% filter out ground points
pcd_above_ground = pcd[pcd[:,2] > est_ground_level] 
print("Points above ground:", pcd_above_ground.shape)




#%%
'''
Task 2 (+1)

Find an optimized value for eps.
Plot the elbow and extract the optimal value from the plot
Apply DBSCAN again with the new eps value and confirm visually that clusters are proper

https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/
https://machinelearningknowledge.ai/tutorial-for-dbscan-clustering-in-python-sklearn/

For both the datasets
Report the optimal value of eps in the Readme to your github project
Add the elbow plots to your github project Readme
Add the cluster plots to your github project Readme
'''
# ════════════════════════════════════════════════════════════════════════════
#  Optimal eps using the k-NN elbow method
# ════════════════════════════════════════════════════════════════════════════
# %%
pcd_xy = pcd_above_ground[::4, :2].astype(np.float32)
k = 5
nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(pcd_xy[::3])
distances, _ = nbrs.kneighbors(pcd_xy[::3])
k_distances = np.sort(distances[:, k - 1])

# Locate elbow as the point of maximum second derivative
second_diff = np.diff(np.diff(k_distances))
elbow_idx   = int(np.argmax(second_diff)) + 1
optimal_eps = float(k_distances[elbow_idx])
print(f"Elbow eps: {optimal_eps:.3f}  (using eps = 1.5 for clustering)")
#%% plot the elbow
plt.figure(figsize=(8, 4))
plt.plot(k_distances, color='steelblue', linewidth=0.8)
plt.scatter(elbow_idx, k_distances[elbow_idx], color='red', zorder=5, s=50,
            label=f'Elbow → eps = {optimal_eps:.3f}')
plt.xlabel('Points sorted by k-NN distance')
plt.ylabel('5th nearest-neighbour distance')
plt.title('Dataset1 – Elbow Plot for Optimal eps')
plt.legend()
plt.tight_layout()
plt.savefig('images/dataset1_task2_elbow.png', dpi=150)
plt.show()

#%% run DBSCAN with optimised eps
eps_optimised = 1.5    
clustering = DBSCAN(eps=eps_optimised, min_samples=5,
                    algorithm='kd_tree').fit(pcd_xy)
clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
print(f"DBSCAN clusters found: {clusters}")

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, clusters)]

# %%
# Plotting resulting clusters
plt.figure(figsize=(10,10))
plt.scatter(pcd_xy[:,0], 
            pcd_xy[:,1],
            c=clustering.labels_,
            cmap=matplotlib.colors.ListedColormap(colors),
            s=2)


plt.title('Dataset1 – DBSCAN: %d clusters' % clusters,fontsize=20)
plt.xlabel('x axis',fontsize=14)
plt.ylabel('y axis',fontsize=14)
plt.tight_layout()
plt.savefig('images/dataset1_task2_clusters.png', dpi=150)
plt.show()




#%%
'''
Task 3 (+1)

Find the largest cluster, since that should be the catenary, 
beware of the noise cluster.

Use the x,y span for the clusters to find the largest cluster

For both the datasets
Report min(x), min(y), max(x), max(y) for the catenary cluster in the Readme of your github project
Add the plot of the catenary cluster to the readme

'''
# The catenary wire runs the full length of the track, so its cluster has
# the largest XY bounding box span. Label -1 is noise and must be skipped.
best_label = None
best_span  = -1

for label in set(clustering.labels_):
    if label == -1:       # skip noise cluster
        continue
    pts    = pcd_xy[clustering.labels_ == label]
    x_span = pts[:, 0].max() - pts[:, 0].min()
    y_span = pts[:, 1].max() - pts[:, 1].min()
    span   = max(x_span, y_span)
    if span > best_span:
        best_span  = span
        best_label = label

catenary_pts = pcd_xy[clustering.labels_ == best_label]
print(f"Catenary cluster label : {best_label}")
print(f"min(x) = {catenary_pts[:,0].min():.3f}")
print(f"max(x) = {catenary_pts[:,0].max():.3f}")
print(f"min(y) = {catenary_pts[:,1].min():.3f}")
print(f"max(y) = {catenary_pts[:,1].max():.3f}")

#%% plot the catenary cluster
plt.figure(figsize=(10, 5))
plt.scatter(catenary_pts[:, 0], catenary_pts[:, 1], c='crimson', s=2)
plt.title('Dataset1 – Catenary Cluster (label %d)' % best_label, fontsize=20)
plt.xlabel('x axis', fontsize=14)
plt.ylabel('y axis', fontsize=14)
plt.tight_layout()
plt.savefig('images/dataset1_task3_catenary.png', dpi=150)
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# DATASET 2 – full pipeline
# ════════════════════════════════════════════════════════════════════════════
#%%
pcd = np.load("dataset2.npy")
print("\nDataset2 shape:", pcd.shape)

# Task 1
est_ground_level = get_ground_level(pcd)
print("Estimated ground level:", est_ground_level)

plt.figure(figsize=(8, 4))
plt.hist(pcd[:, 2], bins=200, color='steelblue', edgecolor='none')
plt.axvline(est_ground_level, color='red', linestyle='--', linewidth=1.5,
            label=f'Ground level = {est_ground_level:.3f}')
plt.xlabel('Z (height)')
plt.ylabel('Number of points')
plt.title('Dataset2 – Z-value Histogram (Ground Detection)')
plt.legend()
plt.tight_layout()
plt.savefig('images/dataset2_task1_histogram.png', dpi=150)
plt.show()

pcd_above_ground = pcd[pcd[:, 2] > est_ground_level]
print("Points above ground:", pcd_above_ground.shape)

# Task 2
pcd_xy = pcd_above_ground[::4, :2].astype(np.float32)

nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(pcd_xy[::3])
distances, _ = nbrs.kneighbors(pcd_xy[::3])
k_distances = np.sort(distances[:, k - 1])
second_diff = np.diff(np.diff(k_distances))
elbow_idx   = int(np.argmax(second_diff)) + 1
optimal_eps = float(k_distances[elbow_idx])
print(f"Elbow eps: {optimal_eps:.3f}  (using eps = 1.5 for clustering)")

plt.figure(figsize=(8, 4))
plt.plot(k_distances, color='steelblue', linewidth=0.8)
plt.scatter(elbow_idx, k_distances[elbow_idx], color='red', zorder=5, s=50,
            label=f'Elbow → eps = {optimal_eps:.3f}')
plt.xlabel('Points sorted by k-NN distance')
plt.ylabel('5th nearest-neighbour distance')
plt.title('Dataset2 – Elbow Plot for Optimal eps')
plt.legend()
plt.tight_layout()
plt.savefig('images/dataset2_task2_elbow.png', dpi=150)
plt.show()

clustering = DBSCAN(eps=eps_optimised, min_samples=5,
                    algorithm='kd_tree').fit(pcd_xy)
clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
print(f"DBSCAN clusters found: {clusters}")

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, clusters)]

plt.figure(figsize=(10, 10))
plt.scatter(pcd_xy[:, 0],
            pcd_xy[:, 1],
            c=clustering.labels_,
            cmap=matplotlib.colors.ListedColormap(colors),
            s=2)
plt.title('Dataset2 – DBSCAN: %d clusters' % clusters, fontsize=20)
plt.xlabel('x axis', fontsize=14)
plt.ylabel('y axis', fontsize=14)
plt.tight_layout()
plt.savefig('images/dataset2_task2_clusters.png', dpi=150)
plt.show()

# Task 3
best_label = None
best_span  = -1

for label in set(clustering.labels_):
    if label == -1:
        continue
    pts    = pcd_xy[clustering.labels_ == label]
    x_span = pts[:, 0].max() - pts[:, 0].min()
    y_span = pts[:, 1].max() - pts[:, 1].min()
    span   = max(x_span, y_span)
    if span > best_span:
        best_span  = span
        best_label = label

catenary_pts = pcd_xy[clustering.labels_ == best_label]
print(f"Catenary cluster label : {best_label}")
print(f"min(x) = {catenary_pts[:,0].min():.3f}")
print(f"max(x) = {catenary_pts[:,0].max():.3f}")
print(f"min(y) = {catenary_pts[:,1].min():.3f}")
print(f"max(y) = {catenary_pts[:,1].max():.3f}")

plt.figure(figsize=(10, 5))
plt.scatter(catenary_pts[:, 0], catenary_pts[:, 1], c='crimson', s=2)
plt.title('Dataset2 – Catenary Cluster (label %d)' % best_label, fontsize=20)
plt.xlabel('x axis', fontsize=14)
plt.ylabel('y axis', fontsize=14)
plt.tight_layout()
plt.savefig('images/dataset2_task3_catenary.png', dpi=150)
plt.show()