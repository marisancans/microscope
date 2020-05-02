import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import hdbscan
from scipy.spatial.distance import cdist
from dataclasses import dataclass
import hdbscan
plt.ion()

# taken from https://hdbscan.readthedocs.io/en/latest/soft_clustering_explanation.html
# this cluster uses HBDSCAN soft clustering
# each data point gets a vector of scores for how close the point is to each cluster
# to calculate labels and similarity / membership scoring for each point create class EmbeddingClusterer and pass embeds to __call__ func


def exemplars(cluster_id, condensed_tree):
    raw_tree = condensed_tree._raw_tree
    # Just the cluster elements of the tree, excluding singleton points
    cluster_tree = raw_tree[raw_tree['child_size'] > 1]
    # Get the leaf cluster nodes under the cluster we are considering
    leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, cluster_id)
    # Now collect up the last remaining points of each leaf cluster (the heart of the leaf)
    result = np.array([])
    for leaf in leaves:
        max_lambda = raw_tree['lambda_val'][raw_tree['parent'] == leaf].max()
        points = raw_tree['child'][(raw_tree['parent'] == leaf) &
                                (raw_tree['lambda_val'] == max_lambda)]
        result = np.hstack((result, points))
    return result.astype(np.int)


def min_dist_to_exemplar(point, cluster_exemplars, data):
    dists = cdist([data[point]], data[cluster_exemplars.astype(np.int32)])
    return dists.min()


def dist_vector(point, exemplar_dict, data):
    result = {}
    for cluster in exemplar_dict:
        result[cluster] = min_dist_to_exemplar(point, exemplar_dict[cluster], data)
    return np.array(list(result.values()))


def dist_membership_vector(point, exemplar_dict, data, softmax=False):
    with np.errstate(divide='ignore',invalid='ignore'): # scoped error suppression, not ideal way to use this
        if softmax:
            result = np.exp(1./dist_vector(point, exemplar_dict, data))
            result[~np.isfinite(result)] = np.finfo(np.double).max
        else:
            result = 1./dist_vector(point, exemplar_dict, data)
            result[~np.isfinite(result)] = np.finfo(np.double).max
        result /= result.sum()
    return result


def max_lambda_val(cluster, tree):
    cluster_tree = tree[tree['child_size'] > 1]
    leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, cluster)
    max_lambda = 0.0
    for leaf in leaves:
        max_lambda = max(max_lambda,
                        tree['lambda_val'][tree['parent'] == leaf].max())
    return max_lambda


def points_in_cluster(cluster, tree):
    leaves = hdbscan.plots._recurse_leaf_dfs(tree, cluster)
    return leaves   


def merge_height(point, cluster, tree, point_dict):
    cluster_row = tree[tree['child'] == cluster]

    if point in point_dict[cluster]:
        merge_row = tree[tree['child'] == float(point)][0]
        return merge_row['lambda_val']
    else:
        while point not in point_dict[cluster]:
            parent_row = tree[tree['child'] == cluster]
            cluster = parent_row['parent'].astype(np.float64)[0]
        for row in tree[tree['parent'] == cluster]:
            child_cluster = float(row['child'])
            if child_cluster == point:
                return row['lambda_val']
            if child_cluster in point_dict and point in point_dict[child_cluster]:
                return row['lambda_val']


def per_cluster_scores(point, cluster_ids, tree, max_lambda_dict, point_dict):
    result = {}
    point_row = tree[tree['child'] == point]
    point_cluster = float(point_row[0]['parent'])
    max_lambda = max_lambda_dict[point_cluster] + 1e-8 # avoid zero lambda vals in odd cases

    for c in cluster_ids:
        height = merge_height(point, c, tree, point_dict)
        result[c] = (max_lambda / (max_lambda - height))
    return result


def outlier_membership_vector(point, cluster_ids, tree,
                            max_lambda_dict, point_dict, softmax=True):
    if softmax:
        result = np.exp(np.array(list(per_cluster_scores(point, cluster_ids, tree, max_lambda_dict, point_dict).values())))
        result[~np.isfinite(result)] = np.finfo(np.double).max
    else:
        result = np.array(list(per_cluster_scores(point, cluster_ids, tree, max_lambda_dict, point_dict).values()))
    result /= result.sum()
    return result


def combined_membership_vector(point, data, tree, exemplar_dict, cluster_ids, max_lambda_dict, point_dict, softmax=False):
    raw_tree = tree._raw_tree
    dist_vec = dist_membership_vector(point, exemplar_dict, data, softmax)
    outl_vec = outlier_membership_vector(point, cluster_ids, raw_tree,
                                        max_lambda_dict, point_dict, softmax)
    result = dist_vec * outl_vec
    result /= result.sum()
    return result


def prob_in_some_cluster(point, tree, cluster_ids, point_dict, max_lambda_dict):
    heights = []
    for cluster in cluster_ids:
        heights.append(merge_height(point, cluster, tree._raw_tree, point_dict))
    height = max(heights)
    nearest_cluster = cluster_ids[np.argmax(heights)]
    max_lambda = max_lambda_dict[nearest_cluster]
    return height / max_lambda

@dataclass
class Clusterer:
    cluster_algo: hdbscan.HDBSCAN
    pca_reduce: int 
    debug: bool
        
    def __call__(self, embeds, pca=None):
        data = np.array(embeds)

        if pca:
            data = pca.fit_transform(data)
            data = data.astype(np.double)
        
        clusterer = self.cluster_algo
        clusterer = clusterer.fit(data)


        tree = clusterer.condensed_tree_
        exemplar_dict = {c:exemplars(c,tree) for c in tree._select_clusters()}
        cluster_ids = tree._select_clusters()
        raw_tree = tree._raw_tree
        all_possible_clusters = np.arange(data.shape[0], raw_tree['parent'].max() + 1).astype(np.float64)
        max_lambda_dict = {c:max_lambda_val(c, raw_tree) for c in all_possible_clusters}
        point_dict = {c:set(points_in_cluster(c, raw_tree)) for c in all_possible_clusters}
        
        labels = clusterer.labels_
        labels = [ str(x) for x in labels ] # make labels string
        mvs = [] # membership vectors

        if self.debug and self.pca_reduce == 2:
            a_set = list(set(clusterer.labels_))
            n_colors = len(a_set)
            pal = {k:v for k, v in zip(a_set, sns.color_palette('deep', n_colors))}
            

            # colors = [sns.desaturate(pal[col], sat) for col, sat in zip(clusterer.labels_, clusterer.probabilities_)]
            colors = [pal[col] for i, col in enumerate(clusterer.labels_)]
            # colors = np.empty((data.shape[0], 3))
        
        # for x in range(data.shape[0]):
            # membership_vector = combined_membership_vector(x, data, tree, exemplar_dict, cluster_ids, max_lambda_dict, point_dict, False)
            # membership_vector *= prob_in_some_cluster(x, tree, cluster_ids, point_dict, max_lambda_dict)
            
            # if self.debug and self.pca_reduce == 2:
                # color = np.argmax(membership_vector)
                # saturation = membership_vector[color]
                # colors[x] = sns.desaturate(pal[color], saturation)

            # mvs.append(membership_vector)            



        return labels, mvs, colors, data
