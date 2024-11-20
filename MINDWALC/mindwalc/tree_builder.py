import MINDWALC.mindwalc.datastructures as ds
from sklearn.base import ClassifierMixin, TransformerMixin, BaseEstimator
from collections import Counter
import copy
import numpy as np
import itertools
from joblib import Parallel, delayed
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from scipy.stats import entropy
import time
import ray
import psutil

#default parameters:
path_min_depth_default = 1

@ray.remote
def _calculate_igs(neighborhoods, labels, walks):
    prior_entropy = entropy(np.unique(labels, return_counts=True)[1])
    results = []
    for (vertex, depth) in walks:
        features = {0: [], 1: []}
        for inst, label in zip(neighborhoods, labels):
            features[int(inst.find_walk(vertex, depth))].append(label)
        pos_frac = len(features[1]) / len(neighborhoods)
        pos_entr = entropy(np.unique(features[1], return_counts=True)[1])
        neg_frac = len(features[0]) / len(neighborhoods)
        neg_entr = entropy(np.unique(features[0], return_counts=True)[1])
        ig = prior_entropy - (pos_frac * pos_entr + neg_frac * neg_entr)

        results.append((ig, (vertex, depth)))

    return results

class MINDWALCMixin():
    def __init__(self, path_max_depth=8, progress=None, n_jobs=1, init=True, fixed_walc_depth=True, path_min_depth=path_min_depth_default):
        if init:
            if n_jobs == -1:
                n_jobs = psutil.cpu_count(logical=False)
            ray.shutdown()
            ray.init(num_cpus=n_jobs, ignore_reinit_error=True)
        self.path_max_depth = path_max_depth
        self.path_min_depth = path_min_depth
        self.progress = progress
        self.n_jobs = n_jobs
        self.fixed_walc_depth = fixed_walc_depth

    def _generate_candidates(self, neighborhoods, sample_frac=None, 
                             useless=None, fixed_walc_depth=True):
        """Generates an iterable with all possible walk candidates."""

        # Generate a set of all possible (vertex, depth) combinations
        walks = set()
        for neighborhood in neighborhoods:
            for d in neighborhood.depth_map.keys():
                for vertex in neighborhood.depth_map[d]:
                    if fixed_walc_depth:
                        walks.add((vertex, d))
                    else: # in flexible depth mode, each walk is a range, which describes where the vertex can be found
                        walks.add((vertex, (self.path_min_depth, self.path_max_depth)))

        # Prune the useless ones if provided
        if useless is not None:
            old_len = len(walks)
            walks = walks - useless

        # Convert to list so we can sample & shuffle
        walks = list(walks)

        # Sample if sample_frac is provided
        if sample_frac is not None:
            walks_ix = np.random.choice(range(len(walks)), replace=False,
                                        size=int(sample_frac * len(walks)))
            walks = [walks[i] for i in walks_ix]

        # Shuffle the walks (introduces stochastic behaviour to cut ties
        # with similar information gains)
        np.random.shuffle(walks)

        return walks

    def _feature_map(self, walk, neighborhoods, labels):
        """Create two lists of labels of neighborhoods for which the provided
        walk can be found, and a list of labels of neighborhoods for which 
        the provided walk cannot be found."""
        features = {0: [], 1: []}
        vertex, depth = walk
        for i, (inst, label) in enumerate(zip(neighborhoods, labels)):
            features[int(inst.find_walk(vertex, depth))].append(label)
        return features


    def _mine_walks(self, neighborhoods, labels, n_walks=1, sample_frac=None,
                    useless=None, fixed_walc_depth=True):
        """Mine the top-`n_walks` walks that have maximal information gain.
        returns walks in shape (vertex, depth) with depth = int, for fixed walk or (int, int) for flexible walk.
        If depth = (int, int, int), this means that the flexible and the fixed walk to this vertex have same info gain."""

        if type(fixed_walc_depth) is bool:
            walk_iterator = self._generate_candidates(neighborhoods,
                                                      sample_frac=sample_frac,
                                                      useless=useless, fixed_walc_depth=fixed_walc_depth)

        elif fixed_walc_depth is None:
            walk_iterator_fix = self._generate_candidates(neighborhoods,
                                                          sample_frac=sample_frac,
                                                          useless=useless, fixed_walc_depth=True)
            walk_iterator_flex = self._generate_candidates(neighborhoods,
                                                           sample_frac=sample_frac,
                                                           useless=useless, fixed_walc_depth=False)
            walk_iterator = walk_iterator_fix + walk_iterator_flex
        else:
            raise ValueError("fixed_walc_depth must be a boolean or None")


        neighborhoods_id = ray.put(neighborhoods)
        labels_id = ray.put(labels)
        walks_id = ray.put(walk_iterator)
        chunk_size = int(np.ceil(len(walk_iterator) / self.n_jobs))

        results = ray.get(
            [_calculate_igs.remote(neighborhoods_id, labels_id,
                                   walk_iterator[i*chunk_size:(i+1)*chunk_size]) 
             for i in range(self.n_jobs)]
        )

        if n_walks > 1:
            top_walks = ds.TopQueue(n_walks)
        else:
            max_ig, best_depth, top_walk = 0, float('inf'), None

        for data in results:
            for ig, (vertex, depth) in data:
                if n_walks > 1:
                    if type(depth) is tuple:
                        top_walks.add((vertex, depth[1]), (ig, -depth[1]))
                    else:
                        top_walks.add((vertex, depth), (ig, -depth))
                else:
                    if ig > max_ig:
                        max_ig = ig
                        best_depth = depth
                        top_walk = (vertex, depth)
                    elif ig == max_ig: # if same info gain, use the one with smaller depth:

                        # but: depth can be an int (fixed walking depth) or a tuple (flexible walking depth)
                        best_depth_is_fix = type(best_depth) is not tuple
                        depth_is_fix = type(depth) is not tuple

                        if best_depth_is_fix and depth_is_fix:
                            if depth < best_depth:
                                max_ig = ig
                                best_depth = depth
                                top_walk = (vertex, depth)
                        elif best_depth_is_fix and not depth_is_fix:
                            # this tuple of len 3 communicates that the flexible and the fixed walk to this vertex have same info gain
                            best_depth = (depth[0], depth[1], best_depth)
                            top_walk = (vertex, best_depth)
                        elif not best_depth_is_fix and depth_is_fix:
                            # this tuple of len 3 communicates that the flexible and the fixed walk to this vertex have same info gain
                            best_depth = (best_depth[0], best_depth[1], depth)
                            top_walk = (vertex, best_depth)
                        else: # both are flexible
                            pass

        if n_walks > 1:
            return top_walks.data
        else:
            return [(max_ig, top_walk)]

    def _prune_useless(self, neighborhoods, labels):
        """Provide a set of walks that can either be found in all 
        neighborhoods or 1 or less neighborhoods."""
        useless = set()
        walk_iterator = self._generate_candidates(neighborhoods, fixed_walc_depth=self.fixed_walc_depth)
        for (vertex, depth) in walk_iterator:
            features = self._feature_map((vertex, depth), neighborhoods, labels)
            if len(features[1]) <= 1 or len(features[1]) == len(neighborhoods):
                useless.add((vertex, depth))
        return useless

    def fit(self, kg, instances, labels):
        if self.progress is not None:
            inst_it = self.progress(instances, desc='Neighborhood extraction')
        else:
            inst_it = instances

        self.neighborhoods = []
        for inst in inst_it:
            neighborhood = kg.extract_neighborhood(inst, depth=self.path_max_depth)
            self.neighborhoods.append(neighborhood)


class MINDWALCTree(BaseEstimator, ClassifierMixin, MINDWALCMixin):
    def __init__(self, path_max_depth=8, min_samples_leaf=1, 
                 progress=None, max_tree_depth=None, n_jobs=1,
                 init=True, fixed_walc_depth=True, path_min_depth=path_min_depth_default):
        super().__init__(path_max_depth, progress, n_jobs, init, fixed_walc_depth, path_min_depth)
        self.min_samples_leaf = min_samples_leaf
        self.max_tree_depth = max_tree_depth

    def _stop_condition(self, neighborhoods, labels, curr_tree_depth):
        return (len(set(labels)) == 1 
                or len(neighborhoods) <= self.min_samples_leaf 
                or (self.max_tree_depth is not None 
                    and curr_tree_depth >= self.max_tree_depth))

    def _build_tree(self, neighborhoods, labels, curr_tree_depth=0, 
                    vertex_sample=None, useless=None):

        majority_class = Counter(labels).most_common(1)[0][0]
        if self._stop_condition(neighborhoods, labels, curr_tree_depth):
            return ds.Tree(walk=None, _class=majority_class)

        walks = self._mine_walks(neighborhoods, labels, 
                                 sample_frac=vertex_sample, 
                                 useless=useless, fixed_walc_depth=self.fixed_walc_depth)

        if len(walks) == 0 or walks[0][0] == 0:
            return ds.Tree(walk=None, _class=majority_class)

        _, best_walk = walks[0]
        best_vertex, best_depth = best_walk

        node = ds.Tree(walk=best_walk, _class=None)

        found_neighborhoods, found_labels = [], []
        not_found_neighborhoods, not_found_labels = [], []
        
        for neighborhood, label in zip(neighborhoods, labels):
            if neighborhood.find_walk(best_vertex, best_depth):
                found_neighborhoods.append(neighborhood)
                found_labels.append(label)
            else:
                not_found_neighborhoods.append(neighborhood)
                not_found_labels.append(label)
            
        node.right = self._build_tree(found_neighborhoods, found_labels, 
                                      curr_tree_depth=curr_tree_depth + 1,
                                      vertex_sample=vertex_sample,
                                      useless=useless)
            
        node.left = self._build_tree(not_found_neighborhoods, not_found_labels, 
                                     curr_tree_depth=curr_tree_depth + 1,
                                     vertex_sample=vertex_sample,
                                     useless=useless)
        
        return node

    def fit(self, kg, instances, labels, post_prune=False, post_prune_method='reduced_error_pruning'): # touched

        super().fit(kg, instances, labels)
        useless = self._prune_useless(self.neighborhoods, labels)
        self.tree_ = self._build_tree(self.neighborhoods, labels,
                                      useless=useless)

        if post_prune:
            self.post_prune_tree(self.validate_tree(kg, instances, labels), method=post_prune_method)

    def predict(self, kg, instances):
        preds = []
        for inst in instances:
            neighborhood = kg.extract_neighborhood(inst, depth=self.path_max_depth)
            preds.append(self.tree_.evaluate(neighborhood))
        return preds

    # new:
    def validate_tree(self, kg, data_ents, data_label):
        '''
        returns a dictionary of the data distribution in the tree.
        shape of the dictionary:
        {decition_node: {label_a: [ent1, ent2, ...], label_b: [ent1, ent2, ...], ...}, ...}
        '''
        # for each decition node, count how many reports are visited:
        data_distribution_in_tree = {}
        for i, inst in enumerate(data_ents):  # for each train report:
            neighborhood = kg.extract_neighborhood(inst, depth=self.path_max_depth)
            path = []
            self.tree_.evaluate_traced(neighborhood, path)
            for node in path:
                if node in data_distribution_in_tree.keys():
                    data_distribution_in_tree[node].append((inst, data_label[i]))
                else:
                    data_distribution_in_tree[node] = [(inst, data_label[i])]
        # order each report list by disorder-label:
        for decition_node in data_distribution_in_tree.keys():
            ents_in_decition_node = data_distribution_in_tree[decition_node]
            label_to_ents = {}
            for label in set(data_label):
                if label not in label_to_ents.keys():
                    label_to_ents[label] = []
                label_to_ents[label] += [e[0] for e in ents_in_decition_node if e[1] == label]

            data_distribution_in_tree[decition_node] = label_to_ents
        return data_distribution_in_tree

    def post_prune_tree(self, data_distribution_in_tree, method='reduced_error_pruning', tolerance=1):
        if method != 'reduced_error_pruning':
            raise NotImplementedError('Only reduced error pruning is implemented')
        found_something_to_prune = True
        while found_something_to_prune:
            found_something_to_prune = False
            for node in data_distribution_in_tree.keys():
                if node._class:
                    continue
                if node.left._class and node.right._class:  # is decition-leaf node?
                    if node.left._class == node.right._class:  # we can directly cut off such unnecessary leafs
                        node._class = node.left._class
                        node.left = None
                        node.right = None
                        node.walk = None
                        found_something_to_prune = True
                    else:

                        left_label = node.left._class
                        right_label = node.right._class

                        amount_correct_instances_in_left = len(data_distribution_in_tree[node][left_label])
                        amount_correct_instances_in_right = len(data_distribution_in_tree[node][right_label])
                        winner_leaf = node.left if amount_correct_instances_in_left > amount_correct_instances_in_right else node.right
                        winner_label = left_label if amount_correct_instances_in_left > amount_correct_instances_in_right else right_label
                        errors_winner_leaf = sum([len(data_distribution_in_tree[node][l]) for l in
                                                  data_distribution_in_tree[node].keys() if l != winner_label])

                        errors_left_leaf = sum([len(data_distribution_in_tree[node.left][l]) for l in
                                                data_distribution_in_tree[node.left].keys() if l != left_label])
                        errors_right_leaf = sum([len(data_distribution_in_tree[node.right][l]) for l in
                                                 data_distribution_in_tree[node.right].keys() if
                                                 l != right_label])
                        errors_decition_node = errors_left_leaf + errors_right_leaf

                        if errors_winner_leaf <= errors_decition_node + tolerance:
                            node._class = winner_label
                            node.left = None
                            node.right = None
                            node.walk = None
                            found_something_to_prune = True


class MINDWALCForest(BaseEstimator, ClassifierMixin, MINDWALCMixin):
    def __init__(self, path_max_depth=1, min_samples_leaf=1, 
                 max_tree_depth=None, n_estimators=10, bootstrap=True, 
                 vertex_sample=0.9, progress=None, n_jobs=1, fixed_walc_depth=True, path_min_depth=path_min_depth_default):
        super().__init__(path_max_depth, progress, n_jobs)
        self.min_samples_leaf = min_samples_leaf
        self.max_tree_depth = max_tree_depth
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.vertex_sample = vertex_sample
        self.fixed_walc_depth = fixed_walc_depth
        self.path_min_depth = path_min_depth


    def _create_estimator(self):
        np.random.seed()

        # Bootstrap the instances if required
        if self.bootstrap:
            sampled_inst_ix = np.random.choice(
                list(range(len(self.neighborhoods))),
                size=len(self.neighborhoods),
                replace=True
            )
            sampled_inst = [self.neighborhoods[i] for i in sampled_inst_ix]
            sampled_labels = [self.labels[i] for i in sampled_inst_ix]
        else:
            sampled_inst = self.neighborhoods
            sampled_labels = self.labels

        # Create a MINDWALCTree, fit it and add to self.estimators_
        tree = MINDWALCTree(self.path_max_depth, self.min_samples_leaf, 
                       self.progress, self.max_tree_depth, self.n_jobs,
                       init=False, fixed_walc_depth=self.fixed_walc_depth, path_min_depth=self.path_min_depth)
        tree.tree_ = tree._build_tree(sampled_inst, sampled_labels, 
                                      vertex_sample=self.vertex_sample,
                                      useless=self.useless)
        return tree

    def fit(self, kg, instances, labels):
        
        super().fit(kg, instances, labels)
        useless = self._prune_useless(self.neighborhoods, labels)

        self.labels = labels
        self.useless = useless

        if self.progress is not None and self.n_jobs == 1:
            estimator_iterator = self.progress(range(self.n_estimators), 
                                               desc='estimator loop', 
                                               leave=True)
        else:
            estimator_iterator = range(self.n_estimators)

        self.estimators_ = []
        for _ in estimator_iterator:
            self.estimators_.append(self._create_estimator())

    def predict(self, kg, instances):
        if self.progress is not None:
            inst_iterator = self.progress(instances, 
                                          desc='Extracting neighborhoods')
        else:
            inst_iterator = instances

        neighborhoods = []
        for inst in inst_iterator:
            neighborhood = kg.extract_neighborhood(inst, depth=self.path_max_depth)
            neighborhoods.append(neighborhood)

        predictions = []
        for neighborhood in neighborhoods:
            inst_preds = []
            for tree in self.estimators_:
                inst_preds.append(tree.tree_.evaluate(neighborhood))
            predictions.append(Counter(inst_preds).most_common()[0][0])
        return predictions

class MINDWALCTransform(BaseEstimator, TransformerMixin, MINDWALCMixin):
    def __init__(self, path_max_depth=8, progress=None, n_jobs=1, 
                 n_features=1, fixed_walc_depth=True, path_min_depth=path_min_depth_default):
        super().__init__(path_max_depth, progress, n_jobs, fixed_walc_depth=fixed_walc_depth, path_min_depth=path_min_depth)
        self.n_features = n_features

    def fit(self, kg, instances, labels):
        if self.progress is not None:
            inst_iterator = self.progress(instances, 
                                          desc='Extracting neighborhoods')
        else:
            inst_iterator = instances

        neighborhoods = []
        for inst in inst_iterator:
            neighborhood = kg.extract_neighborhood(inst, depth=self.path_max_depth)
            neighborhoods.append(neighborhood)

        prior_entropy = entropy(np.unique(labels, return_counts=True)[1])

        cache = {}

        self.walks_ = set()

        if len(np.unique(labels)) > 2:
            _classes = np.unique(labels)
        else:
            _classes = [labels[0]]

        for _class in _classes:
            label_map = {}
            for lab in np.unique(labels):
                if lab == _class:
                    label_map[lab] = 1
                else:
                    label_map[lab] = 0

            new_labels = list(map(lambda x: label_map[x], labels))

            walks = self._mine_walks(neighborhoods, new_labels,
                                     n_walks=self.n_features, fixed_walc_depth=self.fixed_walc_depth)

            prev_len = len(self.walks_)
            n_walks = min(self.n_features // len(np.unique(labels)), len(walks))
            for _, walk in sorted(walks, key=lambda x: x[0], reverse=True):
                if len(self.walks_) - prev_len >= n_walks:
                    break

                if walk not in self.walks_:
                    self.walks_.add(walk)

    def transform(self, kg, instances):
        if self.progress is not None:
            inst_iterator = self.progress(instances, 
                                          desc='Extracting neighborhoods')
        else:
            inst_iterator = instances

        neighborhoods = []
        for inst in inst_iterator:
            neighborhood = kg.extract_neighborhood(inst, depth=self.path_max_depth)
            neighborhoods.append(neighborhood)

        features = np.zeros((len(instances), self.n_features))
        for i, neighborhood in enumerate(neighborhoods):
            for j, (vertex, depth) in enumerate(self.walks_):
                features[i, j] = neighborhood.find_walk(vertex, depth)
        return features