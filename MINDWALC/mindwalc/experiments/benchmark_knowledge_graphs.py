import os.path

import rdflib
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import sys
sys.path.append('..')
from MINDWALC.mindwalc.tree_builder import MINDWALCTree, MINDWALCForest, MINDWALCTransform
from MINDWALC.mindwalc.datastructures import *
import time

from itertools import product
from collections import defaultdict

import pickle

import warnings; warnings.filterwarnings('ignore')

import json

def train_model(g, train_file, test_file, entity_col, label_col, label_predicates, output, relation_tail_merging=False, fixed_walc_depths=[True, False, None], path_max_depth=8):

    n_jobs = -1

    # Create some lists of train and test entities & labels
    train_data = pd.read_csv(train_file, sep='\t')
    test_data = pd.read_csv(test_file, sep='\t')

    train_entities = [rdflib.URIRef(x) for x in train_data[entity_col]]
    train_labels = train_data[label_col]

    test_entities = [rdflib.URIRef(x) for x in test_data[entity_col]]
    test_labels = test_data[label_col]

    results = {}
    results["ground_truth"] = list(test_labels)

    # Convert the rdflib graph to our graph
    start = time.time()
    kg = Graph.rdflib_to_graph(g, label_predicates=label_predicates, relation_tail_merging=relation_tail_merging, skip_literals=False)
    results['graph_convertion_time'] = time.time() - start
    #kg_rtm = Graph.rdflib_to_graph(g, label_predicates=label_predicates, relation_tail_merging=True)

    walc_strategies_str = {True: "fix", False: "flex", None: "both"}  # ['fix', 'flex', 'both']

    for i, fixed_walc_depth in enumerate(fixed_walc_depths):
        walc_mode = walc_strategies_str[fixed_walc_depth]

        transf = MINDWALCTransform(path_max_depth=path_max_depth, n_features=10000, n_jobs=n_jobs, fixed_walc_depth=fixed_walc_depth)
        start = time.time()
        transf.fit(kg, train_entities, train_labels)
        transf_fit_time = time.time() - start

        train_features = transf.transform(kg, train_entities)
        test_features = transf.transform(kg, test_entities)

        useful_features = np.sum(train_features, axis=0) > 1

        train_features = train_features[:, useful_features]
        test_features = test_features[:, useful_features]

        clf = GridSearchCV(RandomForestClassifier(random_state=42, max_features=None),
                   {'n_estimators': [10, 100, 250], 'max_depth': [5, 10, None]})
        start = time.time()
        clf.fit(train_features, train_labels)
        transf_rf_time = time.time() - start
        transf_rf_preds = clf.predict(test_features)

        print(f'[Transform + Random Forest & walk mode {walc_mode}] Test accuracy = {accuracy_score(test_labels, transf_rf_preds)} || Confusion Matrix:')
        print(confusion_matrix(test_labels, transf_rf_preds))

        clf = GridSearchCV(LogisticRegression(random_state=42, penalty='l2'),
                    {'C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]})
        start = time.time()
        clf.fit(train_features, train_labels)
        transf_lr_time = time.time() - start
        transf_lr_preds = clf.predict(test_features)

        print(f'[Transform + Logistic Regression & walk mode {walc_mode}] Test accuracy = {accuracy_score(test_labels, transf_lr_preds)} || Confusion Matrix:')
        print(confusion_matrix(test_labels, transf_lr_preds))

        results['transform_fit_time_' + walc_mode] = transf_fit_time
        results['transform_lr_preds_' + walc_mode] = transf_lr_preds
        results['transform_rf_preds_' + walc_mode] = list(transf_rf_preds)
        results['transf_rf_time_' + walc_mode] = transf_rf_time + transf_fit_time
        results['transf_lr_time_' + walc_mode] = transf_lr_time + transf_fit_time

    N_SPLITS = 5

    # tune forest and tree:
    for i, fixed_walc_depth in enumerate(fixed_walc_depths):
        print(f"Forest and Tree tuning {i+1}/{len(fixed_walc_depths)}")
        for i_try in range(5): # ray can crash some times, therefore i added this re-trying loop
            try:
                walc_mode = walc_strategies_str[fixed_walc_depth]

                params = {
                    'max_tree_depth': [5, None],
                    'vertex_sample': [0.5, 0.9]
                }

                best_params, best_score = None, (0, 0)
                combinations = list(itertools.product(*list(params.values())))
                for combination in combinations:
                    param_combination = dict(zip(params.keys(), combination))
                    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
                    accuracies = defaultdict(list)
                    for train_ix, test_ix in cv.split(train_entities, train_labels):
                        cv_train_entities = [train_entities[ix] for ix in train_ix]
                        cv_train_labels = [train_labels[ix] for ix in train_ix]
                        cv_test_entities = [train_entities[ix] for ix in test_ix]
                        cv_test_labels = [train_labels[ix] for ix in test_ix]

                        clf = MINDWALCForest(path_max_depth=path_max_depth, n_jobs=n_jobs, n_estimators=50, fixed_walc_depth=fixed_walc_depth,
                                        **param_combination)
                        clf.fit(kg, cv_train_entities, cv_train_labels)

                        for n_estimators in [10, 25, 50]:
                            clf_dummy = MINDWALCForest(path_max_depth=path_max_depth, fixed_walc_depth=fixed_walc_depth)
                            clf_dummy.estimators_ = clf.estimators_[:n_estimators]
                            preds = clf_dummy.predict(kg, cv_test_entities)

                            accuracies[n_estimators].append(accuracy_score(cv_test_labels, preds))

                    for n_estimators in [10, 25, 50]:
                        avg_acc = np.mean(accuracies[n_estimators])
                        std_acc = np.std(accuracies[n_estimators])

                        if (avg_acc, -std_acc) > best_score:
                            best_score = (avg_acc, -std_acc)
                            param_combination['n_estimators'] = n_estimators
                            best_params = param_combination


                print('Tuned Forest params = {}'.format(best_params))

                # Fit using the tuned parameters
                clf = MINDWALCForest(path_max_depth=path_max_depth, n_jobs=n_jobs, fixed_walc_depth=fixed_walc_depth, **best_params)

                start = time.time()
                clf.fit(kg, train_entities, train_labels)
                forest_fit_time = time.time() - start

                preds = clf.predict(kg, test_entities)

                print(f'[Forest, {walc_mode} walk] Test accuracy = {accuracy_score(test_labels, preds)} || Confusion Matrix:')
                print(confusion_matrix(test_labels, preds))

                preds = list(preds)

                results['forest_params_' + walc_mode] = best_params
                results['forest_fit_time_' + walc_mode] = forest_fit_time
                results['forest_preds_' + walc_mode] = preds

                # Tune the max_tree_depth
                best_depth, best_score = None, (0, 0)
                for depth in [3, 5, 10, None]:
                    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
                    accuracies = []
                    for train_ix, test_ix in cv.split(train_entities, train_labels):
                        cv_train_entities = [train_entities[ix] for ix in train_ix]
                        cv_train_labels = [train_labels[ix] for ix in train_ix]
                        cv_test_entities = [train_entities[ix] for ix in test_ix]
                        cv_test_labels = [train_labels[ix] for ix in test_ix]

                        clf = MINDWALCTree(path_max_depth=path_max_depth, max_tree_depth=depth, n_jobs=n_jobs, fixed_walc_depth=fixed_walc_depth)
                        clf.fit(kg, cv_train_entities, cv_train_labels)
                        preds = clf.predict(kg, cv_test_entities)

                        accuracies.append(accuracy_score(cv_test_labels, preds))

                        ub_accuracies = accuracies + [1.0] * (N_SPLITS - len(accuracies))
                        if np.mean(ub_accuracies) < best_score[0]:
                            break

                    avg_acc = np.mean(accuracies)
                    std_acc = np.std(accuracies)

                    if (avg_acc, -std_acc) > best_score:
                        best_score = (avg_acc, -std_acc)
                        best_depth = depth

                print('Tuned Tree depth = {}'.format(best_depth))

                # Fit using the tuned depth
                clf = MINDWALCTree(path_max_depth=path_max_depth, max_tree_depth=best_depth, min_samples_leaf=1, n_jobs=n_jobs, fixed_walc_depth=fixed_walc_depth)

                start = time.time()
                clf.fit(kg, train_entities, train_labels)
                tree_fit_time = time.time() - start

                preds = clf.predict(kg, test_entities)

                print(f'[Tree, {walc_mode} walk] Test accuracy = {accuracy_score(test_labels, preds)} || Confusion Matrix:')
                print(confusion_matrix(test_labels, preds))

                preds = list(preds)

                results['tree_depth_' + walc_mode] = best_depth
                results['tree_fit_time_' + walc_mode] = tree_fit_time
                results['tree_preds_' + walc_mode] = preds

                break # break out of the try loop
            except Exception as e:
                if i_try == 4:
                    raise e
                print(f"error during forest and tree tuning, walk mode {walc_mode}: {e}")
                print("retrying...")

    # convert each None to 'None' for json serialization
    for k in results.keys():
        vals = results[k]
        #print(type(vals))
        if isinstance(vals, list):
            for i, v in enumerate(vals):
                if v is None:
                    results[k][i] = 'None'
        elif vals is None:
            results[k] = 'None'
        elif isinstance(vals, np.ndarray):
            results[k] = vals.tolist()
        elif isinstance(vals, dict):
            for k2 in vals.keys():
                #print(type(vals[k2]))
                if vals[k2] is None:
                    results[k][k2] = 'None'

    #print(results)

    output_file = f'{output}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f)

    #pickle.dump(results, open(output_file, 'wb+'))

###################### BGS #####################################
rdf_file = 'MINDWALC/mindwalc/data/BGS/completeDataset.nt'
format = 'nt'
train_file = 'MINDWALC/mindwalc/data/BGS/trainingSet(lith).tsv'
test_file = 'MINDWALC/mindwalc/data/BGS/testSet(lith).tsv'
entity_col = 'rock'
label_col = 'label_lithogenesis'
label_predicates = [
    rdflib.term.URIRef('http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis'),
    rdflib.term.URIRef('http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesisDescription'),
    rdflib.term.URIRef('http://data.bgs.ac.uk/ref/Lexicon/hasTheme')
]
out_folder = "bgs4"

###################### AM ######################################
'''rdf_file = 'MINDWALC/mindwalc/data/AM/rdf_am-data.ttl'
format = 'turtle'
train_file = 'MINDWALC/mindwalc/data/AM/trainingSet.tsv'
test_file = 'MINDWALC/mindwalc/data/AM/testSet.tsv'
entity_col = 'proxy'
label_col = 'label_cateogory'
label_predicates = [
   rdflib.term.URIRef('http://purl.org/collections/nl/am/objectCategory'),
   rdflib.term.URIRef('http://purl.org/collections/nl/am/material')
]
out_folder = "am"'''

##################### AIFB #####################################
'''
rdf_file = 'MINDWALC/mindwalc/data/AIFB/aifb.n3'
format = 'n3'
train_file = 'MINDWALC/mindwalc/data/AIFB/AIFB_train.tsv'
test_file = 'MINDWALC/mindwalc/data/AIFB/AIFB_test.tsv'
entity_col = 'person'
label_col = 'label_affiliation'
label_predicates = [
        rdflib.URIRef('http://swrc.ontoware.org/ontology#affiliation'),
        rdflib.URIRef('http://swrc.ontoware.org/ontology#employs'),
        rdflib.URIRef('http://swrc.ontoware.org/ontology#carriedOutBy')
]
out_folder = "aifb3"'''

##################### MUTAG ####################################
'''rdf_file = 'MINDWALC/mindwalc/data/MUTAG/carcinogenesis.owl'
format = None
train_file = 'MINDWALC/mindwalc/data/MUTAG/trainingSet.tsv'
test_file = 'MINDWALC/mindwalc/data/MUTAG/testSet.tsv'
entity_col = 'bond'
label_col = 'label_mutagenic'
label_predicates = [
    rdflib.term.URIRef('http://dl-learner.org/carcinogenesis#isMutagenic')
]
out_folder = "mutag4"'''


# Load in our graph using rdflib
print(end='Loading data... ', flush=True)
g = rdflib.Graph()
if format is not None:
    g.parse(rdf_file, format=format)
else:
    g.parse(rdf_file)
print('OK')

import time
outputs = []
for with_rtm in [True, False]:
    for i in range(0, 10):
        output = f'MINDWALC/mindwalc/experiments/{out_folder}/res{"_rtm" if with_rtm else ""}_{i}'
        if os.path.exists(f'{output}.json'):
            print(f"Skipping {output}.json")
        else:
            outputs.append((output, with_rtm))

for output, with_rtm in tqdm(outputs):
    print(f"Computing {output}.json")
    for i_try in range(5):
        try:
            train_model(g, train_file, test_file, entity_col, label_col, label_predicates, output, relation_tail_merging=with_rtm, fixed_walc_depths=[True, False, None]) # [True, False, None] [True]
            break
        except Exception as e:
            print(f"Error while processing {output}: {e}")
            if i_try == 4:
                print("giving up...")
                raise e
            print("retrying...")
            time.sleep(1)




