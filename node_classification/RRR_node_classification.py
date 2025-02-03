from warnings import warn
import sys, os
from graph_processing.neo4j2rdf import cypher_to_rdf
from utils.filesystem import create_new_result_folder_in, make_folder
import json
from tqdm import tqdm
import rdflib
from MINDWALC.mindwalc.datastructures import Graph
from MINDWALC.mindwalc.tree_builder import MINDWALCTree, MINDWALCForest, MINDWALCTransform
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    cohen_kappa_score
import numpy as np
import pandas as pd
from utils.decision_tree_visualisation import tree_visualisation_postprocessor
from sklearn import tree as sktree
import graphviz
from neo4j import GraphDatabase
from sys import argv
import yaml

''' 
### What this script does: ###
- Selects a subgraph from a running neo4j graph database e.g. by filtering out some node- and relation-types.
- Then we can step wise destroy the "Instance knowledge" by randomly removing the relations between instance-nodes and its neighbors. 
- We call this process Random Relation Removement (RRR) and we apply it step wise to destroy more and more percentage (e.g. [0%, 10%, 20%, ...90%]).
- On each RRR-level, we train different configured mindwalc classifiers and evaluate them with k-fold cross validation.
- There are some other scripts which can be used to generate tables and plots from the results (e.g. plot_RRR_curves.py).
'''


############## functions #################
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
def get_splits_for_cross_val(features, labels, fold_amount=10, stratified=True):

    # convert features and labels to dataframe:
    dataset = pd.DataFrame({"feature": features, "label": labels})

    if stratified:
        skf = StratifiedKFold(n_splits=fold_amount, random_state=None, shuffle=False)
        for train_index_list, test_index_list in skf.split(dataset['feature'], dataset['label']):
            yield dataset.iloc[train_index_list], dataset.iloc[test_index_list]
    else:
        folds = KFold(n_splits=fold_amount, shuffle=False)
        for train_index_list, test_index_list in folds.split(list(range(len(dataset)))):
            yield dataset.iloc[train_index_list], dataset.iloc[test_index_list]



############# main ####################
def main():
    # load config file with yaml:
    try:
        config_file_path = argv[2]
    except IndexError:
        raise ValueError("Please provide the config file path as second argument.")

    with open(config_file_path, 'r') as f:
        subgraph_generation_config = yaml.safe_load(f)

    print(f"Using config file: {config_file_path}")

    ########### subgraph selection params ############
    node_instance_type = subgraph_generation_config['node_instance_type']
    concepts_to_disconnect = subgraph_generation_config['concepts_to_disconnect']
    relations_to_disconnect = subgraph_generation_config['relations_to_disconnect']
    node_types_to_consider = subgraph_generation_config['node_types_to_consider']
    instance_neighborhood_extraction_size = subgraph_generation_config['instance_neighborhood_extraction_size'] if 'instance_neighborhood_extraction_size' in subgraph_generation_config.keys() else None
    graph_preprocessing_queries = subgraph_generation_config['graph_preprocessing_queries'] if 'graph_preprocessing_queries' in subgraph_generation_config.keys() else []

    ########### random relation removement and graph params ###########
    relation_types_not_allowed_to_delete = subgraph_generation_config['relation_types_not_allowed_to_delete']
    random_relation_removements = subgraph_generation_config['random_relation_removements']
    rdf_predicates_to_filter_out = [rdflib.URIRef(uri) for uri in
                                    subgraph_generation_config['rdf_predicates_to_filter_out']]
    skip_nodes_with_prefix = subgraph_generation_config['skip_nodes_with_prefix'] if 'skip_nodes_with_prefix' in subgraph_generation_config.keys() else []

    ########### tree params ###########
    tree_vis_depth_offset = subgraph_generation_config['tree_vis_depth_offset']
    tree_vis_depth_factor = subgraph_generation_config['tree_vis_depth_factor']
    gv_file_prefix = subgraph_generation_config['gv_file_prefix']
    subgraph_name = subgraph_generation_config['subgraph_name']
    store_all_trees = subgraph_generation_config['store_all_trees']
    post_prune = subgraph_generation_config['post_prune']

    ########### cross validation params ###########
    overwrite_output_files = subgraph_generation_config['overwrite_output_files']
    n_jobs = subgraph_generation_config['n_jobs']
    fold_amount = subgraph_generation_config['fold_amount']

    ############ mindwalc params: ###########
    path_max_depths = subgraph_generation_config['path_max_depths']
    path_min_depths = subgraph_generation_config['path_min_depths']
    max_tree_depth = subgraph_generation_config['max_tree_depth']
    min_samples_leaf = subgraph_generation_config['min_samples_leaf']
    use_forests = subgraph_generation_config['use_forests']
    forest_size = subgraph_generation_config['forest_size']
    fixed_walking_depths = subgraph_generation_config['fixed_walking_depths']
    use_sklearn = subgraph_generation_config['use_sklearn']
    relation_tail_merging = subgraph_generation_config['relation_tail_merging']
    mute_node_properties = subgraph_generation_config['mute_node_properties']

    ##### other params #######
    base_path = subgraph_generation_config['base_path']

    ########## Pokemon labeling params #####
    # for tree of life tree:
    label_name_to_getter_query = subgraph_generation_config['label_name_to_getter_query']

    # connect to neo4j dbms:
    gdb_adress = 'bolt://localhost:7687'
    try:
        pw = argv[1]
    except IndexError:
        raise ValueError("Please provide the password as first argument.")
    auth = ('neo4j', pw)
    driver = GraphDatabase.driver(gdb_adress, auth=auth)
    session = driver.session(database="neo4j")

    if graph_preprocessing_queries:
        print(f"initializing subgraph generation with config: {subgraph_generation_config}")
        for query in graph_preprocessing_queries:
            session.run(query)

    def get_names_of_neo4jrdf_node_ids(entity_list):
        print(f"entity_list: {entity_list}")
        neo4j_node_ids = [entity_string.replace(gv_file_prefix, '') for entity_string in entity_list]
        assert len(list(set(neo4j_node_ids))) == len(neo4j_node_ids), "ERROR: duplicate node ids in entity_list!"
        q = f"match (n) where ID(n) in {neo4j_node_ids} return n.name as name".replace("'", '')
        print(f"q: {q}")
        names = [r['name'] for r in session.run(q)]
        if len(names) != len(entity_list) or len(names) == 0:
            raise ValueError(f"Could only find {len(names)} names for all {len(entity_list)} entities with ids: {entity_list}")
        return names


    #random_relation_removements = [round(x, 2) for x in np.linspace(rrr_start, rrr_max, step_count)]

    print(f"random_relation_removements: {random_relation_removements}")

    result_path_root = create_new_result_folder_in(base_path, overwrite_output_files,
                                                   f'{subgraph_name}_')

    print(f"saving results to {result_path_root}")

    # store the config file:
    with open(result_path_root + "/config.yaml", 'w') as f:
        f.write(yaml.dump(subgraph_generation_config))

    mean_values = {}

    for random_relation_removement in tqdm(random_relation_removements): #####for each random relation removement value

        # reset last selection:
        session.run(f"match (n) remove n.{subgraph_name}")
        session.run(f"match ()-[r]-() remove r.{subgraph_name}")

        # select the instances and the instance knowledge:
        for node_type_to_consider in node_types_to_consider:
            session.run(
                f"match (n:{node_instance_type})-[r]-(t:{node_type_to_consider}) set n.{subgraph_name} = true, r.{subgraph_name} = true")

        if not instance_neighborhood_extraction_size: # select subgraph only based on node-types:
            for node_type_to_consider in node_types_to_consider:
                session.run(f"match (n:{node_type_to_consider}) set n.{subgraph_name} = true")
            # select all relations between selected nodes:
            session.run(
                    f"match (a)-[r]-(b) where a.{subgraph_name} and b.{subgraph_name} set r.{subgraph_name} = true")

        else: # select the subgraph based on the neighborhood of the instances and allowed node-types:
            print(f"Adding neighborhood r={instance_neighborhood_extraction_size} to the subgraph...")
            instance_node_ids = [r["ID(n)"] for r in session.run(f"match (n:{node_instance_type}) return ID(n)")]
            selected_nodes = set()
            selected_relations = set()
            for instance_node_id in instance_node_ids:
                query = f"MATCH (a:{node_instance_type})-[r*1..{instance_neighborhood_extraction_size}]-(b) " \
                        f"WHERE ID(a) = {instance_node_id} " \
                        f"RETURN r"
                results = [r['r'] for r in session.run(query)]
                for relations in results:
                    for relation in relations:
                        # selected_nodes.add(result["ID(a)"])
                        selected_nodes.add(relation.start_node.element_id)
                        selected_nodes.add(relation.end_node.element_id)
                        selected_relations.add(relation.element_id)

            selected_nodes = list(selected_nodes)
            selected_relations = list(selected_relations)
            chunk_size = 100
            selected_nodes_chunks = [selected_nodes[i:i + chunk_size] for i in range(0, len(selected_nodes), chunk_size)]
            selected_relations_chunks = [selected_relations[i:i + chunk_size] for i in range(0, len(selected_relations), chunk_size)]
            for node_ids in selected_nodes_chunks:
                node_ids = str(node_ids).replace("'", '')
                allowed_node_types_str = '( ' + ' or '.join([f"n:{t}" for t in node_types_to_consider]) + ' )'
                session.run(f"match (n) where {allowed_node_types_str} and ID(n) in {node_ids} set n.{subgraph_name} = true")
            for relation_ids in selected_relations_chunks:
                relation_ids = str(relation_ids).replace("'", '')
                session.run(f"match ()-[r]-(b) where ID(r) in {relation_ids} set r.{subgraph_name} = true")



        # remove specific concepts from selection:
        for node_type_to_consider in node_types_to_consider:
            for concept_id in concepts_to_disconnect:
                session.run(f"match (n:{node_type_to_consider}) where ID(n) = {concept_id} set n.{subgraph_name} = false")

        # remove specific relation-types from selection:
        for relation_type in relations_to_disconnect:
            session.run(f"match (a)-[r:{relation_type}]-(b) set r.{subgraph_name} = false")

        #create_report_subgraph(...)
        # remove random relations:
        if random_relation_removement > 0.0:

            relations_to_delete = []
            q = (f"match (a:{node_instance_type})-[r]->(b) where "
                 f"a.{subgraph_name} and r.{subgraph_name} and b.{subgraph_name} "
                 f"return id(a), id(r)")
            relations_of_instances = {}
            for id_a, id_r in [(r["id(a)"], r["id(r)"]) for r in session.run(q)]:  # for each node instance:
                if id_a not in relations_of_instances.keys():
                    relations_of_instances[id_a] = set()
                relations_of_instances[id_a].add(id_r)
            for id_a in relations_of_instances.keys():
                relations = list(relations_of_instances[id_a])
                np.random.shuffle(relations)
                # now remove random_relation_removement from relations list:
                relations_to_delete += relations[:int(len(relations) * random_relation_removement)]
            session.run(f"match ()-[r]-() where ID(r) in {relations_to_delete} remove r.{subgraph_name}")

            # now check if the applied relation removements did produce lonely instance-nodes:
            removed_report_nodes = 0
            q = f"match (a:{node_instance_type}) where a.{subgraph_name} return id(a) as id"
            for id in [r["id"] for r in session.run(q)]: # for each node instance:
                q = (f"match (a)-[r]-(b) where id(a) = {id} and r.{subgraph_name} and b.{subgraph_name} "
                     f"and not (b:{node_instance_type}) return id(b) as id")
                amount_neighbors = len([i for i in session.run(q)])
                if amount_neighbors == 0: # is it a lonely node?
                    # This node-instance is lonely, so completely disconnected from the graph. So we should exclude it from the classification task:
                    session.run(f"match (a) where id(a) = {id} remove a.{subgraph_name}")
                    removed_report_nodes += 1

            print(f"\nRemoved {removed_report_nodes} featureless report nodes from subgraph {subgraph_name} "
                  f"after removing {round(random_relation_removement*100,1)}% relations.", flush=True)

        # adding subgraph size to subgraph metha:
        r = session.run(f"match (c) where c.{subgraph_name} return count(c) as amount")
        subgraph_node_size = [a['amount'] for a in r]
        if subgraph_node_size == 0:
            warn(f"Subgraph {subgraph_name} has node-size 0!")
        session.run(
            f"match (n:{subgraph_name}) where n.subgraph_name = '{subgraph_name}' set n.subgraph_node_size = {subgraph_node_size}")

        print(f"Subgraph {subgraph_name} has {subgraph_node_size} nodes.")

        ########### load and assemble train data from graph database ############
        train_data = {'node_id': [], 'label': []}
        if label_name_to_getter_query:
            session.run(f'match (a:{node_instance_type}) remove a.label')

            for label_name in label_name_to_getter_query.keys():
                session.run(label_name_to_getter_query[label_name] + f" and n.{subgraph_name}" +
                                            f' set n.label = "{label_name}"')

            for res in session.run(f'match (a:{node_instance_type}) return a.label as label, ID(a) as node_id'):
                if not res['label']:
                    continue
                train_data['label'].append(str(res['label']))
                train_data['node_id'].append(str(res['node_id']))

        else:  # dynamicly create labels = diagnoses
            raise NotImplementedError("dynamically label reports currently not implemented...")

        result_path_RRR = create_new_result_folder_in(result_path_root, True,
                                                       f'RRR_{random_relation_removement}')

        ##### save train_data.csv
        train_data_path = f'{result_path_RRR}/dataset{round(random_relation_removement,2)}.csv'
        if len(list(set(train_data['node_id']))) != len(train_data['node_id']):
            print(f'WARNING: some reports appear more than once in test set!')
        df = pd.DataFrame(train_data)
        df.to_csv(train_data_path)

        ###### convert subgraph to rdf file, save it and load it as Graph object: #####
        print(f"converting selected subgraph to rdf file...")
        addr = gdb_adress.replace(':7687', '').replace('bolt://', '')
        rdf_subgraph_file = f'{result_path_RRR}/subgraph.n3'
        report_subgraph_query = f'match (a)-[r]->(b) where a.{subgraph_name} and b.{subgraph_name} and r.{subgraph_name} return *'
        cypher_to_rdf(report_subgraph_query, rdf_subgraph_file, addr, auth)
        g = rdflib.Graph()
        g.parse(rdf_subgraph_file, format='text/n3')
        kg_non_rtm = Graph.rdflib_to_graph(g, relation_tail_merging=False, label_predicates=rdf_predicates_to_filter_out,
                                           skip_literals=mute_node_properties, skip_nodes_with_prefix=skip_nodes_with_prefix)
        kg_rtm = Graph.rdflib_to_graph(g, relation_tail_merging=True, label_predicates=rdf_predicates_to_filter_out,
                                       skip_literals=mute_node_properties, skip_nodes_with_prefix=skip_nodes_with_prefix)

        # store kg_rtm and kg_non_rtm as .gv files as pickled objects:
        '''import pickle
        with open(f'{result_path_RRR}/kg_rtm.pkl', 'wb') as f:
            pickle.dump(kg_rtm, f)
        with open(f'{result_path_RRR}/kg_non_rtm.pkl', 'wb') as f:
            pickle.dump(kg_non_rtm, f)
        print(f"storing rtm graph")
        kg_rtm.graph_to_rdf(f'{result_path_RRR}/subgraph_rtm.n3')
        print(f"storing input-graph")
        kg_non_rtm.graph_to_rdf(f'{result_path_RRR}/subgraph_non_rtm.n3')'''

        # create and clean training data: Remove each report which does not appear in our graph:
        traintest_ents = []
        traintest_labels = []
        ents_sorted_out = []
        for i, entity in enumerate([f'{gv_file_prefix}{id}' for id in train_data['node_id']]):
            if entity in kg_non_rtm.name_to_vertex.keys():
                traintest_ents.append(rdflib.URIRef(entity))
                traintest_labels.append(str(train_data['label'][i]))
            else:
                ents_sorted_out.append(entity)
        if len(ents_sorted_out) > 0:
            print(f"WARNING: {len(ents_sorted_out)} training data points were sorted out, "
                  f"because they do not appear in the knowledge graph '{rdf_subgraph_file}'.")
            print(f"List of sorted out ents: {ents_sorted_out}")



        if len(traintest_ents) == 0:
            warn(f"ERROR: Could not find any training data points in the knowledge graph '{rdf_subgraph_file}' "
                 f"using querie:\n{label_name_to_getter_query}")
            continue
        else:
            print(f"Found {len(traintest_ents)} training data points.", flush=True)

        ########### get some statistics and save them ###########
        labels_set = set(train_data["label"])

        alphabet_labels = list(labels_set)
        label_to_node_list = {}
        for label in alphabet_labels:
            label_to_node_list[label] = []
            for i, node in enumerate(train_data['node_id']):
                if train_data['label'][i] == label:
                    label_to_node_list[label].append(node)

        meta_info = "==== Train Dataset Info ===\n\n"
        meta_info += f"Subgraph: {subgraph_name}\n"
        meta_info += f'\nFound {len(train_data["node_id"])} nodes to classify which ' \
                     f'contain {len(list(labels_set))} different classes:\n'
        for i, label in enumerate(alphabet_labels):
            meta_info += f'{i})\t{len(label_to_node_list[label])}\tx\t{label}\n'
        meta_info += "\n"
        for label in alphabet_labels:
            meta_info += f'Reports labeled with "{label}":\n'
            for node in get_names_of_neo4jrdf_node_ids(label_to_node_list[label]):
                meta_info += f'{node}\n'
            meta_info += '\n'
        meta_info += '\n'
        meta_info += f"{len(ents_sorted_out)} instances have been sorted out due to missing connections to given (knowledge-)graph:\n"
        for ent in get_names_of_neo4jrdf_node_ids(ents_sorted_out):
            meta_info += f"{ent}\n"
        with open(f'{result_path_RRR}/dataset_info.txt', 'w') as f:
            f.write(meta_info)

        # generate cross validation dataset:
        if not fold_amount is None:
            cross_val_data_set = [(x[0], x[1]) for x in get_splits_for_cross_val(traintest_ents, traintest_labels,
                                 fold_amount=fold_amount,
                                 stratified=True)]
        else:
            cross_val_data_set = [(pd.DataFrame({"feature": traintest_ents, "label": traintest_labels}),
                                   pd.DataFrame({"feature": traintest_ents, "label": traintest_labels}))]
            print("WARNING: Cross-validation is disabled. training and testing will be performed on the same data.")

        for setting_id in range(len(path_max_depths)): #################### for each setting:
            path_max_depth = path_max_depths[setting_id]
            use_forest = use_forests[setting_id]
            n_estimators = forest_size[setting_id]
            fixed_walking_depth = fixed_walking_depths[setting_id]
            path_min_depth = path_min_depths[setting_id]
            relation_tail_merge = relation_tail_merging[setting_id]

            kg = kg_rtm if relation_tail_merge else kg_non_rtm

            tree_config_string = (f'{"Fix" if fixed_walking_depth == True  else "Flex" if fixed_walking_depth is not None else "Comb" }'
                         f'WalcDepth{path_min_depth}-{path_max_depth}_'
                         f'{"RF" + str(n_estimators) if use_forest else "DT"}{"_SKL" if use_sklearn[setting_id] else ""}'
                                  f'{"_RTM" if relation_tail_merging[setting_id] else ""}')

            result_path = create_new_result_folder_in(result_path_RRR, True,
                                                      tree_config_string)
            if tree_config_string not in list(mean_values.keys()):
                mean_values[tree_config_string] = {
                    "x": [],

                    "f1_mean": [],
                    "f1_std": [],
                    "cohen_cappa_mean": [],
                    "cohen_cappa_std": [],
                    "accuracy_mean": [],
                    "accuracy_std": [],
                    "precision_mean": [],
                    "precision_std": [],

                    "f1_mean_train": [],
                    "f1_std_train": [],
                    "cohen_cappa_mean_train": [],
                    "cohen_cappa_std_train": [],
                    "accuracy_mean_train": [],
                    "accuracy_std_train": [],
                    "precision_mean_train": [],
                    "precision_std_train": [],

                    "node_count_mean": [],
                    "max_tree_depth_mean": [],
                    "node_count_std": [],
                    "max_tree_depth_std": [],

                    "node_count": [],
                    "relation_count": [],
                }

            cross_val_results = {
                "f1": [],
                "cohen_cappa": [],
                "accuracy": [],
                "precision": [],
                "node_count": [],
                "max_tree_depth": [],

                "f1_train": [],
                "cohen_cappa_train": [],
                "accuracy_train": [],
                "precision_train": [],
            }

            average_setting = "weighted"
            os.mkdir(result_path + "/trees")
            os.mkdir(result_path + "/datasets")
            for i_crossval, (train_dataset, test_dataset) in enumerate(cross_val_data_set):

                if use_sklearn[setting_id]:

                    # collect strongest features:
                    transf = MINDWALCTransform(path_max_depth=8, n_features=1000, n_jobs=n_jobs,
                                               fixed_walc_depth=fixed_walking_depth, path_min_depth=path_min_depth)
                    '''transf = MINDWALCTransform(path_max_depth=path_max_depth, max_tree_depth=max_tree_depth,
                                       min_samples_leaf=min_samples_leaf, fixed_walc_depth=fixed_walking_depth,
                                       path_min_depth=path_min_depth, n_jobs=n_jobs)'''
                    transf.fit(kg, list(train_dataset['feature']), list(train_dataset['label']))
                    walk_candidates = transf.walks_

                    # get binary feature vecs (transform):
                    x_train_binary = transf.transform(kg, list(train_dataset['feature']))

                    # todo: put this logic into transf.transform and push to MINDWALC lib:
                    # adjust x_train_binary to same length as walk_candidates:
                    if len(walk_candidates) < x_train_binary.shape[1]:
                        x_train_binary = x_train_binary[:, :len(walk_candidates)]

                    # train sklearn tree:
                    label_name_to_int = {label: i for i, label in enumerate(set(train_dataset['label']))}
                    int_to_label_name = {i: label for i, label in enumerate(set(train_dataset['label']))}
                    clf_sk_tree = sktree.DecisionTreeClassifier()
                    y_train_numeric = [label_name_to_int[yi] for yi in list(train_dataset['label'])]
                    clf_sk_tree.fit(x_train_binary, y_train_numeric)

                    # predict:
                    x_test_binary = transf.transform(kg, list(test_dataset['feature']))
                    if len(walk_candidates) < x_test_binary.shape[1]:
                        x_test_binary = x_test_binary[:, :len(walk_candidates)]
                    preds = [int_to_label_name[i] for i in clf_sk_tree.predict(x_test_binary)]
                    preds_train = [int_to_label_name[i] for i in clf_sk_tree.predict(x_train_binary)]

                    # feature reduction: Only use those walks which appear more than once:
                    #useful_features = np.sum(x_train_binary, axis=0) > 1
                    #train_features = train_features[:, useful_features]
                    #test_features = test_features[:, useful_features]

                else:
                    if use_forest:
                        clf = MINDWALCForest(path_max_depth=path_max_depth, max_tree_depth=max_tree_depth,
                                             min_samples_leaf=min_samples_leaf, n_estimators=n_estimators,
                                             fixed_walc_depth=fixed_walking_depth, path_min_depth=path_min_depth,
                                             n_jobs=n_jobs)
                    else:
                        clf = MINDWALCTree(path_max_depth=path_max_depth, max_tree_depth=max_tree_depth,
                                           min_samples_leaf=min_samples_leaf, fixed_walc_depth=fixed_walking_depth,
                                           path_min_depth=path_min_depth, n_jobs=n_jobs)

                    # train:
                    try:
                        clf.fit(kg, list(train_dataset['feature']), list(train_dataset['label']), post_prune=post_prune)
                    except Exception as e:
                        print(f"Error while training tree: {e}")
                        continue

                    # predict:
                    preds = clf.predict(kg, test_dataset['feature'])
                    preds_train = clf.predict(kg, train_dataset['feature'])

                # save test metrics:
                cross_val_results["f1"].append(f1_score(test_dataset['label'], preds, average=average_setting))
                cross_val_results["cohen_cappa"].append(cohen_kappa_score(test_dataset['label'], preds))
                cross_val_results["accuracy"].append(accuracy_score(test_dataset['label'], preds))
                cross_val_results["precision"].append(
                    precision_score(test_dataset['label'], preds, average=average_setting))

                # save training metrics:
                cross_val_results["f1_train"].append(
                    f1_score(train_dataset['label'], preds_train, average=average_setting))
                cross_val_results["cohen_cappa_train"].append(cohen_kappa_score(train_dataset['label'], preds_train))
                cross_val_results["accuracy_train"].append(accuracy_score(train_dataset['label'], preds_train))
                cross_val_results["precision_train"].append(
                    precision_score(train_dataset['label'], preds_train, average=average_setting))



                if store_all_trees or i_crossval == 0:

                    # first, store the dataset split used for cross-val:
                    sorted_train_data = train_dataset.to_dict()
                    for label in list(set([label for label in sorted_train_data['label'].values()])):
                        sorted_train_data[label] = []
                        for instance_index in sorted_train_data['label'].keys():
                            if sorted_train_data['label'][instance_index] == label:
                                rdf_node_id = sorted_train_data['feature'][instance_index].replace(gv_file_prefix, '')
                                node_names = [r['name'] for r in  session.run(f"match (n) where ID(n) = {rdf_node_id} return n.name as name")]
                                node_name = node_names[0] if len(node_names) > 0 else sorted_train_data['feature'][instance_index]
                                node_name = node_name if node_name else sorted_train_data['feature'][instance_index]
                                sorted_train_data[label].append(node_name)
                    sorted_train_data.pop('feature')
                    sorted_train_data.pop('label')
                    sorted_test_data = test_dataset.to_dict()
                    for label in list(set([label for label in sorted_test_data['label'].values()])):
                        sorted_test_data[label] = []
                        for instance_index in sorted_test_data['label'].keys():
                            if sorted_test_data['label'][instance_index] == label:
                                rdf_node_id = sorted_test_data['feature'][instance_index].replace(gv_file_prefix, '')
                                node_names = [r['name'] for r in session.run(f"match (n) where ID(n) = {rdf_node_id} return n.name as name")]
                                node_name = node_names[0] if len(node_names) > 0 else sorted_test_data['feature'][instance_index]
                                node_name = node_name if node_name else sorted_test_data['feature'][instance_index]
                                sorted_test_data[label].append(node_name)
                    sorted_test_data.pop('feature')
                    sorted_test_data.pop('label')
                    with open(result_path + f"/datasets/split_{i_crossval}.json", 'w') as f:
                        f.write(json.dumps({
                            "train": sorted_train_data,
                            "test": sorted_test_data,
                        }, indent=4))

                    # now generate the decision tree visualization ans store as pdf
                    if use_sklearn[setting_id]:
                        dot_data = sktree.export_graphviz(clf_sk_tree, out_file=None,
                                                        feature_names=[str(fn) for fn in walk_candidates],
                                                        class_names=list(label_name_to_int.keys()),
                                                        filled=True, rounded=True,
                                                        special_characters=True)
                        # Export tree to gv and pdf
                        src = graphviz.sources.Source(dot_data)
                        src.render(result_path + f"/trees/decision_tree{i_crossval}.gv", view=False)

                        tree_visualisation_postprocessor(result_path + f"/trees/decision_tree{i_crossval}.gv", gdb_adress, auth,
                                                         ["name", "FSN"],
                                                         depth_offset=tree_vis_depth_offset, depth_factor=tree_vis_depth_factor,
                                                         node_labels_to_hide=node_types_to_consider, kg=kg)

                        cross_val_results["node_count"].append(clf_sk_tree.tree_.node_count)
                        cross_val_results["max_tree_depth"].append(clf_sk_tree.tree_.max_depth)

                    else:
                        if use_forest:
                            tree = clf.estimators_[0].tree_
                            data_distribution_in_tree = clf.estimators_[0].validate_tree(kg,
                                                                                         list(train_dataset['feature']),
                                                                                         list(train_dataset['label']))
                        else:
                            tree = clf.tree_
                            data_distribution_in_tree = clf.validate_tree(kg, list(train_dataset['feature']),
                                                                          list(train_dataset['label']))

                        tree.visualise(result_path + f"/trees/decision_tree{i_crossval}.gv", False, as_pdf=False)
                        tree_visualisation_postprocessor(result_path + f"/trees/decision_tree{i_crossval}.gv", gdb_adress, auth,
                                                         ["name", "FSN"], data_distribution_in_tree=data_distribution_in_tree,
                                                         depth_offset=tree_vis_depth_offset, depth_factor=tree_vis_depth_factor, kg=kg)

                        cross_val_results["node_count"].append(tree.node_count)
                        cross_val_results["max_tree_depth"].append(tree.max_tree_depth)

            mean_values[tree_config_string]["node_count_mean"].append(np.mean(cross_val_results["node_count"]))
            mean_values[tree_config_string]["max_tree_depth_mean"].append(np.mean(cross_val_results["max_tree_depth"]))
            mean_values[tree_config_string]["node_count_std"].append(np.std(cross_val_results["node_count"]))
            mean_values[tree_config_string]["max_tree_depth_std"].append(np.std(cross_val_results["max_tree_depth"]))

            mean_values[tree_config_string]["x"].append(random_relation_removement)
            mean_values[tree_config_string]["f1_mean"].append(np.mean(cross_val_results["f1"]))
            mean_values[tree_config_string]["f1_std"].append(np.std(cross_val_results["f1"]))
            mean_values[tree_config_string]["cohen_cappa_mean"].append(np.mean(cross_val_results["cohen_cappa"]))
            mean_values[tree_config_string]["cohen_cappa_std"].append(np.std(cross_val_results["cohen_cappa"]))
            mean_values[tree_config_string]["accuracy_mean"].append(np.mean(cross_val_results["accuracy"]))
            mean_values[tree_config_string]["accuracy_std"].append(np.std(cross_val_results["accuracy"]))
            mean_values[tree_config_string]["precision_mean"].append(np.mean(cross_val_results["precision"]))
            mean_values[tree_config_string]["precision_std"].append(np.std(cross_val_results["precision"]))

            mean_values[tree_config_string]["f1_mean_train"].append(np.mean(cross_val_results["f1_train"]))
            mean_values[tree_config_string]["f1_std_train"].append(np.std(cross_val_results["f1_train"]))
            mean_values[tree_config_string]["cohen_cappa_mean_train"].append(np.mean(cross_val_results["cohen_cappa_train"]))
            mean_values[tree_config_string]["cohen_cappa_std_train"].append(np.std(cross_val_results["cohen_cappa_train"]))
            mean_values[tree_config_string]["accuracy_mean_train"].append(np.mean(cross_val_results["accuracy_train"]))
            mean_values[tree_config_string]["accuracy_std_train"].append(np.std(cross_val_results["accuracy_train"]))
            mean_values[tree_config_string]["precision_mean_train"].append(np.mean(cross_val_results["precision_train"]))
            mean_values[tree_config_string]["precision_std_train"].append(np.std(cross_val_results["precision_train"]))

            mean_values[tree_config_string]["node_count"].append(len(kg.vertices))
            mean_values[tree_config_string]["relation_count"].append(sum([len(x) for x in kg.transition_matrix.values()]))

            print(f"{tree_config_string}\tf1: {mean_values[tree_config_string]['f1_mean'][-1]} +- {mean_values[tree_config_string]['f1_std'][-1]}", flush=True)

            # save data as excel table:
            df = pd.DataFrame(cross_val_results)
            df.to_excel(result_path + "/metrics.xlsx", index=False)

            df = pd.DataFrame(mean_values[tree_config_string])
            df.to_excel(result_path_root + f"/{tree_config_string}_means.xlsx", index=False)

            # save the meta infos.
            with open(result_path + "/meta.json", 'w') as f:
                f.write(json.dumps({
                    "amount_trees": n_estimators if use_forest else 1,
                    "max_tree_depth": max_tree_depth,
                    "max_walk_depth": path_max_depth,
                    "min_walk_depth": path_min_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "measured_with:": f"{fold_amount}-fold cross validations",
                    "concept_types_to_consider": node_types_to_consider,
                    "fixed_walking_depth": fixed_walking_depth,
                    "relation_types_not_allowed_to_delete": relation_types_not_allowed_to_delete,
                    "relations_to_disconnect": relations_to_disconnect,
                    "concepts_to_disconnect": concepts_to_disconnect,
                    "relation_tail_merging": relation_tail_merge,
                    "mute_node_properties": mute_node_properties
                }, indent=4))

    return 0


if __name__ == '__main__':
    main()
