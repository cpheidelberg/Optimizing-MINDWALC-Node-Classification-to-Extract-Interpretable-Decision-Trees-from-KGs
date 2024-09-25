from neo4j import GraphDatabase
import os, sys
import pandas as pd
from tqdm import tqdm
import time
import numpy as np

''' ### what this script does ###
- can be used to randomly generate case-instances and connect them with a connected neo4j knowledge graph database.
- given is an table containing classes (e.g. each class represents a diagnose of a case) and for each class a list of (medical) features which are typically appear in cases of the given class
- This script will generate a random case-instance for each class and connect it with the knowledge graph, according to the provided feature-list of the class.
'''

### params:
# Table params:
feature_id_colum = 'Features'
label_id_colum = 'Case-diagnosis'
in_table_path = 'data/subgraphs/prostate_subgraphs/SyntheticInstanceGeneration1.1.xlsx'

# graph params:
name_of_id_attribute = 'sctid'
type_of_id_attribute = "string"
node_type = "ObjectConcept"
node_instance_type = "ProstateInstance"
interweave_relation = "RECOGNIZED_PATTERN"
name_of_label_node_attribute = "prostate_label"

# Instance generation params:
number_of_instances_per_class = 111#111
feature_weaving_probability = 1.0

clearing_queries = [
    f"MATCH (n:{node_instance_type}) DETACH DELETE n"
]

if __name__ == "__main__":

    if type_of_id_attribute != "string":
        raise NotImplementedError("Only string id_attribute_type is supported.")

    # connect to neo4j dbms:
    gdb_adress = 'bolt://localhost:7687'
    try:
        pw = sys.argv[1]
    except IndexError:
        raise ValueError("Please provide the password as first argument.")
    auth = ('neo4j', pw)
    driver = GraphDatabase.driver(gdb_adress, auth=auth)
    session = driver.session(database="neo4j")

    # load the table:
    print(f"Reading table from '{in_table_path}'")
    df = pd.read_excel(in_table_path)

    # first collect the features for each label:
    label_to_feature_list = {}
    for row in df.iterrows():
        label = row[1][label_id_colum]
        feature = row[1][feature_id_colum]

        if (not type(feature)==str and not feature) or pd.isna(feature):
            continue
        if (not type(label)==str and not label) or pd.isna(label):
            continue

        label = label.lower()
        feature = feature.lower()

        if not label in label_to_feature_list.keys():
            label_to_feature_list[label] = []

        label_to_feature_list[label].append(feature)

    # print:
    for k in [l for l in label_to_feature_list.keys()]:
        label_to_feature_list[k] = list(set(label_to_feature_list[k]))
        print(f"'{k}' has {len(label_to_feature_list[k])} possible features: {label_to_feature_list[k]}")
    print()

    # check if there are any instances in the dbms:
    instances = [r for r in session.run(f"MATCH (n:{node_instance_type}) RETURN ID(n)")]
    if len(instances) > 0:
        print(f"Found {len(instances)} instances in the database.")
        cmd = input("Do you want to delete all instances? (y/n): ")
        if cmd == "y":
            print("Deleting all instances...")
            for q in clearing_queries:
                session.run(q)
        else:
            print("Aborting.")
            exit()

    not_available_features = []

    for i_class, class_label in enumerate(label_to_feature_list.keys()):
        print(f"Generating {number_of_instances_per_class} instances for class '{class_label}'")
        feature_list = label_to_feature_list[class_label]

        # generate instances:
        for i in tqdm(r_i for r_i in range(number_of_instances_per_class)):
            # generate a random instance:
            instance_id = f"case_{i_class}-{i}"
            instance_features = []
            for f in feature_list:
                if np.random.rand() < feature_weaving_probability:
                    instance_features.append(f)

            # create the instance node:
            create_instance_query = f"CREATE (:{node_instance_type} {{ name: '{instance_id}', {name_of_label_node_attribute}: '{class_label}' }})"
            #print(create_instance_query)
            session.run(create_instance_query)

            # connect the instance node with the features:
            for f in instance_features:

                if '|' in f:
                    node_id = f.split('|')[0]
                    while node_id[-1] == ' ':
                        node_id = node_id[:-1]
                    while node_id[0] == ' ':
                        node_id = node_id[1:]
                else:
                    node_id = f

                # check if feature node f is present in the graph:
                check_query = (f"MATCH (f:{node_type}) WHERE f.{name_of_id_attribute} = '{node_id}' "
                               f"RETURN ID(f)")
                result = session.run(check_query)
                if len([r for r in result]) == 0:
                    not_available_features.append(node_id)
                    continue

                connect_feature_query = (f"MATCH (i:{node_instance_type}) WHERE  i.name = '{instance_id}' "
                                         f"MATCH (f:{node_type}) WHERE f.{name_of_id_attribute} = '{node_id}' "
                                         f"MERGE (i)-[:{interweave_relation}]->(f)")
                session.run(connect_feature_query)

    print("\nFinished instance generation and interweaving.")
    created_instances = session.run(f"MATCH (n:{node_instance_type}) RETURN ID(n)")
    print(f"Created {len([r for r in created_instances])} instances")
    created_relations = session.run(f"MATCH ()-[r:{interweave_relation}]->() RETURN ID(r)")
    print(f"Created {len([r for r in created_relations])} relations")

    if len(not_available_features) > 0:
        print(f"\nWARNING: {len(not_available_features)} relations could not been created, \n"
              f"because the following {len(list(set(not_available_features)))} features are not available in the graph:")
        for f in list(set(not_available_features)):
            print(f)

    session.close()
    driver.close()
    exit()