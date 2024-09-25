import os.path

from tqdm import tqdm
from neo4j import GraphDatabase
from sys import argv
from MINDWALC.neo4j2rdf import cypher_to_rdf
import json, yaml


# neo4j db of new neo4j dub for subgraph requirements:
# neo4j.conf:
# dbms.memory.heap.initial_size=4G
# dbms.memory.heap.max_size=4G
# dbms.unmanaged_extension_classes=n10s.endpoint=/rdf
# apoc-conf:
# apoc.import.file.enabled=true
# apoc.export.file.enabled=true

# workflow:
# first, find good value for max_path_length. In best case, you want to get 0.0% lonely center-nodes or very small value.
# During this search, use the much faster test_run mode, which will still give you the correct lonely-center-nodes count
# If you have the optimal max_path_length, set test_run to False and run the script again to get the final subgraph.

# params:
config_file_path = "./data/subgraphs/prostate_subgraphs/prostate-snomed-subgraph-config-1.3.yaml"

if __name__ == '__main__':

    # connect to neo4j dbms:
    gdb_adress = 'bolt://localhost:7687'
    # get pw from first argv:
    try:
        pw = argv[1]
    except IndexError:
        raise ValueError("Please provide the password as first argument.")
    driver = GraphDatabase.driver(gdb_adress, auth=('neo4j', pw))
    session = driver.session(database="neo4j")

    # load config file with yaml:
    with open(config_file_path, 'r') as f:
        subgraph_generation_config = yaml.safe_load(f)

    # handle center node ids:
    if subgraph_generation_config['center_node_label'] and subgraph_generation_config['center_node_ids']:
        raise ValueError("Config error: Please provide either center_node_label or center_node_ids, not both.")
    elif subgraph_generation_config['center_node_label']:
        center_node_ids_query = f"MATCH (n:{subgraph_generation_config['center_node_label']}) return n.{subgraph_generation_config['center_node_identification_property']} as id"
        center_node_ids = [r['id'] for r in session.run(center_node_ids_query)]
    elif subgraph_generation_config['center_node_ids']:
        center_node_ids = list(set(subgraph_generation_config["center_node_ids"]))
        if len(center_node_ids) < len(subgraph_generation_config["center_node_ids"]):
            print(f"WARNING: Some center nodes are duplicates and will be removed ("
                  f"{len(subgraph_generation_config['center_node_ids']) - len(center_node_ids)}).")
    else:
        raise ValueError("Config error: Please provide either center_node_label or center_node_ids.")


    if subgraph_generation_config["test_run"]:
        subgraph_name = subgraph_generation_config["subgraph_name"] + f"_p{subgraph_generation_config['max_path_length']}" + "_TEST"
    else:
        subgraph_name = subgraph_generation_config["subgraph_name"] + f"_p{subgraph_generation_config['max_path_length']}"

    if subgraph_generation_config["center_node_identification_property_type"] == "string":
        center_node_ids = [f'"{x}"' for x in center_node_ids]
    elif subgraph_generation_config["center_node_identification_property_type"] == "int":
        pass
    elif subgraph_generation_config["center_node_identification_property_type"] == "float":
        pass
    else:
        raise ValueError(f"Unknown center_node_identification_property_type: "
                         f"{subgraph_generation_config['center_node_identification_property_type']}")

    #center_node_ids = center_node_ids[:5]

    if subgraph_generation_config["search_for_directed_paths"]:
        raise NotImplementedError("Only undirected paths are implemented so far.")

    id_property = subgraph_generation_config["center_node_identification_property"]
    base_label = subgraph_generation_config["base_label"]
    neighborhood_extraction_size = subgraph_generation_config["neighborhood_extraction_size"]

    possible_path_count = 0
    found_paths_count = 0
    node_connection_map = {n_id : [] for n_id in center_node_ids}

    selected_relations = set()
    selected_nodes = set()

    # collect all possible start-target-node pairs:
    start_target_nodes = []
    for i in range(len(center_node_ids)):
        selected_nodes.add(center_node_ids[i])
        for j in range(i + 1, len(center_node_ids)):
            start_target_nodes.append((center_node_ids[i], center_node_ids[j]))

    print(f"Collecting paths between {len(center_node_ids)} center nodes "
          f"({len(start_target_nodes)} possible combinations)...", flush=True)

    # find paths between all start-target-node pairs:
    for center_node_id, target_node_id in tqdm(start_target_nodes):
        if center_node_id == target_node_id:
            continue

        possible_path_count += 1

        # find path:
        query = f"MATCH p = shortestPath((a:{base_label})-[rels*..{subgraph_generation_config['max_path_length']}]-(b:{base_label})) " \
                f"WHERE a.{id_property} = {center_node_id} AND b.{id_property} = {target_node_id} " \
                f"AND all(r IN rels WHERE {subgraph_generation_config['path_search_relation_condition']}) " \
                f"RETURN p"

        #print(query)
        paths = [r["p"] for r in session.run(query)]

        if not paths:
            pass
        else:
            found_paths_count += 1
            node_connection_map[center_node_id].append(target_node_id)
            node_connection_map[target_node_id].append(center_node_id)
            smallest_path_length = len(paths[0])

            # now, get all shortest paths (there might be multiple shortest paths):
            if not subgraph_generation_config["test_run"]:
                query = f"MATCH p = (a:{base_label})-[rels*{smallest_path_length}]-(b:{base_label}) " \
                        f"WHERE a.{id_property} = {center_node_id} AND b.{id_property} = {target_node_id} " \
                        f"AND all(r IN rels WHERE {subgraph_generation_config['path_search_relation_condition']}) " \
                        f"RETURN p"
                paths = [r["p"] for r in session.run(query)]

            for i, path in enumerate(paths):
                for j, relationship in enumerate(path):
                    for node in relationship.nodes:
                        #node_labels = list(node.labels)
                        selected_nodes.add(node.element_id)
                    selected_relations.add(relationship.element_id)
                    #print()

    # label center nodes:
    if subgraph_generation_config["label_center_nodes_with"]:
        #print("labeling center nodes...")
        session.run(f"match (a) remove a:{subgraph_generation_config['label_center_nodes_with']}")
        for center_node_id in center_node_ids:
            session.run(f"match (a:{base_label}) where a.{id_property} = {center_node_id} set a:{subgraph_generation_config['label_center_nodes_with']}")

    # Use neighborhood_extraction_size to add the whole neighborhood of the center nodes to the subgraph:
    if neighborhood_extraction_size > 0:
        print(f"Selected {len(selected_nodes)} nodes and {len(selected_relations)} relations.")
        print(f"Adding neighborhood of size {neighborhood_extraction_size} to the subgraph...")
        for center_node_id in tqdm(center_node_ids):
            query = f"MATCH (a:{base_label})-[r*1..{neighborhood_extraction_size}]-(b) " \
                    f"WHERE a.{id_property} = {center_node_id} " \
                    f"RETURN r"
            results = [r['r'] for r in session.run(query)]
            for relations in results:
                for relation in relations:
                    #selected_nodes.add(result["ID(a)"])
                    selected_nodes.add(relation.start_node.element_id)
                    selected_nodes.add(relation.end_node.element_id)
                    selected_relations.add(relation.element_id)
        print(f"Selected {len(selected_nodes)} nodes and {len(selected_relations)} relations.")

    # print some stats:
    #print(f"{round(float(found_paths_count) / float(possible_path_count), 2) * 100}% fully connected center nodes.")
    is_node_connected_to_any_center_node = [n_id for n_id in center_node_ids if len(node_connection_map[n_id]) > 0]
    #print(f"{100-round(len(is_node_connected_to_any_center_node)/len(center_node_ids),2)*100}% lonely center-nodes.")
    stats = {
        "center_node_count": len(center_node_ids),
        "possible_path_count": possible_path_count,
        "found_paths_count": found_paths_count,
        "fully_connected_center_nodes": f"{round(float(found_paths_count) / float(possible_path_count), 2) * 100} %",
        "lonely_center_nodes": f"{100-round(len(is_node_connected_to_any_center_node)/len(center_node_ids),2)*100} %",
        "relation_count": len(list(selected_relations)),
        "node_count": len((list(selected_nodes)))
    }
    print("Done.\nStats:")
    for k in stats:
        print(f" -{k}:\t{stats[k]}")
    print()

    # mark each selected node and relation with attribute subgraph_name=True:
    print(f"selecting {len(selected_relations)} relations and {len(selected_nodes)} nodes with .{subgraph_name}=True in neo4j db...")
    session.run(f"match (n) REMOVE n.{subgraph_name}") # clear previous selections
    session.run(f"match (a)-[r]-(b) REMOVE r.{subgraph_name}") # clear previous selections
    # dividing selected_nodes into batches of size 20:
    selected_nodes_batched = [list(selected_nodes)[i:i + 20] for i in range(0, len(selected_nodes), 20)]

    for node_id in tqdm(selected_nodes):
        session.run(f"match (n) where ID(n) = {node_id} set n.{subgraph_name} = True")
    for relation_id in tqdm(selected_relations):
        session.run(f"match (a)-[r]-(b) where ID(r) = {relation_id} set r.{subgraph_name} = True")

    print(f"Done. You can use the following query to get the selected nodes and relations:\n"
      f"match (a)-[r]->(b) where a.{subgraph_name} and b.{subgraph_name} and r.{subgraph_name} return *\n")

    ############# 2. convert selected subgraph to rdf: #############
    if subgraph_generation_config["export_rdf"]:
        rdf_out_file =  os.path.join(os.path.dirname(config_file_path), subgraph_name + '.n3')
        cypher_to_rdf(f"match (a)-[r]->(b) where a.{subgraph_name} and b.{subgraph_name} and r.{subgraph_name} return *",
                        rdf_out_file, 'localhost', ('neo4j', pw))
        print("saved selected subgraph as rdf file at " + rdf_out_file)

        # save configuration as json:
        json_out_path = os.path.join(os.path.dirname(config_file_path), subgraph_name + '.json')
        with open(json_out_path, 'w') as f:
            log_file = {k: subgraph_generation_config[k] for k in subgraph_generation_config}
            log_file["stats"] = stats
            log_file["node_connection_map"] = node_connection_map
            json.dump(log_file, f, indent=4)
        print("saved configuration as json file at " + json_out_path)

    if subgraph_generation_config["test_run"]:
        print("\n=== WARNING ===\nThis was a test run to find good parameters within short time. Not all possible paths are collected!"
              "\nFor the final run, set 'test_run' to False in the config.\n")
        exit()

    ######### 3. save as neo4j cypher import script: #############
    if not subgraph_generation_config["export_as_neo4j_cypher_import_script"]:
        exit()

    #query = "CALL apoc.export.json.query('match (a)-[r]->(b) where a.subgraph_name and b.subgraph_name and r.subgraph_name return a, r, b', null, {stream:true})"
    # file will be droped in /neo4j/db/home/import    the neo4j homedir can be found by clicking on "terminal" in the neo4j browser
    subgraph_neo4j_import_file = subgraph_name + "_import.cypher"
    #query_json_export = f'CALL apoc.export.json.query("match (a)-[r]->(b) where a.subgraph_name and b.subgraph_name and r.subgraph_name return a, r, b", "{json_out_path}")'
    query_cypher_export = (f'MATCH (a)-[r]->(b) where a.{subgraph_name} and b.{subgraph_name} and r.{subgraph_name} '
                           'WITH collect(DISTINCT a) + collect(DISTINCT  b) AS importNodes, collect(r) AS importRels '
                           'CALL apoc.export.cypher.data(importNodes, importRels, '
                           f'"{subgraph_neo4j_import_file}", '
                           '{ format: "plain", cypherFormat: "create" }) '
                           'YIELD file RETURN file;')
    results = session.run(query_cypher_export)
    print("saved selected subgraph as cypher script import file at /path/to/neo4j/import/" + subgraph_neo4j_import_file)

    ############# 4. load cypher into an new empty neo4j db: #############

    print("next, lets convert the rdf file to a graph object and load it into an empty neo4j db.\n"
          "For this, please do the following: \n"
          f" 1) Stop the currently running neo4j db.\n"
          f" 2) Create a new neo4j db (with same password as old db).\n"
          f" 3) Install apoc into the new db.\n"
          f" 4) Set dbms.memory.heap.initial_size=4G and dbms.memory.heap.max_size=4G in the neo4j.conf file.\n"
          f" 5) Set apoc.import.file.enabled=true in the apoc.conf file.\n"
          f" 6) Copy '{subgraph_neo4j_import_file}' from old db to the import folder of the new db.\n"
          f" 7) Start the new neo4j db and wait until it runs."
          "\nIf all done, press enter to continue, or enter 'exit' to abort.")
    cmd = input()
    if cmd.lower() == "exit":
        print("aborted.")
        exit()

    # reconnect to new neo4j db:
    driver = GraphDatabase.driver(gdb_adress, auth=('neo4j', pw))
    session = driver.session(database="neo4j")

    # check if db is empty:
    node_count = session.run("MATCH (n) return count(n)").single().value()
    while node_count > 0:
        cmd = input("Neo4j database is not empty. Can i erase the db? (y/n)")
        if cmd.lower() == "y":
            session.run("MATCH ()-[r]-() DELETE r")
            session.run("MATCH (n) DETACH DELETE n")
            print("done.")
        elif cmd.lower() == "n":
            print("aborted.")
            session.close()
            driver.close()
            exit()
        node_count = session.run("MATCH (n) return count(n)").single().value()

    print("\nimporting usind apoc.cypher.runFile(). This may take a while...")
    query_cypher_import = f'CALL apoc.cypher.runFile("{subgraph_neo4j_import_file}")'
    try:
        results = session.run(query_cypher_import)

        # check if there are duplications and if so, delete them:
        ids = [r["id"] for r in session.run(f"match (a:{base_label}) return distinct a.{id_property} as id")]
        unique_id_count = len(ids)
        all_id_count = len([r for r in session.run(f"match (a:{base_label}) return a.{id_property}")])

        if all_id_count > unique_id_count:
            print(
                f"WARNING: There are {all_id_count - unique_id_count} duplicated ids in the new neo4j db (might caused by nei4j bug?). Let me quickly clean this mess up...")
            for id in tqdm(ids):
                query = f"MATCH (a:{base_label}) where a.{id_property} = '{id}' RETURN ID(a) as id"
                dublicated_ids = [r["id"] for r in session.run(query)]
                if len(dublicated_ids) > 1:
                    for i_d in range(1, len(dublicated_ids)):
                        session.run(f"MATCH (a:{base_label}) where ID(a) = {dublicated_ids[i_d]} DETACH DELETE a")

        # put the config file into the subgraph as metadata
        meta_info = {k: subgraph_generation_config[k] for k in subgraph_generation_config if not str(k) in ["stats", "node_connection_map"]}
        meta_info_str = ', '.join([f"{k}: '{meta_info[k]}'" for k in meta_info.keys()])
        query = f"CREATE (:{subgraph_name} {{ {meta_info_str} }})"
        session.run(query)
        #print(query)

    except Exception as e:
        print(f"Error during import: {e}")
    print("done.")
    print(f"if the import has failed, no worries. You can try again by running the following query in the neo4j browser of the new neo4j db:")
    print(query_cypher_import)

    ''' # load rdf file as mindwalc.datastructures.Graph object:
    g = rdflib.Graph()
    g.parse(subgraph_generation_config["out_directory"], format='text/n3')
    kg = Graph.rdflib_to_graph(g, relation_tail_merging=False)

    # now load kg in neo4j:
    kg.graph_to_neo4j(password=pw)'''