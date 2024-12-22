import re
from neo4j import GraphDatabase
from sys import argv
import sys, os
import yaml
import pickle
from MINDWALC.mindwalc.datastructures import Graph, Vertex
from pyvis.network import Network

########## params ###########
neo4j_node_id_prefix = "neo4j://graph.individuals#"
neo4j_schema_id_prefix = "neo4j://graph.schema#"

######## coloring ##########
# bas colors for leafs:
red_fill = '#F8CECC' # red
red_edge = '#B85450'
blue_fill = '#DAE8FC' # blue
blue_edge = '#6C8EBF'
green_fill = '#D5E8D4' # green
green_edge = '#82B366'
fill_color_dict = {
    'water': blue_fill,  # water
    'fire': red_fill,  # fire
    'grass': green_fill,  # grass
}
edge_color_dict = {
    'water': blue_edge,
    'fire': red_edge,
    'grass': green_edge,
}

def visualize_paths_pyvis(paths, info_text="", output_file='interactive_graph.html', merge_relations=True):
    """
    Visualizes a collection of paths as an interactive HTML file using pyvis with improved physics settings and displays an info text.

    Args:
        paths (list of list of str): A collection of paths, where each path is a list of node names.
        info_text (str): General information text to display above the graph.
        output_file (str): The name of the output HTML file.
        :param merge_relations:  If True, relations are merged into one node. Might look confusing but provides better overview'of all available paths.
    """
    # Initialize a directed graph using pyvis
    net = Network(notebook=False, directed=True, height="600px", width="100%")

    # Collect all nodes and edges
    nodes = set()
    edges = set() # Use a set to ensure edges are unique
    relation_counter = 0 # to make relations unique

    # rename schema=relations=prredicate nodes for better readability:
    for path in paths:
        for i in range(len(path)):
            if neo4j_schema_id_prefix in path[i]:
                path[i] = path[i].replace(neo4j_schema_id_prefix, "RELATION: ") + ("" if merge_relations else f"#REL{relation_counter}")
                relation_counter += 1

    # Add nodes and edges from the paths
    for path in paths:
        for i in range(len(path)):
            nodes.add(path[i])  # Collect all nodes
            if i < len(path) - 1:
                edges.add((path[i], path[i + 1]))  # Collect all edges

    # Identify start and end nodes
    start_nodes = {path[0] for path in paths}  # Nodes at the start of any path
    end_nodes = {path[-1] for path in paths}  # Nodes at the end of any path

    # Add nodes with appropriate colors
    for node in nodes:
        if node in start_nodes:
            color = 'blue' if 'iga' in node.lower() else 'green'
            net.add_node(node, label=node, color=color)  # Start nodes are blue
        elif node in end_nodes:
            net.add_node(node, label=node, color='red')  # End nodes are red
        else:
            if "RELATION: " in node:
                color = 'gray'
            else:
                color = 'orange'
            net.add_node(node, label=node.split('#REL')[0], color=color)

    # Add edges
    for source, target in edges:
        net.add_edge(source, target)

    # Customize physics settings to reduce oscillations
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {
          "iterations": 200,
          "fit": true
        },
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 100,
          "springConstant": 0.04
        },
        "minVelocity": 0.75
      }
    }
    """)

    # Generate the interactive HTML file
    html_content = net.generate_html()

    # Add the info text to the HTML content
    final_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Interactive Graph</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
            .info-text {{ margin: 20px; padding: 10px; background-color: #f9f9f9; border-left: 5px solid #2196F3; }}
            .graph-container {{ margin: 20px; }}
        </style>
    </head>
    <body>
        <div class="info-text">
            <h4>How to read:</h4>
            <p><span style="color:green;">Green</span>: Amyloidosis case, 
            <span style="color:blue;">Blue</span>: IGA case, 
            <span style="color:gray;">Grey</span>: A Relation (in MINDWALC, relation-edges are converted to nodes), 
            <span style="color:red;">Red</span>: Target node of the walk, 
            <span style="color:orange;">Orange</span>: A Concept Node</p>
            {'<p>The relation-nodes are merged together to get a better overview over the existing paths. In the original graph, each relation-node is actually unique!</p>' if merge_relations else ''}
            <h4>Information:</h4>
            <p>{info_text}</p>
        </div>
        <div class="graph-container">
            {html_content}
        </div>
    </body>
    </html>
    """

    # Save the HTML to the output file
    with open(output_file, 'w') as file:
        file.write(final_html)

    #print(f"Interactive visualization with info text saved as {output_file}")

def tree_visualisation_postprocessor(in_gv_file_path, neo4j_url="neo4j://localhost:7687", neo4j_auth=("neo4j", "password"),
                                     node_attribute_names=["name"], node_labels_to_hide=["ObjectConcept"],
                                     data_distribution_in_tree=None, show_data_distribution_in_tree=True, depth_offset=0, depth_factor=1.0,
                                     kg=None):
    '''
    takes in a .gv file and replaces the node ids with the attribute-values of the node in the neo4j db,
    using parameter node_attribute_names.
    Saves the new .gv file and renders it to a .pdf file with name of the input file + "_named.gv" and "_named.pdf"

    :param in_gv_file_path: path to the input .gv file
    :param neo4j_url: url to the neo4j db
    :param neo4j_auth: tuple with username and password for the neo4j db
    :param node_attribute_names: list of strings with the names of the attributes of the nodes in the neo4j db which
    shall be used to replace the node ids (if the first attribute in the list is not available, the second one is used etc.)
    '''

    assert type(node_attribute_names) == list and len(node_attribute_names) > 0

    out_gv_file_path = in_gv_file_path.replace(".gv", "") + "_named.gv" # could also use .dot?
    dt_index = in_gv_file_path.split("/")[-1].replace(".gv", "").replace("decision_tree", "")

    tree_instance_dataset_path = in_gv_file_path.split("trees")[0] + f"datasets/split_{dt_index}.json"
    import json
    with open(tree_instance_dataset_path, 'r') as f:
        tree_instance_dataset = json.load(f)

    config_yaml_path = in_gv_file_path.split("trees")[0] + "../.." + "/config.yaml"
    with open(config_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    #new_line_symbol = "<br/>"
    new_line_symbol = "\\n"

    skip_predicates_during_path_collection = False

    subgraph_name = config["subgraph_name"]
    instance_type = "Report"
    if "_RTM" in in_gv_file_path.split("trees")[0]:
        path_to_kg = in_gv_file_path.split("trees")[0] + ".." + "/kg_rtm.pkl"
        depth_modifier = 1.0
    else:
        path_to_kg = in_gv_file_path.split("trees")[0] + ".." + "/kg_non_rtm.pkl"
        if skip_predicates_during_path_collection:
            depth_modifier = 0.5
        else:
            depth_modifier = 1.0
    if kg is None:
        with open(path_to_kg, 'rb') as f:
            kg = pickle.load(f)

    # connect with neo4j db:
    driver = GraphDatabase.driver(neo4j_url, auth=neo4j_auth)
    session = driver.session()

    import networkx as nx
    g = nx.MultiDiGraph(nx.nx_pydot.read_dot(in_gv_file_path))

    # create meta-data folder if not exists:
    meta_data_folder = os.path.join(os.path.dirname(in_gv_file_path), "meta")
    if not os.path.exists(meta_data_folder):
        os.makedirs(meta_data_folder)

    for gv_target_node_id in g.nodes:
        decision_node = g.nodes[gv_target_node_id]

        # {'color': '"#D6B656"', 'fillcolor': '"#FFF2CC"', 'label': '"neo4j://graph.individuals#1988\\nd = 4"', 'shape': '"box"', 'style': '"rounded,filled"'}

        dt_node_meta_path = f'{meta_data_folder}/DT{dt_index}_infos_Node{gv_target_node_id}.json'
        tree_data_info_path = f'{meta_data_folder}/{in_gv_file_path.split("/")[-1].replace(".gv", "_tree_data_info.json")}'
        if neo4j_node_id_prefix in decision_node["label"]: # is it a decision node?

            # find neo4j node id:
            neo4j_nodeid_start_index = decision_node["label"].find(neo4j_node_id_prefix)
            neo4j_node_id_end_index = None
            for c in range(neo4j_nodeid_start_index + len(neo4j_node_id_prefix), len(decision_node["label"])):
                if not decision_node["label"][c].isnumeric():
                    neo4j_node_id_end_index = c
                    break
            assert neo4j_node_id_end_index is not None and neo4j_node_id_end_index > neo4j_nodeid_start_index
            target_node_neo4j_id = decision_node["label"][neo4j_nodeid_start_index + len(neo4j_node_id_prefix):neo4j_node_id_end_index]
            #node_meta_info = node["label"][neo4j_node_id_end_index:].replace('"', '')

            # get the node name:
            target_node_name = None
            for node_attribute_name in node_attribute_names:
                for r in session.run(f"MATCH (n) WHERE id(n) = {target_node_neo4j_id} "
                                     f"RETURN n.{node_attribute_name} AS name, labels(n) AS label"):
                    if r["name"] is not None and r["name"] != "None":
                        target_node_name = f'id: {target_node_neo4j_id}{new_line_symbol}{node_attribute_name}: {r["name"]}'
                        node_type = [l for l in r["label"] if l not in node_labels_to_hide]
                        if node_type:
                            target_node_name += f"{new_line_symbol}type: " + str(node_type[0]) if len(node_type) == 1 else "&".join(node_type)
                    break
                if target_node_name is not None:
                    break

            # get the node label:
            node_labels = \
            [r["label"] for r in session.run(f"MATCH (n) WHERE id(n) = {target_node_neo4j_id} RETURN labels(n) AS label")][0]

            for label_to_remove in node_labels_to_hide:
                if label_to_remove in node_labels and len(node_labels) > 1:  # avoid removing all labels!
                    node_labels.remove(label_to_remove)

            # find and modify the depths value using depth_offset and depth_factor:
            depth_prefix = "d = "
            end_prefix = '"'
            depth = None
            if depth_prefix in decision_node["label"]:
                i_d_start = decision_node["label"].index(depth_prefix) + len(depth_prefix)
                if end_prefix in decision_node["label"][i_d_start:]:
                    i_d_end = decision_node["label"][i_d_start:].index(end_prefix) + i_d_start
                    depth_string = decision_node["label"][i_d_start:i_d_end]
                    try:
                        depth = int(depth_string)
                    except: # so could be a flexible walk which is a tuple / span:
                        if "(" in depth_string and ")" in depth_string:
                            depths = [d.replace('(', '').replace(')', '').replace(' ', '') for d in depth_string.split(",")]
                            try:
                                depth = [int(d) for d in depths]
                            except:
                                depth = None

            # add data-distribution-infos:
            data_distribution = None
            if data_distribution_in_tree and show_data_distribution_in_tree:
                for mindwalc_tree_node in list(data_distribution_in_tree.keys()):
                    if mindwalc_tree_node.walk:
                        d = mindwalc_tree_node.walk[1]
                        target_node_id_rdf = mindwalc_tree_node.walk[0]
                        target_node_id = target_node_id_rdf.split('#')[-1]
                        if target_node_id == target_node_neo4j_id:
                            data_distribution_d = data_distribution_in_tree[mindwalc_tree_node].items()
                            data_distribution = str({k: len(v) for k, v in data_distribution_d})
                            data_distribution = data_distribution.replace("{", "").replace("}", "")
                            data_distribution = data_distribution.replace("'", "")

                            #print(f"data_distribution: {data_distribution}")
                            #print(f"Walk/DT: d={d} target={target_node_id_rdf}")
                            #g_paths = Graph()
                            available_paths = []
                            for instance_label, instance_nodes in data_distribution_d:
                                #print(f"Searching paths from {len(instance_nodes)} {instance_label}-labeled cases to target node {target_node_id_rdf}...")

                                for instance_node in instance_nodes:
                                    for path in kg.extract_paths(str(instance_node), d if type(d) == int else d[1], skip_predicates=skip_predicates_during_path_collection):
                                        if path[-1].name == str(target_node_id_rdf):
                                            if type(d) == int:  # fixed walk:
                                                if len(path) - 1 == int(d*depth_modifier):
                                                    available_paths.append(path)
                                            else:  # flexible walk:
                                                if len(path) - 1 >= d[0] and len(path) - 1 <= d[1]:
                                                    available_paths.append(path)


                                #amount_instances = len(set(p[0].name for p in available_paths))
                                #print(f"Found {len(available_paths)} paths for {amount_instances} {instance_label}-labeled cases.")

                            # query neo4j to collect all readable names of the nodes:
                            rdf_name_to_neo4j_id =  {}
                            for path in available_paths:
                                for node in path:
                                    if '_MODIFIED_' in node.name:
                                        modifier = node.name.split(neo4j_schema_id_prefix)[1].split(neo4j_node_id_prefix)[0].replace('_MODIFIED_', '')
                                        id = node.name.split(neo4j_node_id_prefix)[1]
                                        id = (modifier, id)
                                    elif neo4j_node_id_prefix in node.name:
                                        id = node.name.split(neo4j_node_id_prefix)[1]
                                    else:
                                        id = None
                                    if id:
                                        rdf_name_to_neo4j_id[node.name] = id
                            rdf_name_to_neo4j_name = {}
                            for rdf_name, id in rdf_name_to_neo4j_id.items():
                                if type(id) == tuple:
                                    modifier, id = id
                                    results = [r for r in session.run(f"MATCH (n) WHERE id(n) = {id} RETURN n.name AS name")]
                                    assert len(results) == 1
                                    rdf_name_to_neo4j_name[rdf_name] = f"{modifier}-modified:\n{results[0]['name']} ({id})"
                                else:
                                    results = [r for r in session.run(f"MATCH (n) WHERE id(n) = {id} RETURN n.name AS name, ID(n) AS id")]
                                    assert len(results) == 1
                                    rdf_name_to_neo4j_name[rdf_name] = results[0]['name'] + f" ({id})"

                            amount_instances = len(set(p[0].name for p in available_paths))

                            '''for p in available_paths:
                                pref_v = None
                                for i, v in enumerate(p):
                                    new_name = rdf_name_to_neo4j_name[v.name] if v.name in rdf_name_to_neo4j_name else v.name
                                    if not new_name in list(g_paths.name_to_vertex.keys()):
                                        new_v = Vertex(new_name, predicate=v.predicate, relation_modified=v.relation_modified)
                                    else:
                                        new_v = g_paths.name_to_vertex[new_name]
                                    g_paths.add_vertex(new_v)

                                    if pref_v:
                                        g_paths.add_edge(pref_v, new_v)
                                    pref_v = new_v

                            fig = g_paths.visualise(draw_predicate_nodes_as_edges=False)
                            fig.suptitle(f'All paths of length {d} from {amount_instances} {instance_type}-nodes to target-node {target_node_id_rdf}')
                            fig.savefig(f"{meta_data_folder}/DT{dt_index}_paths_{gv_node_id}.png", dpi=300)'''

                            available_paths_str = []
                            for p in available_paths:
                                path_str = []
                                for v in p:
                                    new_name = rdf_name_to_neo4j_name[v.name] if v.name in rdf_name_to_neo4j_name else v.name
                                    path_str.append(new_name)
                                available_paths_str.append(path_str)

                            visualize_paths_pyvis(available_paths_str,
                                                  f'<a href="./{dt_node_meta_path.split("/")[-1]}" target="_blank">click here to see more details about this walk/decision node</a>',
                                                  f"{meta_data_folder}/DT{dt_index}_paths_{gv_target_node_id}.html")

                            decision_node['href'] = "./meta/" + f"DT{dt_index}_paths_{gv_target_node_id}.html"

                            ##### collect decision-node metadata:####
                            dt_meta = {}
                            dt_meta["gv_tree_file"] = in_gv_file_path
                            dt_meta["gv_target_node_id"] = gv_target_node_id
                            dt_meta["neo4j_target_node_id"] = target_node_neo4j_id
                            dt_meta["walking_depth"] = depth
                            # dt_meta["walking_depth_str"] = depth_str
                            dt_meta["target_node_name"] = target_node_name
                            dt_meta["target_node_labels"] = node_labels
                            dt_meta["paths"] = []

                            dt_meta["path_count_per_label"] = {}

                            if available_paths_str:
                                for i_p, path in enumerate(available_paths_str):
                                    path_string = ""
                                    path_list = []
                                    allpath_nodes = []
                                    ner_result_of_path = None
                                    instance_node = None
                                    for i, node in enumerate(path):
                                        if i == 0:
                                            instance_node = node
                                        if i == len(path) - 1:
                                            path_string += f"{node}"
                                        else:
                                            path_string += f"{node} -> "
                                        path_list.append(node)
                                        allpath_nodes.append(node)

                                    if len(path) > 2:
                                        instance_id = available_paths[i_p][0].name.replace(neo4j_node_id_prefix, "")
                                        sentence_id = available_paths[i_p][2].name.replace(neo4j_node_id_prefix, "")
                                        q = f"MATCH (n)-[r]->(m) WHERE id(n) = {instance_id} and id(m) = {sentence_id} RETURN r.text AS text"
                                        results = [r for r in session.run(q)]
                                        if results:
                                            ner_result_of_path = results[0]["text"]

                                    pass
                                    if {"path": path_list, "ner_result": ner_result_of_path} not in dt_meta["paths"]:
                                        dt_meta["paths"].append({"path": path_list, "ner_result": ner_result_of_path})
                            else:
                                pass



                            # store the decision-node metadata:
                            import json
                            with open(dt_node_meta_path, 'w') as f:
                                json.dump(dt_meta, f, indent=4)
                            #decision_node['href'] = "./meta/" + dt_node_meta_path.split("/")[-1]

                            if dt_meta:
                                decision_node["label"] = decision_node["label"].replace(f'{depth_prefix}{depth_string}',
                                                                                        f'{new_line_symbol}walk-depth: {d}')
                            else:
                                decision_node["label"] = decision_node["label"].replace(f'{depth_prefix}{depth_string}',
                                                                                        f'{new_line_symbol}walk-depth: {d}')



            # Set new label:
            #node['label'] = f'{node_name}\n{node_neo4j_labels}{node_meta_info}' if node_name else node['label']
            if target_node_name:
                decision_node['label'] = decision_node['label'].replace(neo4j_node_id_prefix + target_node_neo4j_id, target_node_name).replace("_MODIFIED_", f"-MODIFIED:{new_line_symbol}").replace(neo4j_schema_id_prefix, "")
            if data_distribution:
                if decision_node['label'][-1] == '"':
                    decision_node['label'] = decision_node['label'][:-1]
                    decision_node['label'] += f"{new_line_symbol}[{data_distribution}]"
                    decision_node['label'] += '"'
                else:
                    decision_node['label'] += f"{new_line_symbol}[{data_distribution}]"

        elif neo4j_schema_id_prefix in decision_node["label"]: # is it a schema node?
            decision_node['label'] = decision_node['label'].replace(neo4j_schema_id_prefix, "")
        else: # it is a leaf node:
            leaf_node_label = decision_node['label'].replace('"', '')
            if leaf_node_label in list(fill_color_dict.keys()):
                decision_node['fillcolor'] = '"' + fill_color_dict[leaf_node_label] + '"'
                decision_node['color'] = '"' + edge_color_dict[leaf_node_label] + '"'

            # add data-distribution-infos:
            if data_distribution_in_tree:
                for mindwalc_tree_node in list(data_distribution_in_tree.keys()):
                    if not mindwalc_tree_node.walk:
                        if mindwalc_tree_node.node_number == int(gv_target_node_id.replace("Node", "")):
                            data_distribution_d = data_distribution_in_tree[mindwalc_tree_node]
                            true_amount = len(data_distribution_d[leaf_node_label])
                            all_amount = sum([len(v) for k, v in data_distribution_d.items()])
                            decision_node['label'] = f'"{leaf_node_label}{new_line_symbol}{true_amount}/{all_amount} correct"'
                            break
            else:
                decision_node['label'] = f'{leaf_node_label}'

        #decision_node['image'] = "/...jpg"

        # apply changes:
        #g.nodes[gv_node_id] = node
        for k in decision_node.keys():
            g.nodes[gv_target_node_id][k] = decision_node[k]

    # save g as .gv file:
    gv_code_string = None
    try:
        nx.nx_pydot.write_dot(g, out_gv_file_path)
        with open(out_gv_file_path, 'r') as f:
            gv_code_string = f.read()
        from graphviz import Source
        src = Source(gv_code_string)
        src.render(out_gv_file_path, view=False)
    except Exception as e:
        pass
        #print(f"Could not render file {out_gv_file_path} to .pdf file: {e}")
        #print(gv_code_string)


def main():

    # parse arguments:
    try:
        auth = ("neo4j", argv[1])
        url = "neo4j://localhost:7687"
    except IndexError:
        raise ValueError("Please provide neo4j db password as first argument.")

    dataset_path = "data/RRR_node_clf/rrr_curve_IgaAmyReports_4"

    # fins all .gv files in all subdirs in dataset_path:
    gv_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".gv"):
                if "named" not in file:
                    gv_files.append(os.path.join(root, file))

    print(f"Found {len(gv_files)} .gv files in {dataset_path}")

    # go through all gv_files and process them:
    from tqdm import tqdm
    for gv_file in tqdm(gv_files):
        # print(f"Processing {gv_file}")
        tree_visualisation_postprocessor(gv_file, url, auth, ["name"], depth_offset=0, depth_factor=1, show_data_distribution_in_tree=True)

    return 0

if __name__ == "__main__":
    main()