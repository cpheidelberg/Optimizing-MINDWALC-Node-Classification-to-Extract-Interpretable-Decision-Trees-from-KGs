import re
from neo4j import GraphDatabase
import argparse
import sys, os

########## params ###########
neo4j_node_id_prefix = "neo4j://graph.individuals#"
neo4j_schema_id_prefix = "neo4j://graph.schema#"
neo4j_attribute_to_display_in_tree = ["name"]

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

def tree_visualisation_postprocessor(in_gv_file_path, neo4j_url="neo4j://localhost:7687", neo4j_auth=("neo4j", "password"),
                                     node_attribute_names=["name"], node_labels_to_hide=["ObjectConcept"],
                                     data_distribution_in_tree=None, show_data_distribution_in_tree=False, depth_offset=0, depth_factor=1.0):
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

    out_gv_file_path = in_gv_file_path.replace(".gv", "") + "_named.gv"

    # connect with neo4j db:
    driver = GraphDatabase.driver(neo4j_url, auth=neo4j_auth)
    session = driver.session()

    import networkx as nx
    g = nx.MultiDiGraph(nx.nx_pydot.read_dot(in_gv_file_path))
    for gv_node_id in g.nodes:
        node = g.nodes[gv_node_id]
        # {'color': '"#D6B656"', 'fillcolor': '"#FFF2CC"', 'label': '"neo4j://graph.individuals#1988\\nd = 4"', 'shape': '"box"', 'style': '"rounded,filled"'}

        if neo4j_node_id_prefix in node["label"]: # is it a decision node?

            # find neo4j node id:
            neo4j_nodeid_start_index = node["label"].find(neo4j_node_id_prefix)
            neo4j_node_id_end_index = None
            for c in range(neo4j_nodeid_start_index + len(neo4j_node_id_prefix), len(node["label"])):
                if not node["label"][c].isnumeric():
                    neo4j_node_id_end_index = c
                    break
            assert neo4j_node_id_end_index is not None and neo4j_node_id_end_index > neo4j_nodeid_start_index
            neo4j_node_id = node["label"][neo4j_nodeid_start_index + len(neo4j_node_id_prefix):neo4j_node_id_end_index]
            #node_meta_info = node["label"][neo4j_node_id_end_index:].replace('"', '')

            # get the node name:
            node_name = None
            for node_attribute_name in node_attribute_names:
                for r in session.run(f"MATCH (n) WHERE id(n) = {neo4j_node_id} "
                                     f"RETURN n.{node_attribute_name} AS name"):

                    if r["name"] is not None and r["name"] != "None":
                        node_name = r["name"]
                        if type(node_name) == list:
                            node_name = [n.replace('"', "").replace("'", "") for n in node_name]
                            node_name = f"id: {neo4j_node_id}, size: {len(node_name)}"  # this is just temp fix
                    break
                if node_name is not None:
                    break

            # get the node label:
            node_labels = \
            [r["label"] for r in session.run(f"MATCH (n) WHERE id(n) = {neo4j_node_id} RETURN labels(n) AS label")][0]

            for label_to_remove in node_labels_to_hide:
                if label_to_remove in node_labels and len(node_labels) > 1:  # avoid removing all labels!
                    node_labels.remove(label_to_remove)

            # find and modify the depths value using depth_offset and depth_factor:
            depth_prefix = "d = "
            end_prefix = '"'
            depth = None
            if depth_prefix in node["label"]:
                i_d_start = node["label"].index(depth_prefix) + len(depth_prefix)
                if end_prefix in node["label"][i_d_start:]:
                    i_d_end = node["label"][i_d_start:].index(end_prefix) + i_d_start
                    depth_string = node["label"][i_d_start:i_d_end]
                    try:
                        depth = int(depth_string)
                    except: # so could be a flexible walk which is a tuple / span:
                        if "(" in depth_string and ")" in depth_string:
                            depths = [d.replace('(', '').replace(')', '').replace(' ', '') for d in depth_string.split(",")]
                            try:
                                depth = [int(d) for d in depths]
                            except:
                                depth = None

            if depth is not None:
                if type(depth) == list:
                    depth = " - ".join([int((d + depth_offset)*depth_factor) for d in depth])
                else:
                    depth = int((depth + depth_offset) * depth_factor)
                node["label"] = node["label"].replace(f'{depth_prefix}{depth_string}', f'{depth_prefix}{depth}')
            # add data-distribution-infos:
            data_distribution = None
            if data_distribution_in_tree and show_data_distribution_in_tree:
                for dn in list(data_distribution_in_tree.keys()):
                    if dn.walk:
                        node_id = dn.walk[0].replace(neo4j_node_id_prefix, "")
                        if node_id == neo4j_node_id:
                            data_distribution_d = data_distribution_in_tree[dn].items()
                            data_distribution = str({k: len(v) for k, v in data_distribution_d})
                            data_distribution = data_distribution.replace("{", "").replace("}", "")
                            data_distribution = data_distribution.replace("'", "")
                            break

            # Set new label:
            #node['label'] = f'{node_name}\n{node_neo4j_labels}{node_meta_info}' if node_name else node['label']
            if node_name:
                node['label'] = node['label'].replace(neo4j_node_id_prefix + neo4j_node_id, node_name)
            if data_distribution:
                node['label'] += f"<br/>[{data_distribution}]"
        elif neo4j_schema_id_prefix in node["label"]: # is it a schema node?
            node['label'] = node['label'].replace(neo4j_schema_id_prefix, "")
        else: # it is a leaf node:
            leaf_node_label = node['label'].replace('"', '')
            if leaf_node_label in list(fill_color_dict.keys()):
                node['fillcolor'] = '"' + fill_color_dict[leaf_node_label] + '"'
                node['color'] = '"' + edge_color_dict[leaf_node_label] + '"'

            # add data-distribution-infos:
            if data_distribution_in_tree:
                for dn in list(data_distribution_in_tree.keys()):
                    if not dn.walk:
                        if dn.node_number == int(gv_node_id.replace("Node", "")):
                            data_distribution_d = data_distribution_in_tree[dn]
                            true_amount = len(data_distribution_d[leaf_node_label])
                            all_amount = sum([len(v) for k, v in data_distribution_d.items()])
                            node['label'] = f'<{leaf_node_label}<br/>{true_amount}/{all_amount} correct>'
                            break
            else:
                node['label'] = f'{leaf_node_label}'

        # apply changes:
        #g.nodes[gv_node_id] = node
        for k in node.keys():
            g.nodes[gv_node_id][k] = node[k]

    # save g as .gv file:
    nx.nx_pydot.write_dot(g, out_gv_file_path)
    try:
        # to pdf:
        # load gv_data:
        with open(out_gv_file_path, 'r') as f:
            gv_code_string = f.read()
        from graphviz import Source
        src = Source(gv_code_string)
        src.render(out_gv_file_path, view=False)
    except Exception as e:
        print(f"Could not render file {out_gv_file_path} to .pdf file: {e}")
        print(gv_code_string)


def main():

    # parse arguments:
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument("--neo4j_db_adress_user_pw",
                        default='neo4j://localhost:7687 neo4j password')
    args = parser.parse_args()
    neo4j_db = args.neo4j_db_adress_user_pw.split(' ')
    if len(neo4j_db) != 3:
        print("Wrong value for argument --neo4j_db_adress_user_pw ")
        print(
            "Please pass address, username and password for --neo4j_db_adress_user_pw separated with space like that:")
        print("--neo4j_db_adress_user_pw \"neo4j://ipaddress:port username password\"")
        exit(1)
    url = neo4j_db[0]
    auth = (neo4j_db[1], neo4j_db[2])

    for tree_to_posprocess in [f"data/tree_clf_performance/kg_curve_PokeReport_8/RRR_0.2/FixWalcDepth4-4_DT/trees/example_tree{i}.gv" for i in range(10)]:
        print(f"processing tree {os.path.basename(tree_to_posprocess)} in {os.path.dirname(tree_to_posprocess)}...")
        tree_visualisation_postprocessor(tree_to_posprocess, url, auth, neo4j_attribute_to_display_in_tree, depth_factor=0.5, depth_offset=-2, show_data_distribution_in_tree=False)

    return 0

if __name__ == "__main__":
    main()