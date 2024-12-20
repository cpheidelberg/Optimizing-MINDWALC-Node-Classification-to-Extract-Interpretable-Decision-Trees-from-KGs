import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp

from collections import defaultdict, Counter, OrderedDict
from functools import lru_cache
import heapq

import os
import itertools
import time
from graphviz import Source
import rdflib
from tqdm import tqdm
import re

from scipy.stats import entropy

# The idea of using a hashing function is taken from:
# https://github.com/benedekrozemberczki/graph2vec
from hashlib import md5
import copy


class Vertex(object):

    def __init__(self, name, predicate=False, _from=None, _to=None, relation_modified=False):
        self.name = name
        self.predicate = predicate
        self.relation_modified = relation_modified
        self._from = _from
        self._to = _to

    def __eq__(self, other):
        if other is None:
            return False
        return self.__hash__() == other.__hash__()

    def get_name(self):
        return self.name

    def __hash__(self):
        if self.predicate:
            return hash((self._from, self._to, self.name))
        else:
            return hash(self.name)

    def __lt__(self, other):
        if self.predicate and not other.predicate:
            return False
        if not self.predicate and other.predicate:
            return True
        if self.predicate:
            return (self.name, self._from, self._to) < (other.name, other._from, other._to)
        else:
            return self.name < other.name


class Graph(object):
    _id = 0

    def __init__(self):
        self.vertices = set()
        self.transition_matrix = defaultdict(set)
        self.name_to_vertex = {}
        self.root = None
        self._id = Graph._id
        Graph._id += 1

    def add_vertex(self, vertex):
        if vertex not in self.vertices:
            self.vertices.add(vertex)

        self.name_to_vertex[vertex.name] = vertex

    def add_edge(self, v1, v2):
        self.transition_matrix[v1].add(v2)

    def get_neighbors(self, vertex):
        return self.transition_matrix[vertex]

    def visualise(self, figsize=(10, 10), draw_predicate_nodes_as_edges=False): # TODO: push this change to MINMDWALC repo!
        nx_graph = nx.DiGraph()

        if draw_predicate_nodes_as_edges:
            for v in self.vertices:
                if not v.predicate:
                    name = v.name.split('/')[-1]
                    nx_graph.add_node(name, name=name, pred=v.predicate)

            for v in tqdm(self.vertices):
                if not v.predicate:
                    v_name = v.name.split('/')[-1]
                    # Neighbors are predicates
                    for pred in self.get_neighbors(v):
                        pred_name = pred.name.split('/')[-1]
                        for obj in self.get_neighbors(pred):
                            obj_name = obj.name.split('/')[-1]
                            nx_graph.add_edge(v_name, obj_name, name=pred_name)
        else:
            for v in self.vertices:
                name = v.name.split('/')[-1]
                if v.predicate:
                    name += "\n(P)"
                elif v.relation_modified:
                    name += "\n(RTM)"
                nx_graph.add_node(name, name=name, pred=v.predicate)

            for v in tqdm(self.vertices):
                name_v = v.name.split('/')[-1]
                if v.predicate:
                    name_v += "\n(P)"
                elif v.relation_modified:
                    name_v += "\n(RTM)"
                for neighbor in self.get_neighbors(v):
                    name_b = neighbor.name.split('/')[-1]
                    if neighbor.predicate:
                        name_b += "\n(P)"
                    elif neighbor.relation_modified:
                        name_b += "\n(RTM)"

                    nx_graph.add_edge(name_v, name_b)

        plt.figure(figsize=figsize)
        _pos = nx.circular_layout(nx_graph)
        nx.draw_networkx_nodes(nx_graph, pos=_pos)
        nx.draw_networkx_edges(nx_graph, pos=_pos)
        nx.draw_networkx_labels(nx_graph, pos=_pos)
        nx.draw_networkx_edge_labels(nx_graph, pos=_pos,
                                     edge_labels=nx.get_edge_attributes(nx_graph, 'name'))
        plt.show()

    def extract_neighborhood(self, instance, depth=8):
        neighborhood = Neighborhood()
        root = self.name_to_vertex[str(instance)]
        to_explore = {root}

        for d in range(depth + 1):
            new_explore = set()
            for v in list(to_explore):
                if not v.predicate:
                    neighborhood.depth_map[d].add(v.get_name())
                for neighbor in self.get_neighbors(v):
                    new_explore.add(neighbor)
            to_explore = new_explore

        return neighborhood

    def extract_paths(self, instance, depth):

        assert depth > 0, "Depth must be greater than 0."
        root = self.name_to_vertex[str(instance)]
        to_explore = {root}
        paths = [[root]]

        print (f"added path: {[n.name for n in paths[0]]} to paths")

        for d in range(depth + 1):
            new_explore = set()
            pref_paths = [] + paths
            for v in list(to_explore):
                for pref_path in pref_paths:
                    pref_path_ent_neighborhood = self.get_neighbors(pref_path[-1])
                    #print(f"Neighbors of {pref_path[-1].name}: {[n.name for n in pref_path_ent_neighborhood]}")
                    is_v_in_neighbors_of_pref_path_end = sum(1 if n == v else 0 for n in pref_path_ent_neighborhood)
                    if  is_v_in_neighbors_of_pref_path_end > 0:
                        paths.append(pref_path + [v])
                        #print(f"added path: {[n.name for n in pref_path + [v]]} to paths")
                        #print(f"{v.name} is in the neighbors of {pref_path[-1].name} (which is end of path {[n.name for n in pref_path]})")

                for neighbor in self.get_neighbors(v):
                    new_explore.add(neighbor)
            to_explore = new_explore

        # remove predicate nodes from paths:
        paths = [[v for v in p if not v.predicate] for p in paths]

        # make paths unique:
        paths = [list(x) for x in set(tuple(x) for x in paths)]

        return paths

    def graph_to_neo4j(self, uri='bolt://localhost', user='neo4j', password='password'):
        '''
        Converts the graph to a neo4j database. Needs an empty running neo4j db.
        :param uri: address where neo4j db is running
        :param user: username of neo4j db
        :param password: password of neo4j db
        :return: None
        '''

        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError("Please install the neo4j-driver package to use this function.")
        from tqdm import tqdm

        use_nodes_for_predicates = True # if false, the predicates are used as edges. Otherwise as nodes.
        relation_name = 'R'

        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            # check if db is empty:
            node_count = session.run("MATCH (n) return count(n)").single().value()
            if node_count > 0:
                print("Neo4j database is not empty, aborting graph to neo4h db convertion to avoid data loss.")
                return

            for v in self.vertices:
                if not v.predicate:
                    # name = v.name.split('/')[-1]
                    name = v.name.replace("'", "")
                    session.run(f"CREATE (a:Node" + (":RelationModified" if v.relation_modified else "") +
                                " {name: '" + name + "'})")  # .split(' ')[0] + '_' + vertex.__hash__()

            for v in tqdm(self.vertices):
                if not v.predicate:
                    # v_name = v.name.split('/')[-1]
                    v_name = v.name.replace("'", "")

                    node_type = "Node" + (":RelationModified" if v.relation_modified else "")

                    ids_v = [r["id(v)"] for r in
                             session.run(
                                 "MATCH (v:" + node_type + " {name: '" + v_name + "'}) where not (v:Predicate) RETURN id(v)")]
                    if len(ids_v) == 0:
                        raise Exception(f"no id found for {v_name}")
                    elif len(ids_v) == 1:
                        id_v = ids_v[0]
                    else:
                        raise Exception(f"multiple ids found for {v_name}: {ids_v}")

                    for pred in self.get_neighbors(v):

                        if pred.predicate:
                            pred_name = "".join(
                                [c for c in pred.name.split('/')[-1].replace("#", "_").replace('-', '_') if
                                 not c.isdigit()])
                            pred_name = pred_name[1:] if pred_name[0] in ["_", "-"] else pred_name

                            for obj in self.get_neighbors(pred):
                                # obj_name = obj.name.split('/')[-1]
                                obj_name = obj.name.replace("'", "")

                                ids_obj = [r["id(obj)"] for r in
                                           session.run(
                                               "MATCH (obj:Node {name: '" + obj_name + "'}) where not (obj:Predicate) RETURN id(obj)")]
                                if len(ids_obj) == 0:
                                    raise Exception(f"no id found for {obj_name}")
                                elif len(ids_obj) == 1:
                                    id_obj = ids_obj[0]
                                else:
                                    raise Exception(f"multiple ids found for {obj_name}: {ids_obj}")

                                if use_nodes_for_predicates:
                                    q = (f"MATCH (a), (b) WHERE ID(a)={id_v} AND ID(b)={id_obj} "
                                         "MERGE (a)-[:") + relation_name + "]->(c:Predicate {name: '" + pred_name + "'})-[:" + relation_name + "]->(b)"
                                else:
                                    q = f"MATCH (a), (b) WHERE ID(a)={id_v} AND ID(b)={id_obj} MERGE (a)-[:" + pred_name + "]->(b)"
                                session.run(q)

                        else:
                            obj_name = pred.name.replace("'", "")

                            ids_obj = [r["id(obj)"] for r in
                                       session.run(
                                           "MATCH (obj:Node {name: '" + obj_name + "'}) RETURN id(obj)")]
                            if len(ids_obj) == 0:
                                raise Exception(f"no id found for {obj_name}")
                            elif len(ids_obj) == 1:
                                id_obj = ids_obj[0]
                            else:
                                raise Exception(f"multiple ids found for {obj_name}: {ids_obj}")

                            q = f"MATCH (a), (b) WHERE ID(a)={id_v} AND ID(b)={id_obj} MERGE (a)-[:" + relation_name + "]->(b)"
                            session.run(q)

        driver.close()

    def graph_to_rdf(self, save_path):
        '''
        Converts the graph to an rdf file.
        :param save_path: path where the rdf file should be saved
        :return: None
        '''

        relation_name = 'R'

        g = rdflib.Graph()

        for v in self.vertices:
            if not v.predicate:
                s = rdflib.URIRef(v.name)
                g.add((s, rdflib.URIRef("rdf:type"), rdflib.URIRef("owl:NamedIndividual")))

        for v in self.vertices:
            if not v.predicate:
                s = rdflib.URIRef(v.name)
                for pred in self.get_neighbors(v):
                    if pred.predicate:
                        p = rdflib.URIRef(pred.name)
                        for obj in self.get_neighbors(pred):
                            if not obj.predicate:
                                o = rdflib.URIRef(obj.name)
                                g.add((s, p, o))
                    elif pred.relation_modified: # for rtm nodes
                        p = rdflib.URIRef(relation_name)
                        o = rdflib.URIRef(pred.name)
                        g.add((s, p, o))
                    else:
                        p = rdflib.URIRef(relation_name)
                        o = rdflib.URIRef(pred.name)
                        g.add((s, p, o))
        g.serialize(destination=save_path, format='n3')


    @staticmethod
    def rdflib_to_graph(rdflib_g, label_predicates=[], relation_tail_merging=False, skip_literals=False,
                        skip_nodes_with_prefix=[]):
        '''
        Converts an rdflib graph to a Graph object.
        During the conversion, a multi-relation graph (head)-[relation]->(tail) (aka subject, predicate, object)is converted to a non-relational graph.
        e.g. converting it to (head)-->(relation)-->(tail), or, if apply_relation_tail_merging is True, to (head)-->(relation_tail).

        :param rdflib_g: An rdflib graph, e.g. loaded with rdflib.Graph().parse('file.n3')
        :param label_predicates: a list of predicates that are used as labels, and should not be converted to edges?
        :param relation_tail_merging: If true, relation-tail-merging is applioed, as described in the paper
        "Investigating and Optimizing MINDWALC Node Classification to Extract Interpretable DTs from KGs":
        The process of relation-tail merging works as follows: First, a specific tail node is
        selected, t, as well as a set of nr relations of identical type, r, where the topological
        form (*)-r->(t) is given. The process of relation-tail merging then involves inserting
        a new node, rt, so that (*)-r->(t) turns into (*)-->(rt)-->(t). The new directional
        edges, -->, are now typeless, and the new inserted node, rt, represents a relationmodified node and is
        named accordingly in the form <type_of_r>_<name_of_t>.
        :param skip_literals: If True, literals (=node properties/attributes) are skipped during the conversion.
        Otherwise, they are converted to nodes. so that a node (n: {name: 'John'}) becomes (n)-->(name)-->(john).
        :param skip_nodes_with_prefix: A list of prefixes that are used to skip nodes with that prefix.
        :return: A Graph object of type datastructures::Graph
        '''

        kg = Graph()

        for (s, p, o) in rdflib_g:
            if p not in label_predicates:

                if skip_literals and (isinstance(o, rdflib.term.Literal) or isinstance(s, rdflib.term.Literal)):
                    continue

                if any([str(s).startswith(prefix) for prefix in skip_nodes_with_prefix]):
                    continue

                if any([str(o).startswith(prefix) for prefix in skip_nodes_with_prefix]):
                    continue

                # Literals are attribute values in RDF, for instance, a person’s name, the date of birth, height, etc.
                if isinstance(s, rdflib.term.Literal) and not str(s):
                    s = "EmptyLiteral"
                if isinstance(p, rdflib.term.Literal) and not str(p):
                    p = "EmptyLiteral"
                if isinstance(o, rdflib.term.Literal) and not str(o):
                    o = "EmptyLiteral"

                s = str(s)
                p = str(p)
                o = str(o)

                s_v = Vertex(s)

                if relation_tail_merging:
                    o_v_relation_mod = Vertex(f"{p}_MODIFIED_{o}", relation_modified=True)
                    o_v = Vertex(o)
                    kg.add_vertex(s_v)
                    kg.add_vertex(o_v_relation_mod)
                    kg.add_vertex(o_v)
                    kg.add_edge(s_v, o_v_relation_mod)
                    kg.add_edge(o_v_relation_mod, o_v)
                else:
                    o_v = Vertex(o)
                    p_v = Vertex(p, predicate=True, _from=s_v, _to=o_v)
                    kg.add_vertex(s_v)
                    kg.add_vertex(p_v)
                    kg.add_vertex(o_v)
                    kg.add_edge(s_v, p_v)
                    kg.add_edge(p_v, o_v)
        return kg


class Neighborhood(object):
    def __init__(self):
        self.depth_map = defaultdict(set)

    def find_walk(self, vertex, depth): # new!!!
        '''
        Check if a vertex is in the neighborhood at a certain depth
        If depth is None, check if the vertex is in the neighborhood at any depth
        '''
        if type(depth) is tuple: # if walc_depth is flexible:
            # check in depth_map if the vertex can be found at any depth between depth[0] and depth[1]
            for d in self.depth_map:
                if d in range(depth[0], depth[1] + 1) and vertex in self.depth_map[d]:
                    return True
            return False
        return vertex in self.depth_map[depth]


class Walk(object):
    def __init__(self, vertex, depth):
        self.vertex = vertex
        self.depth = depth

    def __eq__(self, other):
        return (hash(self.vertex) == hash(other.vertex)
                and self.depth == other.depth)

    def __hash__(self):
        return hash((self.vertex, self.depth))

    def __lt__(self, other):
        return (self.depth, self.vertex) < (other.depth, other.vertex)


class TopQueue:
    def __init__(self, size):
        self.size = size
        self.data = []

    def add(self, x, priority): # (vertex, depth), (ig, -depth)
        if len(self.data) == self.size:
            heapq.heappushpop(self.data, (priority, x))
        else:
            heapq.heappush(self.data, (priority, x))


class Tree():
    def __init__(self, walk=None, _class=None):
        self.left = None
        self.right = None
        self._class = _class
        self.walk = walk
        self.node_number = None #new

    def evaluate(self, neighborhood):
        if self.walk is None:
            return self._class

        if neighborhood.find_walk(self.walk[0], self.walk[1]):
            return self.right.evaluate(neighborhood)
        else:
            return self.left.evaluate(neighborhood)

    def evaluate_traced(self, neighborhood, path: list): # new!!!
        path.append(self)
        if self.walk is None:
            return path
        if neighborhood.find_walk(self.walk[0], self.walk[1]):
            self.right.evaluate_traced(neighborhood, path)
        else:
            self.left.evaluate_traced(neighborhood, path)

    @property
    def node_count(self):
        '''Returns the number of nodes in the tree'''
        left_count, right_count = 0, 0
        if self.left is not None:
            left_count = self.left.node_count
        if self.right is not None:
            right_count = self.right.node_count
        return 1 + left_count + right_count



    def visualise(self, output_path, _view=True, node_properties_function=None, meta_infos='', as_pdf=True):
        """Visualise the tree with [graphviz](http://www.graphviz.org/),
         using `decisiontree.DecisionTree.convert_to_dot`
        **Params**
        ----------
          - `output_path` (string) - where the file needs to be saved
          - `_view` (boolean) - open the pdf after generation or not
          - `node_properties_function` (function) - function (params: xxx). Passed function returns a
            dict which contains the properties of the generated graph,
            using the Graphviz dot languade. see self._default_tree_visualization as example!
          - `meta_infos` (string) - a string, usually containing some usefully infos about the tree
            (which paramse used, which dataset uses ...)
        **Returns**
        -----------
            a pdf with the rendered Graphviz dot code of the tree
        """
        if not node_properties_function:
            node_properties_function = self._default_node_visualization
        dot_code = self.convert_to_dot(node_properties_function, infos=meta_infos)
        if as_pdf:
            src = Source(dot_code)
            src.render(output_path, view=_view)
        else:
            with open(output_path, 'w') as f:
                f.write(dot_code)

    def convert_to_dot(self, label_func, font='Times-Roman', infos=''):
        """Converts a decision tree object to DOT code
        **Params**
        ----------
          - ...
        **Returns**
        -----------
            a string with the dot code for the decision tree
        """
        self.nummerate_nodes_of_tree()
        s = 'digraph DT{\n'
        s += f'label="{infos}"\nfontname="{font}"\n'
        s += f'node[fontname="{font}"];\n'
        s += self._convert_node_to_dot(label_func)
        s += '}'
        return s

    def _convert_node_to_dot(self, node_vis_props):
        """Convert node to dot format in order to visualize our tree using graphviz
        :param count: parameter used to give nodes unique names
        :return: intermediate string of the tree in dot format, without preamble (this is no correct dot format yet!)
        """

        num = self.node_number
        if self._class:  # leaf node:
            node_render_props = node_vis_props(self)
            node_props_dot = str([f'{k}="{node_render_props[k]}"' for k in node_render_props.keys()]).replace("'", '')
            s = f'Node{str(num)} ' + node_props_dot + ';\n'
        else:  # decision node:
            node_render_props = node_vis_props(self)
            node_props_dot = str([f'{k}="{node_render_props[k]}"' for k in node_render_props.keys()]).replace("'", '')
            s = f'Node{str(num)} ' + node_props_dot + ';\n'
            s += self.left._convert_node_to_dot(node_vis_props)
            s += 'Node' + str(num) + ' -> ' + 'Node' + str(num + 1) + ' [label="not found"];\n'
            amount_subnodes_left = self.left.node_count
            s += self.right._convert_node_to_dot(node_vis_props)
            s += 'Node' + str(num) + ' -> ' + 'Node' + str(num + amount_subnodes_left + 1) + ' [label="found"];\n'

        return s

    def nummerate_nodes_of_tree(self, count=1):
        self.node_number = count
        if not self._class:
            self.left.nummerate_nodes_of_tree(count=count + 1)
            amount_subnodes_left = self.left.node_count
            self.right.nummerate_nodes_of_tree(count=count + amount_subnodes_left + 1)

    def get_max_tree_depth(self, root):
        if root is None:
            return 0

        left = self.get_max_tree_depth(root.left)
        right = self.get_max_tree_depth(root.right)
        return max(left, right) + 1

    @property
    def max_tree_depth(self):
        return self.get_max_tree_depth(self)


    def _default_node_visualization(self, node):
        is_leaf_node = True if node._class else False
        if not is_leaf_node:
            if type(node.walk[1]) is tuple:
                #walking_depth_info = f'\nd = [{node.walk[1][0]}, {node.walk[1][1]}]'
                walking_depth_info = f', d = {node.walk[1]}'
            else:
                walking_depth_info = f', d = {int(node.walk[1])}'
        out = {'label': (node._class if is_leaf_node else node.walk[0] + walking_depth_info),
               'fillcolor': '#DAE8FC' if is_leaf_node else '#FFF2CC',
               'color': '#6C8EBF' if is_leaf_node else "#D6B656",
               'style': "rounded,filled",
               'shape': 'ellipse' if is_leaf_node else 'box'}
        return out


if __name__ == '__main__': # TODO: add this example to the MINDWALC repo?

    # create simple example graph:
    g = Graph()
    start_node = Vertex('A')
    g.add_vertex(start_node)
    g.add_vertex(Vertex('B', predicate=True))
    g.add_vertex(Vertex('C', predicate=True))
    g.add_vertex(Vertex('D'))
    target_node = Vertex('E')
    g.add_vertex(target_node)
    g.add_vertex(Vertex('F', predicate=True))
    g.add_edge(g.name_to_vertex['A'], g.name_to_vertex['B'])
    g.add_edge(g.name_to_vertex['A'], g.name_to_vertex['C'])
    g.add_edge(g.name_to_vertex['A'], g.name_to_vertex['F'])
    g.add_edge(g.name_to_vertex['B'], g.name_to_vertex['D'])
    g.add_edge(g.name_to_vertex['B'], g.name_to_vertex['E'])
    g.add_edge(g.name_to_vertex['C'], g.name_to_vertex['E'])

    # visualize the graph:
    g.visualise()

    d = 1

    for p in g.extract_paths('A', d):
        print([v.name for v in p])

    print()

    print(g.extract_neighborhood('A', d).depth_map)

    exit()