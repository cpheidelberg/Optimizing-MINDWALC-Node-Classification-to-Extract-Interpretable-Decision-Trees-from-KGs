import rdflib
from MINDWALC.mindwalc.datastructures import Graph
from neo4j import GraphDatabase
import os, sys

'''
This script does load a given n3 file, turns it into a MINDWALC Graph object (relation-tail-merging can be applied)
and then saves it into a running empty neo4j database.

This allowes us to investigate the graph structure in the neo4j browser.
'''

# params:
rdf_subgraph_file = "./data/RRR_node_clf/rrr_curve_prostate_subgraph_p3_0/RRR_0.0/subgraph.n3"
use_relation_tail_merging = True

# connect to neo4j dbms:
gdb_adress = 'bolt://localhost:7687'
try:
    pw = sys.argv[1]
except IndexError:
    raise ValueError("Please provide the password as first argument.")
auth = ('neo4j', pw)
driver = GraphDatabase.driver(gdb_adress, auth=auth)
session = driver.session(database="neo4j")

g = rdflib.Graph()
g.parse(rdf_subgraph_file, format='text/n3')
kg_non_rtm = Graph.rdflib_to_graph(g, relation_tail_merging=False)
kg_rtm = Graph.rdflib_to_graph(g, relation_tail_merging=True)

if use_relation_tail_merging:
    kg_rtm.graph_to_neo4j(password=pw)
else:
    kg_non_rtm.graph_to_neo4j(password=pw)