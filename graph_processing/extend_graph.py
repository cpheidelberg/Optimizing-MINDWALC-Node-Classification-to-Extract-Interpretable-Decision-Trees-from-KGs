from neo4j import GraphDatabase
import os, sys
import pandas as pd
from tqdm import tqdm
import time

'''
### what this script does ###
- Table with Head-, Relation-, and Tail-columns can be passed to add new nodes and relations to the given neo4j graph 
- It is also possible to add properties to the nodes with the given head- and tail-props columns in the table.
'''


### params:
# columns in table:
head_id_colum = 'Head-sctid'
relation_id_colum = 'Relation'
tail_id_colum = 'Tail-sctid'
head_name_colum = 'Head'
tail_name_colum = 'Tail'
head_props_colum = 'Head-props'
tail_props_colum = 'Tail-props'

# if set to false, only the required cypher queries will be logged and printed. But not commited to the neo4j db.
# The cypher querie log can be controlled in the stored log table. If everithink looks good, you can commit the changes.
instant_commit = True

# 15 new nodes, 63 new relations (15 new relation-types)
# need to add this to snomed graph: merge (a:ObjectConcept:QualifierValue {sctid: "1279783001", FSN: "Cribriform histologic pattern"})
in_table_path = 'data/subgraphs/prostate_subgraphs/prostate-graph-1.3.xlsx'

name_of_id_attribute = 'sctid'
type_of_id_attribute = "string"
name_of_name_attribute = 'FSN'
node_type = "ObjectConcept"
modification_symbol = "custom_prostata_concept"

clearing_queries = [
    f"MATCH (h)-[r]->(t) where r.{modification_symbol} delete r",
    f"MATCH (n) where n.{modification_symbol} delete n",
    "match (a:ProstateCenterNode) remove a:ProstateCenterNode",
    "match (a:ObjectConcept) remove a.prostate_label"
]

def node_id_post_processor(node_id_string):
    if type(node_id_string) == str:
        if "|" in node_id_string:
            return node_id_string.split("|")[0].replace(" ", "")
        else:
            return None
    else:
        return None

def merge_HRT_triblet_into_neo4jDB(head_id, relation_name, tail_id, session,
                                   head_name=None, tail_name=None,
                                   head_props=None, tail_props=None,
                                   modification_symbol = "modified"):
    '''
    Adds a new relation between head and tail with the given relation name to the neo4j db.
    If one of the passed nodes head_id or tail_id is None, the missing one will be added as new node to the neo4j db,
    using head_name or tail_name as name and id for the new node.
    WARNING: This function does not commit any changes to the neo4j db. Instead it returns the cypher query to do so.
    The modification cypher queries will allways mark each newly added relation or node with the given modification_symbol.
    So you can easily find all new nodes and relations in the neo4j db and remove them if needed.
    Querys to delete the modifications, if modification_symbol = "modified":
    MATCH (h)-[r]->(t) where r.modified delete r
    MATCH (n) where n.modified delete n

    :param head_id: The id of the head node.
    :param relation_name: The name of the relation between head and tail.
    :param tail_id: The id of the tail node.
    :param session: A neo4j session object.
    :return: modification_cypher, log (A cypher query to commit the changes to the neo4j db and a log string).
    '''

    log = ""
    mod_query = ""

    head_props = "" if type(head_props)!=str else head_props
    tail_props = "" if type(tail_props)!=str else tail_props

    if not head_id and not tail_id:
        # try to find the head and tail nodes by their names:
        if type(head_name)==str:
            query = f"MATCH (n:{node_type}) WHERE n.{name_of_id_attribute}='{head_name}' RETURN n.{name_of_id_attribute} AS id"
            results = [r["id"] for r in session.run(query)]
            if len(results) > 1:
                raise Exception(f"WARNING: Found more than one node with n.{name_of_id_attribute}='{head_name}'. "
                                f"This id should be unique. Please fix this!")
            elif len(results) == 1:
                head_id = results[0]
            else:
                head_id = None

        if type(tail_name)==str:
            query = f"MATCH (n:{node_type}) WHERE n.{name_of_id_attribute}='{tail_name}' RETURN n.{name_of_id_attribute} AS id"
            results = [r["id"] for r in session.run(query)]
            if len(results) > 1:
                raise Exception(f"WARNING: Found more than one node with n.{name_of_id_attribute}='{tail_name}'. "
                                f"This id should be unique. Please fix this!")
            elif len(results) == 1:
                tail_id = results[0]
            else:
                tail_id = None

        if not head_id and not tail_id:
            log += "Could not find Head_id or Tail_id. Skipping."
            return "", log

    if type(relation_name) != str:
        log += "WARNING: Relation-name is None.\n"

        # of no relation present, lets add the props to the nodes:
        if (type(head_id)==str and type(head_props)==str) or (type(tail_id)==str and type(tail_props)==str):
            if type(head_id)==str and type(head_props)==str:
                mod_query += f"MATCH (n:{node_type}) WHERE n.{name_of_id_attribute}='{head_id}' SET {head_props};\n"
                log += f"Added properties to head-node {head_id}: {head_props}\n"
            if type(tail_id)==str and type(tail_props)==str:
                mod_query += f"MATCH (n:{node_type}) WHERE n.{name_of_id_attribute}='{tail_id}' SET {tail_props};"
                log += f"Added properties to tail-node {tail_id}: {tail_props}"
        else:
            log += "ERROR: No node_id and node_propy is given. So i dont know what todo with that. Skipping."

        return mod_query, log

    if None in [head_id, tail_id]: # one of both in None:
        if head_id is None:
            log += f"Only Head is None. So we will add this node to the neo4j db.\n"
            if head_name is None or pd.isna(head_name):
                log += f"ERROR: Head-name is None. Cant add a new node without a name. Skipping."
                return "", log
            else:
                query = f"MATCH (n:{node_type}) WHERE n.{name_of_id_attribute} = '{tail_id}' RETURN n"
                result = session.run(query)
                if not result.single():
                    log += f"ERROR: Tail-Node with {name_of_id_attribute}='{tail_id}' does not exist in neo4j db! Skipping."
                    return "", log
            new_node_id = head_name.lower()
            new_node_name = head_name
            head_id = new_node_id

            # head will be created and tail does already exist. so lets get the name of tail:
            query = f"MATCH (n:{node_type}) WHERE n.{name_of_id_attribute}='{tail_id}' RETURN n.{name_of_name_attribute} AS name"
            tail_name = session.run(query).single()["name"]

        else:
            log += f"Only Tail is None. So we will add this node to the neo4j db.\n"
            if tail_name is None or pd.isna(tail_name):
                log += f"ERROR: Tail-name is None. Cant add a new node without a name. Skipping."
                return "", log
            else:
                query = f"MATCH (n:{node_type}) WHERE n.{name_of_id_attribute} = '{head_id}' RETURN n"
                result = session.run(query)
                if not result.single():
                    log += f"ERROR: Head-Node with {name_of_id_attribute}='{head_id}' does not exist in neo4j db! Skipping."
                    return "", log
            new_node_id = tail_name.lower()
            new_node_name = tail_name
            tail_id = new_node_id

            # tail will be created and head does already exist in db. so lets get the name of head:
            query = f"MATCH (n:{node_type}) WHERE n.{name_of_id_attribute}='{head_id}' RETURN n.{name_of_name_attribute} AS name"
            head_name = session.run(query).single()["name"]

        mod_query += f"MERGE (n:{node_type} " + "{" + f"{name_of_id_attribute}: '{new_node_id}', {name_of_name_attribute}: '{new_node_name}', {modification_symbol}: True" + "});\n"
        log += f"Added new node n with n.{name_of_id_attribute}='{new_node_id}' and n.{name_of_name_attribute}='{new_node_name}'.\n"
    else: # both ids are given:
        new_node_id = None
        # check if head and tail exists in neo4j graph:
        for node in [head_id, tail_id]:
            query = f"MATCH (n:{node_type}) WHERE n.{name_of_id_attribute} = '{node}' RETURN n"
            result = session.run(query)
            if not result.single():
                log += f"ERROR: Node with {name_of_id_attribute}='{node}' does not exist in neo4j db! Skipping."
                return "", log

        # get the names of head and tail:
        query = f"MATCH (n:{node_type}) WHERE n.{name_of_id_attribute}='{head_id}' RETURN n.{name_of_name_attribute} AS name"
        head_name = session.run(query).single()["name"]
        query = f"MATCH (n:{node_type}) WHERE n.{name_of_id_attribute}='{tail_id}' RETURN n.{name_of_name_attribute} AS name"
        tail_name = session.run(query).single()["name"]

    # check if there are already relation between head and tail:
    if new_node_id is None:
        query = f"MATCH (h:{node_type})-[r]-(t:{node_type}) WHERE h.{name_of_id_attribute}='{head_id}' AND t.{name_of_id_attribute}='{tail_id}' RETURN type(r), r, ID(h)"
        res_are_there_relations_between = [r for r in session.run(query)]
    else: # in this case, head or tail are newly added to the graph. So there will be no relation between them.
        res_are_there_relations_between = []

    if not res_are_there_relations_between:  # if no relation exists: add new relation
        # add new relation:
        mod_query += f"MATCH (h:{node_type}) WHERE h.{name_of_id_attribute}='{head_id}' " \
                    f"MATCH (t:{node_type}) WHERE t.{name_of_id_attribute}='{tail_id}' " \
                    f"MERGE (h)-[:{relation_name}"+" {"+f"{modification_symbol}: True"+"}]->(t);\n"

        log += f"Added relation ({head_id}|{head_name})-[{relation_name}]->({tail_id}|{tail_name}).\n"
    else:  # if relation exists: check if it is the same or a similar relation. If not add new relation.
        for r in res_are_there_relations_between:
            if r["type(r)"].lower() == relation_name.lower():
                log += f"Relation between {head_id}|'{head_name}' and {tail_id}|'{tail_name}' with relation {relation_name} already exists.\n"
            else:
                situation_description = f"Relation between ({head_id}|{head_name})-[{relation_name}]->({tail_id}|{tail_name}) does not exist yet.\n"
                if r["r"].start_node == r["ID(h)"]:
                    user_input_request = situation_description + f"But ({head_id}|{head_name})-[{r['type(r)']}]->({tail_id}|{tail_name}) does exist."
                else:
                    user_input_request = situation_description + f"But ({head_id}|{head_name})<-[{r['type(r)']}]-({tail_id}|{tail_name}) does exist."

                log += user_input_request + "\n"
                print(user_input_request, flush=True)
                cmd = input("Is it similar? (y/n): ").lower()

                if cmd == "n":
                    # add new relation:
                    mod_query += f"MATCH (h:{node_type}) WHERE h.{name_of_id_attribute}='{head_id}' " \
                                f"MATCH (t:{node_type}) WHERE t.{name_of_id_attribute}='{tail_id}' " \
                                f"MERGE (h)-[:{relation_name}"+" {"+f"{modification_symbol}: True"+"}]->(t);\n"

                    log += f"Is not similar. Added relation ({head_id}|{head_name})-[{relation_name}]->({tail_id}|{tail_name}).\n"

                else:
                    log += "Is similar enough. Relation not added.\n"

    # at the end, all nodes should exist, now we can add the props to them:
    if head_props:
        mod_query += f"MATCH (n:{node_type}) WHERE n.{name_of_id_attribute}='{head_id}' SET {head_props};\n"
        log += f"Added properties to head-node {head_id}: {head_props}\n"
    if tail_props:
        mod_query += f"MATCH (n:{node_type}) WHERE n.{name_of_id_attribute}='{tail_id}' SET {tail_props};"
        log += f"Added properties to tail-node {tail_id}: {tail_props}"

    return mod_query, log

def commit_cypher_query_list_to_neo4jDB(mod_query: str, session):
    '''
    Commits a list of cypher queries to the neo4j db.
    :param query_list: A list of cypher queries, given as string, seperated with ; symbol
    :param session: A neo4j session object.
    :return: None
    '''

    if mod_query and type(mod_query) == str:
        if ";" in mod_query:  # multiple queries?
            mod_query = mod_query.split(";")
            for q in [m.replace("\n", "") for m in mod_query]:
                if q and type(q) == str:
                    session.run(q)
        else:
            session.run(mod_query)

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

    # first check if there are some old modifications in the neo4j db:
    query = f"match (a) where a.{modification_symbol} return distinct a.FSN"
    node_count = len([r for r in session.run(query)])
    query = f"match (a)-[r]-(b) where r.{modification_symbol} return distinct ID(r)"
    relation_count = len([r for r in session.run(query)])
    if node_count > 0 or relation_count > 0:
        print(
            f"==> WARNING: There are already {node_count} nodes and {relation_count} relations with the modification symbol '{modification_symbol}' in the neo4j db.")
        print("==> Do you want to delete the old modifications before commiting the new ones? "
              "\n(each modification will be applied with MERGE, not CREATE, so duplications will be avoided)",
              flush=True)
        cmd = input("(y/n): ").lower()
        if cmd == "y":
            for q in clearing_queries:
                session.run(q)
            time.sleep(1)
            print("Old modifications deleted.")
        else:
            print("Old modifications not deleted.")

    # read in table:
    table = pd.read_excel(in_table_path)

    table["cypher-query"] = "" # lets log all cypher queries here to be able to rerun them if needed and to see what was done.
    table["log"] = ""

    # add nodes and relations to neo4j:
    for i, row in table.iterrows():
        print(f"{i+2})")
        head_id = node_id_post_processor(row[head_id_colum])
        relation = row[relation_id_colum]
        tail_id = node_id_post_processor(row[tail_id_colum])

        head_name = row[head_name_colum]
        tail_name = row[tail_name_colum]

        head_props = row[head_props_colum]
        tail_props = row[tail_props_colum]

        # if the row is empty, continue
        if pd.isna(head_id) and pd.isna(relation) and pd.isna(tail_id) and pd.isna(head_name) and pd.isna(tail_name):
            print("Empty row. Skipping.")
            continue

        # head and tail name postprocessing:
        if type(head_name) == str:
            head_name = head_name.lower()
        if type(tail_name) == str:
            tail_name = tail_name.lower()

        # relation-string postprocessing
        if type(relation) == str:
            relation = relation.upper()
            replacements = [(" ", "_"), ("-", "_"), ("/", "_OR_"), (",", "_")]
            for r in replacements:
                relation = relation.replace(r[0], r[1])
            while relation[-1] == "_":
                relation = relation[:-1]
            while relation[0] == "_":
                relation = relation[1:]

        mod_query, log = merge_HRT_triblet_into_neo4jDB(head_id, relation, tail_id, session,
                                                        head_name=head_name, tail_name=tail_name,
                                                        head_props=head_props, tail_props=tail_props,
                                                        modification_symbol=modification_symbol)

        if instant_commit:
            commit_cypher_query_list_to_neo4jDB(mod_query, session)

        table.at[i, "cypher-query"] = mod_query
        table.at[i, "log"] = log
        print(f"log:\n{log}\nquery:\n{mod_query}\n")

    #save cypher log modified tyble:
    table.to_excel(in_table_path.replace(".xlsx", "_log.xlsx"), index=False)
    print(f"==> Done.\nSaved logs into {in_table_path.replace('.xlsx', '_log.xlsx')}.")

    if not instant_commit:
        print("Commit changes to connected neo4j db?", flush=True)
        cmd = input("(y/n): ").lower()
        if cmd == "y":

            print("Commiting changes to neo4j db...", flush=True)
            for i, row in tqdm(table.iterrows(), total=len(table)):
                mod_query = row["cypher-query"]
                commit_cypher_query_list_to_neo4jDB(mod_query, session)

            time.sleep(1)
            print("==> Done.")
        else:
            print("==> Done. Changes not commited to neo4j.")

    query = f"match (a) where a.{modification_symbol} return distinct a.FSN"
    node_count = len([r for r in session.run(query)])
    query = f"match (a)-[r]-(b) where r.{modification_symbol} return distinct ID(r)"
    relation_count = len([r for r in session.run(query)])
    query = f"match (a)-[r]-(b) where r.{modification_symbol} return distinct type(r)"
    relationtype_count = len([r for r in session.run(query)])
    print(f"\nNow there are {node_count} custom nodes, \n"
          f"{relation_count} custom relations ({relationtype_count} custom relation-types)\n"
          f"with the modification symbol '{modification_symbol}' in the neo4j db.")
    print(f"\nTipp:\nTo undo the commited changes, run the following cypher queries:\n"
          f"MATCH (h)-[r]->(t) where r.{modification_symbol} delete r\n"
          f"MATCH (n) where n.{modification_symbol} delete n")

    # close connection:
    session.close()
    driver.close()

    exit()