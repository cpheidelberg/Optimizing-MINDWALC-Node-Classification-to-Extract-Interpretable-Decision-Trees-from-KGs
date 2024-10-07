from graphdatascience import GraphDataScience
from tqdm import tqdm

# params:
NEO4J_URI = "bolt://localhost:7687"
auth = ("neo4j", "password")

gds = GraphDataScience(NEO4J_URI, auth=auth)

print(f'collecting RoleGroup nodes...')
group_node_ids = [x[1].id for x in gds.run_cypher('match (a:RoleGroup) return ID(a) as id').iterrows()]

print("interweaving new connections...")
for g_id in tqdm(group_node_ids):
    q_group = f"match (a)-[r_in]->(g)-[r_out]->(b) where ID(g) = {g_id} " \
        f"return r_out.sctid as r_out_sctid, r_out.rolegroup as r_out_rolegroup, type(r_out) as r_out_type, " \
        f"ID(a) as a_id, ID(b) as b_id"
    results_group = gds.run_cypher(q_group)

    for i, r in enumerate(results_group.iterrows()):
        a_id = r[1].a_id

        r_out_sctid = r[1].r_out_sctid
        r_out_rolegroup = r[1].r_out_rolegroup
        r_out_type = r[1].r_out_type

        b_id = r[1].b_id

        # create new connections without roulegroup-nodes:
        new_rel_props = f'sctid: {r_out_sctid}, rolegroup: {r_out_rolegroup}'
        q = f'match (a), (b) where ID(a) = {a_id} and ID(b) = {b_id} merge ' \
            f'(a)-[r:{r_out_type}'+' {'+new_rel_props+'}]->(b)'

        gds.run_cypher(q)

    q = f'match (a)-[r_in]->(g:RoleGroup)-[r_out]->(b) where id(g) = {g_id} delete r_in, r_out, g'
    gds.run_cypher(q)