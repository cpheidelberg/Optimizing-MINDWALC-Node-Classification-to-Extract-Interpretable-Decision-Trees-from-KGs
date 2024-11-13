import sys, os
import argparse
import requests
from requests.auth import HTTPBasicAuth


def cypher_to_rdf(cypher, save_path, addr, auth, format='text/n3'):
    '''

    This function sends a cypher query (which defines how to get a certain subgraph) to a neo4j database and saves the result as a rdf file.

    Requirements:
    - neo4j database with neosemantics plugin installed
    - in neo4j.conf: server.unmanaged_extension_classes=n10s.endpoint=/rdf

    :param cypher: Cypher query to select the part of the running neo4j database that should be saved as rdf file
    :param save_path: Path where the rdf file should be saved
    :param addr: Address of the neo4j database
    :param auth: Tuple with username and password for the neo4j database
    :param format: Format of the rdf file (default: 'text/n3')
    :return: None
    '''

    url = f'http://{addr}:7474/rdf/neo4j/cypher'

    basic = HTTPBasicAuth(auth[0], auth[1])
    json_cypher = {"cypher": cypher, "format": format}
    headers = {"Accept": "text/plain", "Content-Type": "application/json"}
    # print(f"requesting {format} file at {url} via cypher query\n{cypher}")
    resp = requests.post(url, headers=headers, auth=basic, json=json_cypher)
    if resp.status_code == 200:
        text = resp.text.replace('\n', '').replace('\r', '\n')
        f = open(save_path, 'w', encoding="utf-8")
        f.write(text)
        f.close()
        # print(f"saved at {save_path}")
    else:
        raise Exception(f'cypher_to_rdf FAILED with {resp.status_code}. Did you configure neo4h correctly?\n'
                        'This are the required configs:\n'
                        '- neo4j database with neosemantics plugin installed\n'
                        '- in neo4j.conf: server.unmanaged_extension_classes=n10s.endpoint=/rdf (newer neo4j versions) '
                        'OR dbms.unmanaged_extension_classes=n10s.endpoint=/rdf (older versions)')

def main():

    '''
    i labeled the corpus with these queries:

    match (a:Report) set a.label = 'neo4j://individuals#119845'
    match (a:Report)-[r:MENTIONS_IN_DIAGNOSE]->(b:Disorder) where b.sctid = '32916005' set a.label = 'neo4j://individuals#32815'
    match (a:Report) return 'neo4j://individuals#' + toString(id(a)) as report, a.label as label_disorder
    '''

    addr = "localhost"#"147.142.106.218"
    auth = ('neo4j', 'pathohd42')

    outfile = './MINDWALC/mindwalc/data/test.n3'

    format = 'n3'
    relations = 'MENTIONS_IN_DESCRIPTION|MENTIONS_IN_SAMPLEINFO|MENTIONS_IN_CLINICINFO'
    attribute_types = ['MorphologicAbnormality', 'CellStructure', 'Disorder', 'TumorStaging', 'Cell', 'BodyStructure']
    '''attribute_types = ['MorphologicAbnormality', 'BodyStructure', 'Disorder', 'CellStructure', 'Finding', 'Substance',
                           'ObservableEntity', 'Organism', 'Procedure', 'Occupation', 'MedicinalProduct', 'Cell',
                           'RegimeorTherapy', 'SpecialConcept', 'TumorStaging', 'Attribute']'''
    attribute_types = ['Disorder', 'MorphologicAbnormality']
    # attribute_types = ['MorphologicAbnormality', 'CellStructure', 'TumorStaging', 'Cell', 'BodyStructure']

    attribute_types_as_string_b = '(' + ' OR '.join(
        ['b:' + c for c in attribute_types]) + ')' if attribute_types else 'b:ObjectConcept'

    attribute_types_as_string_c = '(' + ' OR '.join(
        ['c:' + c for c in attribute_types]) + ')' if attribute_types else 'c:ObjectConcept'

    subgraph_cypher = f'match (a:Report)-[r:{relations}]->(b)-[r2]->(c) ' \
             f'where {attribute_types_as_string_b} and {attribute_types_as_string_c} ' \
             f'return *'
            # and not b.sctid = "32916005"

    cypher_to_rdf(subgraph_cypher, outfile, addr, auth, format)

if __name__ == '__main__':
    main()