import os, sys
import yaml


base_path = 'data/SNOMED_DTs_2/AllIgaAmyReports/deepL-translated'
#base_path = 'data/KBC_DTs_1/AllIgaAmyReports/deepL-translated'

base_config_file = 'node_classification/configs/IgaAmyReports.yaml'

with_cross_validation = True

# for SNOMED tests:
configs_to_test = [
    {
        'subgraph_name': 'SnomedDiagnoseDisorder',
        'relations_to_disconnect': ['EXISTENCE_IN_CLINICINFO', 'EXISTENCE_IN_SAMPLEINFO', 'EXISTENCE_IN_DESCRIPTION'], # 'EXISTENCE_IN_CLINICINFO', 'EXISTENCE_IN_SAMPLEINFO', 'EXISTENCE_IN_DESCRIPTION', 'EXISTENCE_IN_DIAGNOSE'
        'node_types_to_consider': ['Context', 'Disorder'], # 'ObjectConcept', 'MorphologicAbnormality', 'ObservableEntity', 'Finding', 'Disorder', 'Procedure', 'Cell'
        'base_path': base_path,
        'fold_amount': 10 if with_cross_validation else None
    },
    {
        'subgraph_name': 'SnomedDiagnoseObjectConcept',
        'relations_to_disconnect': ['EXISTENCE_IN_CLINICINFO', 'EXISTENCE_IN_SAMPLEINFO', 'EXISTENCE_IN_DESCRIPTION'], # 'EXISTENCE_IN_CLINICINFO', 'EXISTENCE_IN_SAMPLEINFO', 'EXISTENCE_IN_DESCRIPTION', 'EXISTENCE_IN_DIAGNOSE'
        'node_types_to_consider': ['Context', 'ObjectConcept'], # 'ObjectConcept', 'MorphologicAbnormality', 'ObservableEntity', 'Finding', 'Disorder', 'Procedure', 'Cell'
        'base_path': base_path,
        'fold_amount': 10 if with_cross_validation else None
    },
    {
        'subgraph_name': 'SnomedDescriptionObjectConcept',
        'relations_to_disconnect': ['EXISTENCE_IN_CLINICINFO', 'EXISTENCE_IN_SAMPLEINFO', 'EXISTENCE_IN_DIAGNOSE'], # 'EXISTENCE_IN_CLINICINFO', 'EXISTENCE_IN_SAMPLEINFO', 'EXISTENCE_IN_DESCRIPTION', 'EXISTENCE_IN_DIAGNOSE'
        'node_types_to_consider': ['Context', 'ObjectConcept'], # 'ObjectConcept', 'MorphologicAbnormality', 'ObservableEntity', 'Finding', 'Disorder', 'Procedure', 'Cell'
        'base_path': base_path,
        'fold_amount': 10 if with_cross_validation else None
    },
    {
        'subgraph_name': 'SnomedDescriptionMorphologicAbnormality',
        'relations_to_disconnect': ['EXISTENCE_IN_CLINICINFO', 'EXISTENCE_IN_SAMPLEINFO', 'EXISTENCE_IN_DIAGNOSE'], # 'EXISTENCE_IN_CLINICINFO', 'EXISTENCE_IN_SAMPLEINFO', 'EXISTENCE_IN_DESCRIPTION', 'EXISTENCE_IN_DIAGNOSE'
        'node_types_to_consider': ['Context', 'MorphologicAbnormality'], # 'ObjectConcept', 'MorphologicAbnormality', 'ObservableEntity', 'Finding', 'Disorder', 'Procedure', 'Cell'
        'base_path': base_path,
        'fold_amount': 10 if with_cross_validation else None
    },
    {
        'subgraph_name': 'SnomedDescription6Types',
        'relations_to_disconnect': ['EXISTENCE_IN_CLINICINFO', 'EXISTENCE_IN_SAMPLEINFO', 'EXISTENCE_IN_DIAGNOSE'], # 'EXISTENCE_IN_CLINICINFO', 'EXISTENCE_IN_SAMPLEINFO', 'EXISTENCE_IN_DESCRIPTION', 'EXISTENCE_IN_DIAGNOSE'
        'node_types_to_consider': ['Context', 'MorphologicAbnormality', 'ObservableEntity', 'Finding', 'Disorder', 'Procedure', 'Cell'], # 'ObjectConcept', 'MorphologicAbnormality', 'ObservableEntity', 'Finding', 'Disorder', 'Procedure', 'Cell'
        'base_path': base_path,
        'fold_amount': 10 if with_cross_validation else None
    },
]



# for KBC:
'''configs_to_test = [
    {
        'subgraph_name': 'KBCDiagnoseObjectConcept',
        'relations_to_disconnect': ['EXISTENCE_IN_CLINICINFO', 'EXISTENCE_IN_SAMPLEINFO', 'EXISTENCE_IN_DESCRIPTION'], # 'EXISTENCE_IN_CLINICINFO', 'EXISTENCE_IN_SAMPLEINFO', 'EXISTENCE_IN_DESCRIPTION', 'EXISTENCE_IN_DIAGNOSE'
        'node_types_to_consider': ['Context', 'ObjectConcept'], # 'ObjectConcept', 'MorphologicAbnormality', 'ObservableEntity', 'Finding', 'Disorder', 'Procedure', 'Cell'
        'base_path': base_path,
        'fold_amount': 10 if with_cross_validation else None
    },
    {
        'subgraph_name': 'KBCDescriptionObjectConcept',
        'relations_to_disconnect': ['EXISTENCE_IN_CLINICINFO', 'EXISTENCE_IN_SAMPLEINFO', 'EXISTENCE_IN_DIAGNOSE'], # 'EXISTENCE_IN_CLINICINFO', 'EXISTENCE_IN_SAMPLEINFO', 'EXISTENCE_IN_DESCRIPTION', 'EXISTENCE_IN_DIAGNOSE'
        'node_types_to_consider': ['Context', 'ObjectConcept'], # 'ObjectConcept', 'MorphologicAbnormality', 'ObservableEntity', 'Finding', 'Disorder', 'Procedure', 'Cell'
        'base_path': base_path,
        'fold_amount': 10 if with_cross_validation else None
    }
]'''

#configs_to_test = [configs_to_test[2]] # retrying crashed job(s)

# check if we are at correct working directory:
workdir = os.getcwd()
if not workdir[-len('Optimizing-MINDWALC-Node-Classification-to-Extract-Interpretable-Decision-Trees-from-KGs'):] == 'Optimizing-MINDWALC-Node-Classification-to-Extract-Interpretable-Decision-Trees-from-KGs':
    print(workdir + " is the wrong working directory.")
    print("please make shure to run this script with working directory '.../path/to/Optimizing-MINDWALC-Node-Classification-to-Extract-Interpretable-Decision-Trees-from-KGs'.")
    exit(1)


for changes in configs_to_test:
    changes['dt_label'] = base_path.split('/')[2] + '\\n' + base_path.split('/')[3] + '\\n' + changes['subgraph_name']
    print(changes['dt_label'])
    configuration = yaml.safe_load(open(base_config_file, 'r'))
    for key, value in changes.items():
        configuration[key] = value

    path_to_new_yaml = f"node_classification/configs/last_pipeline_run.yaml"

    with open(path_to_new_yaml, 'w') as file:
        yaml.dump(configuration, file)

    # run script node_classification/RRR_node_classification.py with new config
    script = f"python node_classification/RRR_node_classification.py pathohd42 '{path_to_new_yaml}'"
    print(f"\n######################################## {configuration['subgraph_name']} ###########################################")
    os.system(script)

