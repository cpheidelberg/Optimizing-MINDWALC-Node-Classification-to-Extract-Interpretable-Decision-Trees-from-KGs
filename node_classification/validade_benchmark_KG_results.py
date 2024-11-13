import os, sys
import json
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np


dataset = "aifb3" # mutag3 mutag4 bgs3 aifb3 aifb_old am
dataset_path = f"MINDWALC/mindwalc/experiments/{dataset}"
clf_methods = ["tree", "forest", "transform_lr", "transform_rf"]
walking_strategy = ["fix", "flex", "both"]
metrics = ['preds', '_fit_time_', '_params_', '_depth_']

f1_scores = {}
accuracies = {}

# list all json files in the directory:
files_to_process = [f for f in os.listdir(dataset_path)]

in_paths = [f'{dataset_path}/{f}' for f in files_to_process if f.endswith(".json")]

print(f"files to process: {in_paths}")

def latex_code_cleaner(code):
    replacement_list = [('±', '$\pm$'), ('\\toprule', '\\hline'), ('\\midrule', '\\hline'), ('\\bottomrule', '\\hline'),
                        ('_', '\\_'),
                        ('fix &', 'fix\t\t\t&'), ('fix + rtm ', 'fix + rtm\t'),
                        ('flex &', 'flex\t\t&'), ('flex + rtm ', 'flex + rtm\t'),
                        ('both &', 'both\t\t&'), ('both + rtm ', 'both + rtm\t')
                        ]
    for replace_a, replace_b in replacement_list:
        code = code.replace(replace_a, replace_b)
    return code

for in_path in in_paths:
    print(f"\n=== {in_path} ===")
    # load json file:
    try:
        with open(in_path, 'r') as f:
            data = json.load(f)
    except json.decoder.JSONDecodeError as e:
        print(f'\nError while reading {in_path}: {e}')
        print(f"trying to fix the formation of the file...")
        with open(in_path, 'r') as f:
            data_txt = f.read()

        # remove trailing commas
        data_txt = data_txt.replace("'", '"')
        with open(in_path.replace('.json', '_fixed.json'), 'w') as f:
            f.write(data_txt)

        with open(in_path.replace('.json', '_fixed.json'), 'r') as f:
            data = json.load(f)

        print("success!\n")

    file_name = in_path.split('/')[-1].replace('.json', '')

    is_rtm = True if 'rtm' in file_name.lower() else False

    ground_truth = data['ground_truth']

    for clf_method in clf_methods:
        for i_walk_meth, walking in enumerate(walking_strategy):

            try:
                preds = data[f'{clf_method}_preds{"_" + walking if walking else ""}']
            except:
                continue

            try:
                time = data[f'{clf_method}_fit_time_{walking}']
            except:
                time = None

            try:
                params = data[f'{clf_method}_params_{walking}']
            except KeyError:
                params = None

            try:
                depth = data[f'{clf_method}_depth_{walking}']
            except KeyError:
                depth = None

            # calculate metrics
            acc = accuracy_score(ground_truth, preds)
            f1 = f1_score(ground_truth, preds, average='weighted')

            #acc = round(accuracy_score(ground_truth, preds), 3)
            #f1 = round(f1_score(ground_truth, preds, average='weighted'), 3)

            print(f'{clf_method} - {walking} - acc: {acc} - f1: {f1} - time: {time} - params: {params if params is not None else f"depth: {depth}"}')

            key_name = f'{clf_method}_{walking}' + (' + rtm' if is_rtm else '')

            if key_name not in f1_scores:
                f1_scores[key_name] = []
                accuracies[key_name] = []

            f1_scores[key_name].append(f1)
            accuracies[key_name].append(acc)

for clf_method in clf_methods:
    for walking in walking_strategy:
        for with_rtm in [True, False]:

            key_name = f'{clf_method}_{walking}' + (' + rtm' if with_rtm else '')

            if key_name in f1_scores:
                f1_array = f1_scores[key_name]
                acc_array = accuracies[key_name]

                f1_scores[key_name] = np.mean(f1_array)
                accuracies[key_name] = np.mean(acc_array)

                # calc std:
                f1_scores[key_name + '_std'] = np.std(f1_array)
                accuracies[key_name + '_std'] = np.std(acc_array)

# re-arrenge the data to be printed in a table using walking_strategy as index
out_table_f1 = {}
out_table_acc = {}
factor = 100.0

for clf_method in clf_methods:
    index_acc = []
    index_f1 = []
    for i_walk_meth, walking in enumerate(walking_strategy):

        for with_rtm in [False, True]:

            k = f'{clf_method}_{walking}' + (' + rtm' if with_rtm else '')

            if k in f1_scores:
                if f'{clf_method}' not in out_table_f1:
                    out_table_f1[f'{clf_method}'] = []
                    out_table_acc[f'{clf_method}'] = []

                out_table_f1[f'{clf_method}'].append(f"{f1_scores[k]*factor:.2f} ± {f1_scores[k + '_std']*factor:.2f}")
                out_table_acc[f'{clf_method}'].append(f"{accuracies[k]*factor:.2f} ± {accuracies[k + '_std']*factor:.2f}")

                index_f1.append(k.replace(f'{clf_method}_', ''))
                index_acc.append(k.replace(f'{clf_method}_', ''))

print(out_table_acc)

df_f1 = pd.DataFrame(out_table_f1, index=index_f1)
df_acc = pd.DataFrame(out_table_acc, index=index_acc)

print(f"\n\n=== f1 {dataset_path} ====\n")
print(latex_code_cleaner(df_f1.to_latex()))
print(f"\n\n=== accuracy {dataset_path} ====\n")
print(latex_code_cleaner(df_acc.to_latex()))

