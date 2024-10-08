# Optimizing-MINDWALC-Node-Classification-to-Extract-Interpretable-Decision-Trees-from-KGs

[![image.png](./title-img.png)](./title-img.png) 

This python project was created as part of the article: \
"_Investigating and Optimizing MINDWALC Node Classification to Extract Interpretable Decision Trees from Knowledge Graphs_".

The paper can be found [here](https://) (link commming soon). \
The MINDWALC optimizations proposed in the article are implemented in the original [MINDWALC repo](https://github.com/predict-idlab/MINDWALC). \
This Project contains the scripts used to generate the results in the article.

## Requirements and Installation

### 1) Install MINDWALC

To use this project, you need to install the latest MINDWALC package. \
Currently, MINDWALC is not available on PyPi, so you need to install it manually.

For this, please first clone the [MINDWALC repo](https://github.com/predict-idlab/MINDWALC) into another directory.\
To avoid git issues, delete the ````.git```` folder in the MINDWALC directory. \
Then, copy the MINDWALC directory into the root directory of this project. \
Check the ````MINDWALC/README.md```` for installation instructions and make sure that the MINDWALC 
is working properly by running the example script in ```MINDWALC/mindwalc/Example (AIFB).ipynb```.

### 2) Install the required python packages

Create a new environment, then install the required python packages with: \
```pip install -r requirements.txt```

### 3) Install Neo4j Desktop

All your tests scripts do process graphs stored as Neo4j databases, so you need to install Neo4j Desktop. 

So please download and install Neo4j Desktop on your OS ([https://neo4j.com/download/](https://neo4j.com/download/)).

## How to get the graph-datasets

### ProstateToyGraph

This graph is based on the Snomed CT ontology. Therefore, we cannot provide the graph directly. 
However, if you have access to Snomed CT, you can generate the graph yourself.

This are the necessary steps:
1. Generate a locally running Snomed CT Neo4j graph database. For this, please follow the guide provided at [./graph_processing/snomed_graph_installation/README.md](./graph_processing/snomed_graph_installation/README.md).
2. Next, run the script [./graph_processing/ProstateToyGraph/extend_graph.py](./graph_processing/ProstateToyGraph/extend_graph.py) to extend the graph with the prostate cancer related nodes and edges. It will also mark some nodes with label `ProstateCenterNode`. \
Attention: The execution-terminal will ask you several question about how to handle certain conflicts. To create the exact same dataset as in our paper, always type '`n`' for '`Is it similar?`' questions and '`y`' for '`Do you want to add A) to the graph ...`' questions.
3. Next. run the script [./graph_processing/centernode_based_subgraph_generation.py](./graph_processing/centernode_based_subgraph_generation.py) to extract a smaller part of the whole snomed graph that is centered around the prostate cancer related nodes of type `ProstateCenterNode`. \
This way smaller graph will be easier to handle, explore and process. Also here, the execution-terminal will sometimes stop to give you some instructions (copy paste some files, launch new neo4j db) and wait for your input.
4. Finally, run the script [./graph_processing/ProstateToyGraph/add_instances_to_graph.py](./graph_processing/ProstateToyGraph/add_instances_to_graph.py) to randomly add synthetic labeled case-instance-nodes to the graph.

### GottaGraphEmAll Pokemon Graph
... coming soon...

### TreeOfLife Pokemon Graph
... coming soon...

### Combined Pokemon graph
... coming soon...

## Usage

To run our experiments, you first need to start the corresponding Neo4j graph (e.g. firtst start the neo4j db of the dataset xxx to run tests on this db).

Then, configure (edit global variables in script) 
and run the script [node_classification/RRR_node_classification.py](node_classification/RRR_node_classification.py)
which stepwise destroys the "Instance Knowledge" by randomly removing the relations between instance-nodes and its neighbors.
(Process is called RTM - Relation Tail Merging, aka RRR - Random Relation Removement). \
On Each destruction-step, the script will run the MINDWALC node classification algorithm (according to your configuration) 
on the graph and store the results in a structured way (see console output for details).

Then, to plot the same curves as shown in the paper, configure and run the script [node_classification/RRR_node_clfs_plot.py](node_classification/RRR_node_clfs_plot.py) accordingly.

## citation

... coming soon...



