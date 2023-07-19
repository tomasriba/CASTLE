import pandas as pd
import os
os.environ['CASTLE_BACKEND'] = 'pytorch'
from collections import OrderedDict
import numpy as np
import networkx as nx
import castle
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation, DAG
from castle.algorithms import PC, GES, ICALiNGAM, GOLEM
import matplotlib.pyplot as plt

# specify columns
 
require_cols = [0,1,2,3]
 
df = pd.read_excel('/Users/tomas/Documents/Python/Trial_pandas.xlsx', usecols = require_cols)

# convert variables to list

x = df['VAR1'].tolist()
y = df['VAR2'].tolist()
z = df['VAR3'].tolist()
w = df['VAR4'].tolist()

print('\n',x,'\n',y,'\n',z,'\n',w)

# To matrix

pc_dataset = np.vstack([x, y, z, w]).T
print(pc_dataset.shape)

# Model building

pc = PC()
pc.learn(pc_dataset)

# Print learned matrix

print(pc.causal_matrix)

learned_graph = nx.DiGraph(pc.causal_matrix)

# Relabel nodes

MAPPING = {k: v for k, v in zip(range(4), ['X', 'Y', 'Z', 'W'])}
learned_graph = nx.relabel_nodes(learned_graph, MAPPING, copy=True)

# Plot the graph

nx.draw(
    learned_graph, 
    with_labels=True,
    node_size=800,
    font_size=12,
    font_color='white'
)

plt.show()
