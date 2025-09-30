# Causal DAG (CASTLE)
Python script to learn a causal graph (DAG) from tabular data stored in Excel using the [python-castle] library. Supports multiple algorithms (PC, GES, GOLEM, ICALiNGAM), saves the learned adjacency matrix and a DAG plot, and lets you choose the variables/columns to include.

Arguments:

--input Path to .xlsx file

--cols Variable names to include (space-separated). If omitted, use --col-indexes

--col-indexes Zero-based column indexes (alternative to --cols)

--algo One of: pc, ges, golem, icalingam (default: pc)

--backend CASTLE backend, e.g., pytorch (default: pytorch)

--out-matrix Output CSV for adjacency (default: outputs/adjacency.csv)

--out-graph Output PNG for DAG plot (default: outputs/dag.png)

--rename Optional node labels (space-separated), same length as selected variables
