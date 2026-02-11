# ============================================
# LeakDB Full GNN Pipeline (Single File)
# ============================================

import zipfile
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# -------------------------------------------------
# 1. Load ENTIRE LeakDB (all scenarios)
# -------------------------------------------------
def load_leakdb(zip_path):
    rows = []

    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith(".csv") and "/Demands/" in name:
                try:
                    scenario = name.split("/")[2]
                    node = name.split("/")[-1].replace("Node_", "").replace(".csv", "")
                    df = pd.read_csv(z.open(name))

                    df["Sensor_ID"] = f"{scenario}_Node_{node}"
                    df["Demand"] = df.get("Value", 0.0)
                    df["Pressure"] = 0.0
                    df["Leak Status"] = 0  # placeholder label

                    rows.append(df[["Timestamp", "Sensor_ID", "Demand", "Pressure", "Leak Status"]])
                except Exception:
                    continue

    df = pd.concat(rows, ignore_index=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])

    print(f"Loaded {len(df)} rows from {df['Sensor_ID'].nunique()} sensors")
    return df


# -------------------------------------------------
# 2. Build Global Pipe-Network Graph
# -------------------------------------------------
def build_graph(df):
    sensors = sorted(df["Sensor_ID"].unique())
    sensor_to_idx = {s: i for i, s in enumerate(sensors)}

    edges = []
    for i in range(len(sensors) - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])

    edge_index = torch.tensor(edges, dtype=torch.long).T
    return sensor_to_idx, edge_index


# -------------------------------------------------
# 3. Create Temporal Graphs (one per timestep)
# -------------------------------------------------
def create_graphs(df, sensor_to_idx, edge_index):
    graphs = []

    for t, group in df.groupby("Timestamp"):
        x = torch.zeros(len(sensor_to_idx), 2)
        y = torch.zeros(len(sensor_to_idx))

        for _, row in group.iterrows():
            idx = sensor_to_idx[row["Sensor_ID"]]
            x[idx, 0] = row["Pressure"]
            x[idx, 1] = row["Demand"]
            y[idx] = row["Leak Status"]

        graphs.append(Data(x=x, edge_index=edge_index, y=y))

    print(f"Created {len(graphs)} temporal graphs")
    return graphs


# -------------------------------------------------
# 4. Graph Neural Network (TRUE GNN)
# ------------------------
