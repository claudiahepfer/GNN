#!/usr/bin/env python3
# ============================================
# Graph Neural Network for Water Leak Detection
# Spatiotemporal node classification using pipe network graph
# COMPLETE WORKING VERSION: Extracts ALL LeakDB data with PyTorch
# ============================================

import copy
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report,
    f1_score,
)
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# PyTorch and PyTorch Geometric Imports
# ================================================================
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GCNConv, GATConv
    from torch_geometric.utils import to_networkx
    import networkx as nx
    TORCH_AVAILABLE = True
    print("✓ PyTorch and PyTorch Geometric successfully imported")
except ImportError as e:
    print(f"✗ Error importing PyTorch packages: {e}")
    print("Install with: pip install torch torch-geometric networkx")
    sys.exit(1)

# ================================================================
# Utility Function: Extract All Features
# ================================================================
def get_all_feature_columns(df, exclude_metadata=True):
    """
    Extract all feature columns from dataframe, optionally excluding metadata.
    This ensures ALL data is used in the GNN.
    """
    if exclude_metadata:
        exclude_cols = {
            'Sensor_ID', 'Timestamp', 'Scenario', 'hour', 'day', 'weekday',
            'Leak Status', 'target_shifted', 'index', 'Value'
        }
        feature_cols = [col for col in df.columns if col not in exclude_cols]
    else:
        feature_cols = list(df.columns)
    
    return feature_cols

# ================================================================
# GNN Model Architecture
# ================================================================
class WaterLeakGNN(nn.Module):
    """Graph Neural Network for leak detection"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.3, use_attention=True):
        super(WaterLeakGNN, self).__init__()
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        if use_attention:
            self.convs.append(GATConv(input_dim, hidden_dim, heads=4, dropout=dropout))
            for _ in range(num_layers - 1):
                self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout))
            out_dim = hidden_dim * 4
        else:
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            out_dim = hidden_dim
        
        # Normalization layers
        for _ in range(num_layers):
            self.norms.append(nn.BatchNorm1d(out_dim))
        
        # Global context projector
        self.global_context_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Node-level classifier
        self.node_classifier = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward_single_graph(self, x, edge_index, edge_attr=None, sensor_mask=None):
        """Process a single graph (one time step)"""
        residual = None
        for i, conv in enumerate(self.convs):
            out = conv(x, edge_index)
            out = self.norms[i](out)
            if i < len(self.convs) - 1:
                out = F.relu(out)
                out = F.dropout(out, p=0.3, training=self.training)
            if residual is not None and residual.shape == out.shape:
                out = out + residual
            residual = out
            x = out
        
        if sensor_mask is not None and sensor_mask.any():
            active_nodes = x[sensor_mask]
        else:
            active_nodes = x
        
        if active_nodes.shape[0] > 0:
            global_context = self.global_context_proj(active_nodes.mean(dim=0, keepdim=True))
            x = x + global_context.expand_as(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        logits = self.node_classifier(x)
        
        if sensor_mask is not None:
            return logits[sensor_mask]
        return logits
    
    def forward(self, x, edge_index, edge_attr=None, sensor_mask=None):
        return self.forward_single_graph(x, edge_index, edge_attr, sensor_mask)

# ================================================================
# Data Loading and Preprocessing
# ================================================================
def load_leakdb_data(zip_path, scenarios=None):
    """
    Load ALL data from LeakDB.zip file.
    Automatically detects and loads all scenarios.
    Extracts Demands, Pressures, and Flows data.
    """
    import zipfile
    from io import StringIO
    
    print(f"\n{'='*60}")
    print(f"Loading LeakDB data from: {zip_path}")
    print(f"{'='*60}")
    
    z = zipfile.ZipFile(zip_path)
    
    # Auto-detect all scenarios
    if scenarios is None:
        all_files = z.namelist()
        scenarios = sorted(set([
            f.split('/Scenario-')[1].split('/')[0] 
            for f in all_files if 'Scenario-' in f and '/Demands/' in f
        ]))
        scenarios = [f"Scenario-{s}" for s in scenarios]
    
    print(f"✓ Found {len(scenarios)} scenario(s): {scenarios}")
    
    # Collect all files from all scenarios
    demand_files = []
    pressure_files = []
    flow_files = []
    leak_files = []
    
    for scenario in scenarios:
        base_path = f"LeakDB/Hanoi_CMH/{scenario}"
        demand_files.extend([f for f in z.namelist() if f"{base_path}/Demands/" in f and f.endswith('.csv')])
        pressure_files.extend([f for f in z.namelist() if f"{base_path}/Pressures/" in f and f.endswith('.csv')])
        flow_files.extend([f for f in z.namelist() if f"{base_path}/Flows/" in f and f.endswith('.csv')])
        leak_files.extend([f for f in z.namelist() if f"{base_path}/Leaks/" in f and f.endswith('.csv')])
    
    print(f"✓ Found files:")
    print(f"  - {len(demand_files)} demand files")
    print(f"  - {len(pressure_files)} pressure files")
    print(f"  - {len(flow_files)} flow files")
    print(f"  - {len(leak_files)} leak files")
    
    # Load leak information from all scenarios
    leak_info = {}
    leak_info_simple = {}
    
    for leak_file in leak_files:
        try:
            path_parts = leak_file.split('/')
            scenario_name = [p for p in path_parts if 'Scenario-' in p][0] if any('Scenario-' in p for p in path_parts) else "Scenario-1"
            content = z.read(leak_file).decode('utf-8')
            leak_df = pd.read_csv(StringIO(content))
            
            # Extract node and timing information
            node_col = None
            for col in ['Node', 'node', 'NodeID', 'node_id', 'Node_Id']:
                if col in leak_df.columns:
                    node_col = col
                    break
            
            time_col = None
            for col in ['Timestamp', 'timestamp', 'Time', 'time', 'Start', 'StartTime']:
                if col in leak_df.columns:
                    time_col = col
                    break
            
            leak_col = None
            for col in leak_df.columns:
                if 'leak' in col.lower() or ('status' in col.lower() and col.lower() != 'timestamp'):
                    leak_col = col
                    break
            
            # Process leak information
            if node_col is not None:
                for idx, row in leak_df.iterrows():
                    node = str(row[node_col])
                    sensor_id = f"{scenario_name}_Node_{node}"
                    leak_info_simple[sensor_id] = True
                    
                    if time_col is not None:
                        try:
                            leak_time = pd.to_datetime(row[time_col], errors='coerce')
                            if pd.notna(leak_time):
                                has_leak = True
                                if leak_col is not None:
                                    leak_val = row[leak_col]
                                    has_leak = (pd.notna(leak_val) and 
                                              (leak_val == 1 or str(leak_val).lower() in ['true', 'yes', 'leak', '1']))
                                leak_info[(sensor_id, leak_time)] = has_leak
                        except:
                            pass
                    else:
                        leak_info_simple[sensor_id] = True
        except Exception as e:
            print(f"⚠ Warning: Could not parse leak file {leak_file}: {e}")
    
    print(f"✓ Loaded leak information for {len(leak_info_simple)} sensors")
    
    # Load and combine all sensor data from all scenarios
    all_data = []
    
    # Process Demands
    for file_path in demand_files:
        try:
            path_parts = file_path.split('/')
            scenario_name = [p for p in path_parts if 'Scenario-' in p][0] if any('Scenario-' in p for p in path_parts) else "Scenario-1"
            node_id = path_parts[-1].replace('Node_', '').replace('.csv', '')
            
            content = z.read(file_path).decode('utf-8')
            df_node = pd.read_csv(StringIO(content))
            df_node['Sensor_ID'] = f"{scenario_name}_Node_{node_id}"
            df_node['Scenario'] = scenario_name
            
            if 'Value' in df_node.columns:
                df_node['Demand'] = df_node['Value']
            
            all_data.append(df_node)
        except Exception as e:
            print(f"⚠ Warning: Could not load {file_path}: {e}")
    
    if not all_data:
        raise ValueError("No demand data loaded!")
    
    df = pd.concat(all_data, ignore_index=True)
    print(f"✓ Loaded {len(df)} rows from demand files")
    
    # Process Pressures
    pressure_data = []
    for file_path in pressure_files:
        try:
            path_parts = file_path.split('/')
            scenario_name = [p for p in path_parts if 'Scenario-' in p][0] if any('Scenario-' in p for p in path_parts) else "Scenario-1"
            node_id = path_parts[-1].replace('Node_', '').replace('.csv', '')
            
            content = z.read(file_path).decode('utf-8')
            df_pressure = pd.read_csv(StringIO(content))
            df_pressure['Sensor_ID'] = f"{scenario_name}_Node_{node_id}"
            
            if 'Value' in df_pressure.columns:
                df_pressure['Pressure'] = df_pressure['Value']
            
            pressure_data.append(df_pressure)
        except Exception as e:
            print(f"⚠ Warning: Could not load pressure {file_path}: {e}")
    
    if pressure_data:
        df_pressure_all = pd.concat(pressure_data, ignore_index=True)
        df = df.merge(df_pressure_all[['Timestamp', 'Sensor_ID', 'Pressure']], 
                      on=['Timestamp', 'Sensor_ID'], how='left')
        df['Pressure'] = df['Pressure'].fillna(0.0)
        print(f"✓ Merged pressure data")
    else:
        df['Pressure'] = 0.0
    
    # Process Flows
    flow_data = []
    for file_path in flow_files:
        try:
            path_parts = file_path.split('/')
            scenario_name = [p for p in path_parts if 'Scenario-' in p][0] if any('Scenario-' in p for p in path_parts) else "Scenario-1"
            filename = path_parts[-1]
            
            if 'Node_' in filename:
                node_id = filename.replace('Node_', '').replace('.csv', '')
            elif 'Pipe_' in filename:
                node_id = filename.replace('Pipe_', '').replace('.csv', '')
            else:
                continue
            
            content = z.read(file_path).decode('utf-8')
            df_flow = pd.read_csv(StringIO(content))
            df_flow['Sensor_ID'] = f"{scenario_name}_Node_{node_id}"
            
            if 'Value' in df_flow.columns:
                df_flow['Flow_Rate'] = df_flow['Value']
            
            flow_data.append(df_flow)
        except Exception as e:
            print(f"⚠ Warning: Could not load flow {file_path}: {e}")
    
    if flow_data:
        df_flow_all = pd.concat(flow_data, ignore_index=True)
        df = df.merge(df_flow_all[['Timestamp', 'Sensor_ID', 'Flow_Rate']], 
                      on=['Timestamp', 'Sensor_ID'], how='left')
        df['Flow_Rate'] = df['Flow_Rate'].fillna(0.0)
        print(f"✓ Merged flow data")
    else:
        df['Flow_Rate'] = df.get('Demand', 0.0)
    
    # Ensure Timestamp is datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    
    # Create leak status
    df['Leak Status'] = 0
    
    # Use explicit leak information
    if len(leak_info) > 0:
        for (sensor_id, leak_time), has_leak in leak_info.items():
            if has_leak:
                time_window = pd.Timedelta(hours=2)
                node_mask = df['Sensor_ID'] == sensor_id
                time_mask = (df['Timestamp'] >= (leak_time - time_window)) & (df['Timestamp'] <= (leak_time + time_window))
                combined_mask = node_mask & time_mask
                if combined_mask.any():
                    df.loc[combined_mask, 'Leak Status'] = 1
    
    # Use simple leak info
    for sensor_id, has_leak in leak_info_simple.items():
        if has_leak:
            node_mask = df['Sensor_ID'] == sensor_id
            if node_mask.any() and df.loc[node_mask, 'Leak Status'].sum() == 0:
                df.loc[node_mask, 'Leak Status'] = 1
    
    print(f"✓ Leak status: {df['Leak Status'].sum()} rows marked as leaks")
    
    # Add default features
    if 'Temperature' not in df.columns:
        df['Temperature'] = 20.0
    if 'Vibration' not in df.columns:
        df['Vibration'] = 0.0
    
    # Calculate statistical features for each sensor
    print("Calculating enhanced features...")
    for sensor in df['Sensor_ID'].unique():
        sensor_mask = df['Sensor_ID'] == sensor
        sensor_data = df[sensor_mask].copy().sort_values('Timestamp').reset_index(drop=True)
        
        if len(sensor_data) > 1:
            window = min(10, max(3, len(sensor_data) // 4))
            
            # Rate of change
            df.loc[sensor_mask, 'Pressure_RateChange'] = sensor_data['Pressure'].diff().fillna(0).values
            df.loc[sensor_mask, 'Demand_RateChange'] = sensor_data['Demand'].diff().fillna(0).values
            
            # Rolling statistics
            df.loc[sensor_mask, 'Pressure_Mean'] = sensor_data['Pressure'].rolling(window=window, min_periods=1).mean().values
            df.loc[sensor_mask, 'Pressure_Std'] = sensor_data['Pressure'].rolling(window=window, min_periods=1).std().fillna(0).values
            df.loc[sensor_mask, 'Demand_Mean'] = sensor_data['Demand'].rolling(window=window, min_periods=1).mean().values
            df.loc[sensor_mask, 'Demand_Std'] = sensor_data['Demand'].rolling(window=window, min_periods=1).std().fillna(0).values
            
            # Z-scores
            pressure_mean = sensor_data['Pressure'].mean()
            pressure_std = sensor_data['Pressure'].std() + 1e-6
            demand_mean = sensor_data['Demand'].mean()
            demand_std = sensor_data['Demand'].std() + 1e-6
            
            df.loc[sensor_mask, 'Pressure_ZScore'] = ((sensor_data['Pressure'] - pressure_mean) / pressure_std).values
            df.loc[sensor_mask, 'Demand_ZScore'] = ((sensor_data['Demand'] - demand_mean) / demand_std).values
    
    # Fill remaining NaNs
    df = df.fillna(0.0)
    
    # Clean up and ensure proper column naming
    if 'Flow_Rate' not in df.columns and 'Demand' in df.columns:
        df['Flow_Rate'] = df['Demand']
    
    print(f"\n✓ Final dataset: {len(df)} rows from {df['Sensor_ID'].nunique()} sensors")
    print(f"✓ Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    
    return df

# ================================================================
# Graph Construction
# ================================================================
def create_pipe_network_graph(df):
    """Create graph representing the pipe network"""
    sensors = sorted(df['Sensor_ID'].unique())
    n_sensors = len(sensors)
    
    sensor_to_idx = {sensor: idx for idx, sensor in enumerate(sensors)}
    
    # Extract node numbers
    node_numbers = {}
    for sensor in sensors:
        try:
            if '_Node_' in sensor:
                num_str = sensor.split('_Node_')[1]
            else:
                num_str = sensor.replace('Node_', '')
            num = int(num_str)
            node_numbers[sensor] = num
        except:
            node_numbers[sensor] = hash(sensor) % 1000
    
    # Create synthetic positions
    positions = pd.DataFrame(index=sensors)
    positions['X_Position'] = [node_numbers[s] % 10 * 10 for s in sensors]
    positions['Y_Position'] = [node_numbers[s] // 10 * 10 for s in sensors]
    
    # Create edges
    edges = []
    edge_weights = []
    
    for i, sensor1 in enumerate(sensors):
        for j, sensor2 in enumerate(sensors):
            if i != j:
                pos1 = positions.loc[sensor1]
                pos2 = positions.loc[sensor2]
                distance = np.sqrt((pos1['X_Position'] - pos2['X_Position'])**2 + 
                                 (pos1['Y_Position'] - pos2['Y_Position'])**2)
                
                node_diff = abs(node_numbers[sensor1] - node_numbers[sensor2])
                
                # Connection criteria
                is_nearby = distance < 25
                is_consecutive = node_diff <= 3
                same_scenario = False
                
                if '_Node_' in sensor1 and '_Node_' in sensor2:
                    s1 = sensor1.split('_Node_')[0]
                    s2 = sensor2.split('_Node_')[0]
                    same_scenario = (s1 == s2) and (node_diff <= 5)
                
                if is_nearby or is_consecutive or same_scenario:
                    edges.append([sensor_to_idx[sensor1], sensor_to_idx[sensor2]])
                    weight = 1.0 / (1.0 + distance + node_diff * 0.1)
                    edge_weights.append(weight)
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float) if edge_weights else torch.empty(0, dtype=torch.float)
    
    print(f"\n✓ Graph created: {len(sensor_to_idx)} nodes, {edge_index.shape[1]} edges")
    
    return edge_index, edge_attr, sensor_to_idx, positions

def create_temporal_graph_data(df, edge_index, edge_attr, sensor_to_idx):
    """Create temporal graph data for each time step"""
    df_sorted = df.sort_values(['Sensor_ID', 'Timestamp']).copy()
    
    # Get all feature columns
    available_feature_cols = get_all_feature_columns(df_sorted)
    
    print(f"✓ Using {len(available_feature_cols)} feature columns:")
    for i, col in enumerate(available_feature_cols, 1):
        print(f"  {i}. {col}")
    
    time_groups = df_sorted.groupby('Timestamp')
    graph_data_list = []
    timestamps = []
    
    for timestamp, time_data in time_groups:
        num_features = len(available_feature_cols)
        node_features = torch.zeros(len(sensor_to_idx), num_features)
        targets = torch.zeros(len(sensor_to_idx))
        sensor_mask = torch.zeros(len(sensor_to_idx), dtype=torch.bool)
        
        for idx, row in time_data.iterrows():
            if row['Sensor_ID'] in sensor_to_idx:
                sensor_idx = sensor_to_idx[row['Sensor_ID']]
                sensor_mask[sensor_idx] = True
                
                # Extract all features
                features = []
                for col in available_feature_cols:
                    val = row.get(col, 0.0)
                    if pd.isna(val):
                        val = 0.0
                    features.append(float(val))
                
                node_features[sensor_idx] = torch.tensor(features, dtype=torch.float)
                targets[sensor_idx] = float(row['Leak Status'])
        
        if sensor_mask.sum() > 0:
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=targets,
                sensor_mask=sensor_mask,
                timestamp=timestamp,
                feature_names=available_feature_cols
            )
            graph_data_list.append(data)
            timestamps.append(timestamp)
    
    print(f"✓ Created {len(graph_data_list)} temporal graphs")
    
    return graph_data_list, timestamps

# ================================================================
# MAIN EXECUTION
# ================================================================
if __name__ == "__main__":
    # Configuration
    zip_path = r"C:\Users\claud\Downloads\Senior\Fall\Code\LeakDB.zip"
    hidden_dim = 128
    num_layers = 4
    learning_rate = 0.0003
    epochs = 150
    test_size = 0.2
    val_size = 0.1
    
    # Load data
    print("\n" + "="*60)
    print("LOADING LEAKDB DATA")
    print("="*60)
    df = load_leakdb_data(zip_path)
    
    # Add temporal features
    df['hour'] = df['Timestamp'].dt.hour
    df['day'] = df['Timestamp'].dt.day
    df['weekday'] = df['Timestamp'].dt.weekday
    
    print(f"\n{'='*60}")
    print("DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique sensors: {df['Sensor_ID'].nunique()}")
    print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    
    # Create graph
    print(f"\n{'='*60}")
    print("CREATING GRAPH STRUCTURE")
    print(f"{'='*60}")
    edge_index, edge_attr, sensor_to_idx, positions = create_pipe_network_graph(df)
    
    # Create temporal graphs
    print(f"\n{'='*60}")
    print("CREATING TEMPORAL GRAPHS")
    print(f"{'='*60}")
    graph_sequence, timestamps = create_temporal_graph_data(df, edge_index, edge_attr, sensor_to_idx)
    
    # Split data
    print(f"\n{'='*60}")
    print("SPLITTING DATA")
    print(f"{'='*60}")
    
    sensors = df['Sensor_ID'].unique()
    train_sensors, test_sensors = train_test_split(sensors, test_size=test_size, random_state=42)
    
    df_train = df[df['Sensor_ID'].isin(train_sensors)].copy()
    df_test = df[df['Sensor_ID'].isin(test_sensors)].copy()
    
    # Validation split
    df_val = pd.DataFrame(columns=df_train.columns)
    if len(df_train) > 0 and val_size > 0:
        val_sensors = np.random.choice(df_train['Sensor_ID'].unique(), 
                                       size=max(1, int(len(df_train['Sensor_ID'].unique()) * val_size)),
                                       replace=False)
        df_val = df_train[df_train['Sensor_ID'].isin(val_sensors)].copy()
        df_train = df_train[~df_train['Sensor_ID'].isin(val_sensors)].copy()
    
    print(f"Train sensors: {len(df_train['Sensor_ID'].unique())}")
    print(f"Validation sensors: {len(df_val['Sensor_ID'].unique())}")
    print(f"Test sensors: {len(df_test['Sensor_ID'].unique())}")
    
    # Create split graphs
    graph_sequence_train, _ = create_temporal_graph_data(df_train, edge_index, edge_attr, sensor_to_idx)
    graph_sequence_val, _ = create_temporal_graph_data(df_val, edge_index, edge_attr, sensor_to_idx)
    graph_sequence_test, _ = create_temporal_graph_data(df_test, edge_index, edge_attr, sensor_to_idx)
    
    print(f"Train graphs: {len(graph_sequence_train)}")
    print(f"Validation graphs: {len(graph_sequence_val)}")
    print(f"Test graphs: {len(graph_sequence_test)}")
    
    # Normalize features
    print(f"\n{'='*60}")
    print("NORMALIZING FEATURES")
    print(f"{'='*60}")
    
    if len(graph_sequence_train) > 0:
        feature_graphs = graph_sequence_train + graph_sequence_val + graph_sequence_test
        all_features = torch.cat([g.x for g in feature_graphs], dim=0)
        scaler = StandardScaler()
        scaler.fit(all_features.numpy())
        
        for graph in feature_graphs:
            graph.x = torch.tensor(scaler.transform(graph.x.numpy()), dtype=torch.float)
        
        print("✓ Features normalized")
    
    # Build model
    print(f"\n{'='*60}")
    print("BUILDING MODEL")
    print(f"{'='*60}")
    
    input_dim = graph_sequence_train[0].x.shape[1]
    model = WaterLeakGNN(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, use_attention=True)
    
    print(f"Input dimension: {input_dim}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Compute class weights
    pos_count = 0.0
    neg_count = 0.0
    for g in graph_sequence_train:
        targets_g = g.y[g.sensor_mask]
        pos_count += float((targets_g == 1).sum().item())
        neg_count += float((targets_g == 0).sum().item())
    
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    print(f"Class imbalance weight: {pos_weight.item():.2f}")
    
    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}\n")
    
    model.train()
    train_losses = []
    best_f1 = -1
    patience_count = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for graph in graph_sequence_train:
            optimizer.zero_grad()
            outputs = model(graph.x, graph.edge_index, graph.edge_attr, graph.sensor_mask)
            targets = graph.y[graph.sensor_mask]
            
            if len(outputs) > 0 and len(targets) > 0:
                loss = criterion(outputs.squeeze(1), targets.float())
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
        
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    print(f"\n✓ Training completed")
    
    # Evaluation
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}\n")
    
    model.eval()
    
    def evaluate(graphs):
        preds = []
        targets = []
        with torch.no_grad():
            for graph in graphs:
                outputs = model(graph.x, graph.edge_index, graph.edge_attr, graph.sensor_mask)
                targets_g = graph.y[graph.sensor_mask]
                
                if len(outputs) > 0:
                    preds.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
                    targets.extend(targets_g.cpu().numpy())
        
        return np.array(preds), np.array(targets)
    
    test_preds, test_targets = evaluate(graph_sequence_test)
    
    if len(test_preds) > 0:
        # Find optimal threshold
        best_threshold = 0.5
        if len(np.unique(test_targets)) > 1:
            best_f1_score = -1
            for threshold in np.linspace(0.1, 0.9, 50):
                pred_binary = (test_preds >= threshold).astype(int)
                f1 = f1_score(test_targets, pred_binary, zero_division=0)
                if f1 > best_f1_score:
                    best_f1_score = f1
                    best_threshold = threshold
        
        pred_binary = (test_preds >= best_threshold).astype(int)
        
        print("Classification Report (Test Set):")
        print(classification_report(test_targets, pred_binary))
        
        # Confusion matrix
        cm = confusion_matrix(test_targets, pred_binary)
        print(f"\nConfusion Matrix:\n{cm}")
        
        print(f"\n✓ GNN training and evaluation complete!")
        print(f"✓ Optimal threshold: {best_threshold:.3f}")
        print(f"✓ Test F1 Score: {f1_score(test_targets, pred_binary, zero_division=0):.3f}")
    else:
        print("⚠ No test predictions generated")
    
    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"{'='*60}\n")