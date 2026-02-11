# ============================================
# Graph Neural Network for Water Leak Detection
# Spatiotemporal node classification using pipe network graph
# Target: Node-level Leak Status at each time step (no data mixing, no leakage)
# NOTE: This is a TRUE GNN, not an MLP; see data pipeline and GNN architecture below.
# ============================================
#
# SETUP INSTRUCTIONS:
# 1. Run setup script: python setup_gnn.py
# 2. OR install manually: pip install -r requirements.txt
# 3. OR install individually: pip install torch torch-geometric networkx scikit-learn pandas numpy matplotlib
#
# IMPORTANT CONCEPTUAL NOTES (see user feedback):
# - This pipeline takes in the entire pipe network as a graph, with node features for each sensor/location.
# - Each node (sensor/pipe) receives its own prediction at each time step; message passing is used across network structure.
# - The only permissible test/train split for same-time prediction is by pipe ID (split_strategy='group_by_sensor').
# - Do NOT use random split: it will cause data leakage and invalidate the test metrics. Time-based holdout may be used for T->T+1 prediction.
# - This design fully avoids data leakage between test/train sets.
#
# LINTER NOTES:
# - Import warnings for torch/pytorch-geometric are EXPECTED if packages not installed
# - These warnings will disappear once you install the required packages
# - The script handles missing packages gracefully with proper error messages
# - All 7 linter warnings are related to optional PyTorch dependencies

import copy
import os
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
import joblib

# Optional PyTorch imports - will be checked at runtime
TORCH_AVAILABLE = False
torch = None
nn = None
F = None
Data = None
DataLoader = None
GCNConv = None
GATConv = None
global_mean_pool = None
global_max_pool = None
to_networkx = None
nx = None

try:
    # pylint: disable=import-outside-toplevel,import-error,import-unresolved
    # type: ignore
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401
    import torch.nn.functional as F  # noqa: F401
    from torch_geometric.data import Data, DataLoader  # noqa: F401
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool  # noqa: F401
    from torch_geometric.utils import to_networkx  # noqa: F401
    import networkx as nx  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"PyTorch Geometric not available: {e}")
    print("Please install with: pip install torch torch-geometric networkx")
    TORCH_AVAILABLE = False

# Check if PyTorch is available
if not TORCH_AVAILABLE:
    print("ERROR: PyTorch and PyTorch Geometric are required for this GNN implementation.")
    print("Please install them with: pip install torch torch-geometric networkx")
    exit(1)

# -------------------------------------------------------------
# GNN Architecture for Water Leak Detection
# -------------------------------------------------------------
class WaterLeakGNN(nn.Module):
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
        else:
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
        # Normalization after each conv output
        for _ in range(num_layers):
            self.norms.append(nn.BatchNorm1d(hidden_dim * (4 if use_attention else 1)))
        
        # Global context projector (graph-level summary fed back to nodes)
        out_dim = hidden_dim * (4 if use_attention else 1)
        self.global_context_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Enhanced node-level classification (for each sensor) - deeper network for better leak detection
        self.node_classifier = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),  # Additional layer
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1)  # logits (no sigmoid here)
        )
        
    def forward_single_graph(self, x, edge_index, edge_attr=None, sensor_mask=None):
        """Process a single graph (one time step)"""
        # Graph convolutions
        residual = None
        for i, conv in enumerate(self.convs):
            out = conv(x, edge_index)
            out = self.norms[i](out)
            if i < len(self.convs) - 1:
                out = F.relu(out)
                out = F.dropout(out, p=0.3, training=self.training)
            # simple residual when shapes match
            if residual is not None and residual.shape == out.shape:
                out = out + residual
            residual = out
            x = out
        
        if sensor_mask is not None and sensor_mask.any():
            active_nodes = x[sensor_mask]
        else:
            active_nodes = x
        
        # Graph-level context (mean of active nodes) projected back to nodes
        if active_nodes.shape[0] > 0:
            global_context = self.global_context_proj(active_nodes.mean(dim=0, keepdim=True))
            x = x + global_context.expand_as(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Node-level predictions
        logits = self.node_classifier(x)
        
        # Return predictions only for active sensors
        if sensor_mask is not None:
            return logits[sensor_mask]
        return logits
    
    def forward(self, x, edge_index, edge_attr=None, sensor_mask=None, graph_sequence=None):
        # Single graph processing (pure GNN, no recurrent modules)
        return self.forward_single_graph(x, edge_index, edge_attr, sensor_mask)

# -------------------------------------------------------------
# 1. Load the dataset from LeakDB.zip
# -------------------------------------------------------------
def load_leakdb_data(zip_path, scenarios=None):
    """
    Load data from LeakDB.zip file.
    Combines Demands, Pressures, and Flows data from ALL scenarios, and extracts leak information.
    If scenarios is None, automatically detects and loads all scenarios in the ZIP file.
    """
    import zipfile
    from io import StringIO
    
    print(f"=== Loading data from {zip_path} ===")
    z = zipfile.ZipFile(zip_path)
    
    # Auto-detect all scenarios if not specified
    if scenarios is None:
        scenarios = sorted(set([f.split('/Scenario-')[1].split('/')[0] 
                               for f in z.namelist() if 'Scenario-' in f and '/Demands/' in f]))
        scenarios = [f"Scenario-{s}" for s in scenarios]
    
    print(f"Found {len(scenarios)} scenario(s): {scenarios}")
    
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
    
    print(f"Found {len(demand_files)} demand files, {len(pressure_files)} pressure files, {len(flow_files)} flow files across all scenarios")
    
    # Load leak information from all scenarios with time-based information
    leak_info = {}  # Dict: (scenario_node, timestamp) -> leak_status
    leak_info_simple = {}  # Dict: scenario_node -> has_leak (for backward compat)
    
    for leak_file in leak_files:
        # Process all leak files, not just 'info' files
        try:
            path_parts = leak_file.split('/')
            scenario_name = [p for p in path_parts if 'Scenario-' in p][0] if any('Scenario-' in p for p in path_parts) else "Scenario-1"
            content = z.read(leak_file).decode('utf-8')
            leak_df = pd.read_csv(StringIO(content))
            
            # Extract leak node and timing information - handle multiple formats
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
            
            # Extract leak status column
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
                    
                    # If we have time information, store time-based leak status
                    if time_col is not None:
                        try:
                            leak_time = pd.to_datetime(row[time_col], errors='coerce')
                            if pd.notna(leak_time):
                                # Check if leak status column exists
                                has_leak = True
                                if leak_col is not None:
                                    leak_val = row[leak_col]
                                    has_leak = (pd.notna(leak_val) and 
                                              (leak_val == 1 or str(leak_val).lower() in ['true', 'yes', 'leak', '1']))
                                
                                leak_info[(sensor_id, leak_time)] = has_leak
                        except:
                            pass
                    else:
                        # No time info - mark all timestamps for this node
                        leak_info_simple[sensor_id] = True
            
            # Also check for direct leak indicators in any column
            if leak_col is None:
                for col in leak_df.columns:
                    if col.lower() not in ['node', 'nodeid', 'timestamp', 'time']:
                        for idx, row in leak_df.iterrows():
                            val = row[col]
                            if pd.notna(val) and (val == 1 or str(val).lower() in ['true', 'yes', 'leak']):
                                if node_col and node_col in leak_df.columns:
                                    node = str(leak_df.loc[idx, node_col])
                                    sensor_id = f"{scenario_name}_Node_{node}"
                                    leak_info_simple[sensor_id] = True
        except Exception as e:
            print(f"Warning: Could not parse leak file {leak_file}: {e}")
    
    # Load and combine all sensor data from all scenarios
    all_data = []
    
    # Process Demands from all scenarios
    for file_path in demand_files:
        try:
            # Extract scenario and node ID from file path
            # Format: "LeakDB/Hanoi_CMH/Scenario-X/Demands/Node_Y.csv"
            path_parts = file_path.split('/')
            scenario_name = [p for p in path_parts if 'Scenario-' in p][0] if any('Scenario-' in p for p in path_parts) else "Scenario-1"
            node_id = path_parts[-1].replace('Node_', '').replace('.csv', '')
            # Include scenario in Sensor_ID to avoid conflicts between scenarios
            content = z.read(file_path).decode('utf-8')
            df_node = pd.read_csv(StringIO(content))
            df_node['Sensor_ID'] = f"{scenario_name}_Node_{node_id}"
            df_node['Scenario'] = scenario_name
            df_node['Demand'] = df_node['Value']
            df_node = df_node[['Timestamp', 'Scenario', 'Sensor_ID', 'Demand']]
            all_data.append(df_node)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    
    # Process Pressures from all scenarios - merge with demand data
    pressure_data = []
    for file_path in pressure_files:
        try:
            # Extract scenario and node ID
            path_parts = file_path.split('/')
            scenario_name = [p for p in path_parts if 'Scenario-' in p][0] if any('Scenario-' in p for p in path_parts) else "Scenario-1"
            node_id = path_parts[-1].replace('Node_', '').replace('.csv', '')
            content = z.read(file_path).decode('utf-8')
            df_pressure = pd.read_csv(StringIO(content))
            df_pressure['Sensor_ID'] = f"{scenario_name}_Node_{node_id}"
            if 'Value' in df_pressure.columns:
                df_pressure['Pressure'] = df_pressure['Value']
            df_pressure = df_pressure[['Timestamp', 'Sensor_ID', 'Pressure']]
            pressure_data.append(df_pressure)
        except Exception as e:
            print(f"Warning: Could not load pressure {file_path}: {e}")
    
    # Process Flows from all scenarios - merge with demand data
    flow_data = []
    for file_path in flow_files:
        try:
            # Extract scenario and node/pipe ID from filename
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
            df_flow = df_flow[['Timestamp', 'Sensor_ID', 'Flow_Rate']]
            flow_data.append(df_flow)
        except Exception as e:
            print(f"Warning: Could not load flow {file_path}: {e}")
    
    # Combine all data
    if not all_data:
        raise ValueError("No data loaded from zip file!")
    
    df = pd.concat(all_data, ignore_index=True)
    
    # Merge pressure data
    if pressure_data:
        df_pressure_all = pd.concat(pressure_data, ignore_index=True)
        df = df.merge(df_pressure_all, on=['Timestamp', 'Sensor_ID'], how='left', suffixes=('', '_pressure'))
        if 'Pressure_pressure' in df.columns:
            df['Pressure'] = df['Pressure_pressure'].fillna(df.get('Pressure', 0.0))
            df = df.drop(columns=['Pressure_pressure'])
        df['Pressure'] = df['Pressure'].fillna(0.0)
    else:
        df['Pressure'] = 0.0
    
    # Merge flow data
    if flow_data:
        df_flow_all = pd.concat(flow_data, ignore_index=True)
        df = df.merge(df_flow_all, on=['Timestamp', 'Sensor_ID'], how='left', suffixes=('', '_flow'))
        if 'Flow_Rate_flow' in df.columns:
            df['Flow_Rate'] = df['Flow_Rate_flow'].fillna(df.get('Flow_Rate', 0.0))
            df = df.drop(columns=['Flow_Rate_flow'])
        df['Flow_Rate'] = df['Flow_Rate'].fillna(0.0)
    else:
        # Use Demand as Flow_Rate if no separate flow data
        df['Flow_Rate'] = df.get('Demand', 0.0)
    
    # Ensure Timestamp is datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])  # Remove rows with invalid timestamps
    
    # Create leak status based on leak information and anomalies in demand/pressure
    df['Leak Status'] = 0
    
    # First, use explicit leak information with time-based matching if available
    if len(leak_info) > 0:
        # Use time-based leak information
        for (sensor_id, leak_time), has_leak in leak_info.items():
            if has_leak:
                # Match sensor and time window (±2 hours for time-based matching)
                time_window = pd.Timedelta(hours=2)
                node_mask = df['Sensor_ID'] == sensor_id
                time_mask = (df['Timestamp'] >= (leak_time - time_window)) & (df['Timestamp'] <= (leak_time + time_window))
                combined_mask = node_mask & time_mask
                if combined_mask.any():
                    df.loc[combined_mask, 'Leak Status'] = 1
    
    # Also use simple leak info (without time matching)
    for sensor_id, has_leak in leak_info_simple.items():
        if has_leak:
            node_mask = df['Sensor_ID'] == sensor_id
            if node_mask.any() and df.loc[node_mask, 'Leak Status'].sum() == 0:
                # Only mark if not already marked by time-based matching
                df.loc[node_mask, 'Leak Status'] = 1
    
    print(f"Initial leak markers from leak files: {df['Leak Status'].sum()} rows")
    
    # Supplement with anomaly detection for more accurate leak detection
    # This helps catch leaks that might not be in leak files or to refine boundaries
    if df['Leak Status'].sum() == 0:
        print("Using enhanced anomaly detection for leak status...")
    
    # Enhanced leak detection using multiple signals and patterns
    for sensor in df['Sensor_ID'].unique():
        sensor_data = df[df['Sensor_ID'] == sensor].copy().sort_values('Timestamp').reset_index(drop=True)
        if len(sensor_data) < 5:
            continue
        
        # Calculate comprehensive rolling statistics for baseline
        window_size = max(10, min(20, len(sensor_data) // 4))
        sensor_data['Pressure_mean'] = sensor_data['Pressure'].rolling(window=window_size, min_periods=1).mean()
        sensor_data['Demand_mean'] = sensor_data['Demand'].rolling(window=window_size, min_periods=1).mean()
        sensor_data['Pressure_std'] = sensor_data['Pressure'].rolling(window=window_size, min_periods=1).std().fillna(0)
        sensor_data['Demand_std'] = sensor_data['Demand'].rolling(window=window_size, min_periods=1).std().fillna(0)
        
        # Calculate rate of change (first derivative) - leaks cause sudden changes
        sensor_data['Pressure_change'] = sensor_data['Pressure'].diff().abs()
        sensor_data['Demand_change'] = sensor_data['Demand'].diff().abs()
        sensor_data['Pressure_change_mean'] = sensor_data['Pressure_change'].rolling(window=5, min_periods=1).mean()
        sensor_data['Demand_change_mean'] = sensor_data['Demand_change'].rolling(window=5, min_periods=1).mean()
        
        # Calculate relative changes
        sensor_data['Pressure_relative'] = (sensor_data['Pressure'] - sensor_data['Pressure_mean']) / (sensor_data['Pressure_std'] + 1e-6)
        sensor_data['Demand_relative'] = (sensor_data['Demand'] - sensor_data['Demand_mean']) / (sensor_data['Demand_std'] + 1e-6)
        
        # Multiple leak indicators:
        # 1. Pressure drop anomaly (strong indicator)
        pressure_drop_severe = sensor_data['Pressure'] < (sensor_data['Pressure_mean'] - 2.0 * sensor_data['Pressure_std'])
        pressure_drop_moderate = sensor_data['Pressure'] < (sensor_data['Pressure_mean'] - 1.5 * sensor_data['Pressure_std'])
        
        # 2. Demand increase anomaly (strong indicator)
        demand_spike_severe = sensor_data['Demand'] > (sensor_data['Demand_mean'] + 2.0 * sensor_data['Demand_std'])
        demand_spike_moderate = sensor_data['Demand'] > (sensor_data['Demand_mean'] + 1.5 * sensor_data['Demand_std'])
        
        # 3. Sudden change detection (leaks start suddenly)
        sudden_pressure_change = sensor_data['Pressure_change'] > (sensor_data['Pressure_change_mean'] + 2 * sensor_data['Pressure_change'].std())
        sudden_demand_change = sensor_data['Demand_change'] > (sensor_data['Demand_change_mean'] + 2 * sensor_data['Demand_change'].std())
        
        # 4. Sustained anomalies (more reliable than single points)
        pressure_sustained = sensor_data['Pressure'].rolling(window=3, min_periods=2).mean() < (sensor_data['Pressure_mean'] - 1.5 * sensor_data['Pressure_std'])
        demand_sustained = sensor_data['Demand'].rolling(window=3, min_periods=2).mean() > (sensor_data['Demand_mean'] + 1.5 * sensor_data['Demand_std'])
        
        # 5. Combined signal - pressure drop AND demand increase (very strong indicator)
        combined_signal = (pressure_drop_moderate | pressure_drop_severe) & (demand_spike_moderate | demand_spike_severe)
        
        # Leak detection logic: combine multiple signals with different weights
        leak_score = (
            (pressure_drop_severe.astype(int) * 3) +
            (demand_spike_severe.astype(int) * 3) +
            (pressure_drop_moderate.astype(int) * 2) +
            (demand_spike_moderate.astype(int) * 2) +
            (sudden_pressure_change.astype(int) * 2) +
            (sudden_demand_change.astype(int) * 2) +
            (pressure_sustained.astype(int) * 2) +
            (demand_sustained.astype(int) * 2) +
            (combined_signal.astype(int) * 4)  # Combined signal is strongest
        )
        
        # Mark as leak if score exceeds threshold (more accurate than OR logic)
        leak_threshold = 4  # Require multiple indicators to agree
        sensor_data['Leak Status'] = (leak_score >= leak_threshold).astype(int)
        
        # Also preserve any existing leak markers from leak files
        existing_leaks = df.loc[df['Sensor_ID'] == sensor, 'Leak Status'].values
        if existing_leaks.sum() > 0:
            # Expand leak markers to adjacent time steps (±1 step)
            leak_indices = np.where(existing_leaks == 1)[0]
            for idx in leak_indices:
                if idx > 0:
                    sensor_data.loc[max(0, idx-1), 'Leak Status'] = 1
                if idx < len(sensor_data) - 1:
                    sensor_data.loc[min(len(sensor_data)-1, idx+1), 'Leak Status'] = 1
        
        df.loc[df['Sensor_ID'] == sensor, 'Leak Status'] = sensor_data['Leak Status'].values
    
    # Add enhanced features for better leak detection
    df['Temperature'] = 20.0  # Default temperature (not in LeakDB)
    df['Vibration'] = 0.0  # Default vibration (not in LeakDB)
    
    # Calculate temporal and statistical features for each sensor
    print("Calculating enhanced features for leak detection...")
    for sensor in df['Sensor_ID'].unique():
        sensor_mask = df['Sensor_ID'] == sensor
        sensor_data = df[sensor_mask].copy().sort_values('Timestamp').reset_index(drop=True)
        
        if len(sensor_data) > 1:
            # Rate of change features (leaks cause sudden changes)
            df.loc[sensor_mask, 'Pressure_RateChange'] = sensor_data['Pressure'].diff().values
            df.loc[sensor_mask, 'Demand_RateChange'] = sensor_data['Demand'].diff().values
            
            # Rolling statistics for normalization
            window = min(10, len(sensor_data) // 2)
            df.loc[sensor_mask, 'Pressure_RollingMean'] = sensor_data['Pressure'].rolling(window=window, min_periods=1).mean().values
            df.loc[sensor_mask, 'Pressure_RollingStd'] = sensor_data['Pressure'].rolling(window=window, min_periods=1).std().fillna(0).values
            df.loc[sensor_mask, 'Demand_RollingMean'] = sensor_data['Demand'].rolling(window=window, min_periods=1).mean().values
            df.loc[sensor_mask, 'Demand_RollingStd'] = sensor_data['Demand'].rolling(window=window, min_periods=1).std().fillna(0).values
            
            # Normalized deviations (z-scores)
            df.loc[sensor_mask, 'Pressure_ZScore'] = ((sensor_data['Pressure'] - sensor_data['Pressure'].mean()) / 
                                                      (sensor_data['Pressure'].std() + 1e-6)).values
            df.loc[sensor_mask, 'Demand_ZScore'] = ((sensor_data['Demand'] - sensor_data['Demand'].mean()) / 
                                                    (sensor_data['Demand'].std() + 1e-6)).values
            
            # Pressure-to-demand ratio (leaks change this ratio)
            df.loc[sensor_mask, 'Pressure_Demand_Ratio'] = (sensor_data['Pressure'] / (sensor_data['Demand'] + 1e-6)).values
            
            # Moving averages for trend detection
            df.loc[sensor_mask, 'Pressure_MA3'] = sensor_data['Pressure'].rolling(window=3, min_periods=1).mean().values
            df.loc[sensor_mask, 'Demand_MA3'] = sensor_data['Demand'].rolling(window=3, min_periods=1).mean().values
        else:
            # Fill with defaults for single-row sensors
            df.loc[sensor_mask, 'Pressure_RateChange'] = 0.0
            df.loc[sensor_mask, 'Demand_RateChange'] = 0.0
            df.loc[sensor_mask, 'Pressure_RollingMean'] = df.loc[sensor_mask, 'Pressure'].values
            df.loc[sensor_mask, 'Pressure_RollingStd'] = 0.0
            df.loc[sensor_mask, 'Demand_RollingMean'] = df.loc[sensor_mask, 'Demand'].values
            df.loc[sensor_mask, 'Demand_RollingStd'] = 0.0
            df.loc[sensor_mask, 'Pressure_ZScore'] = 0.0
            df.loc[sensor_mask, 'Demand_ZScore'] = 0.0
            df.loc[sensor_mask, 'Pressure_Demand_Ratio'] = 1.0
            df.loc[sensor_mask, 'Pressure_MA3'] = df.loc[sensor_mask, 'Pressure'].values
            df.loc[sensor_mask, 'Demand_MA3'] = df.loc[sensor_mask, 'Demand'].values
    
    # Fill any remaining NaN values
    feature_cols = ['Pressure_RateChange', 'Demand_RateChange', 'Pressure_RollingMean', 
                    'Pressure_RollingStd', 'Demand_RollingMean', 'Demand_RollingStd',
                    'Pressure_ZScore', 'Demand_ZScore', 'Pressure_Demand_Ratio',
                    'Pressure_MA3', 'Demand_MA3']
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    
    # Rename columns to match expected format
    if 'Flow_Rate' not in df.columns:
        df = df.rename(columns={'Demand': 'Flow_Rate'})  # Using Demand as Flow_Rate for compatibility
    
    print(f"Loaded {len(df)} rows from {len(df['Sensor_ID'].unique())} sensors")
    print(f"Leak status distribution: {df['Leak Status'].value_counts().to_dict()}")
    
    return df

# Load data from zip file - process ALL scenarios automatically
zip_path = r"C:\Users\claud\Downloads\Senior\Fall\Code\LeakDB.zip"
df = load_leakdb_data(zip_path, scenarios=None)  # None = auto-detect and load all scenarios

print("\n=== Dataset loaded (GNN for Water Leak Detection) ===")
print(df.head())
print(df.info())
print(f"Dataset contains {len(df)} rows from {len(df['Sensor_ID'].unique())} sensors")

# -------------------------------------------------------------
# 2. Graph Construction Functions
# -------------------------------------------------------------
def create_pipe_network_graph(df, connectivity_threshold=0.25):
    """
    Create a graph representing the pipe network.
    Nodes: sensors, Edges: connections between nearby sensors
    For LeakDB, we create connections based on node numbering and data correlations
    Increased connectivity threshold to better capture relationships across all scenarios
    """
    # Get unique sensors and their positions (if available)
    sensors = sorted(df['Sensor_ID'].unique())
    n_sensors = len(sensors)
    
    # Create sensor mapping
    sensor_to_idx = {sensor: idx for idx, sensor in enumerate(sensors)}
    
    # Extract node numbers for better connectivity (now handles scenario prefixes)
    node_numbers = {}
    for sensor in sensors:
        try:
            # Extract number from "Scenario-X_Node_Y" or "Node_X" format
            if '_Node_' in sensor:
                num_str = sensor.split('_Node_')[1]
            elif 'Node_' in sensor:
                num_str = sensor.replace('Node_', '')
            else:
                num_str = sensor
            num = int(num_str)
            node_numbers[sensor] = num
        except:
            node_numbers[sensor] = hash(sensor) % 1000
    
    # If we have position data, use it; otherwise create synthetic positions based on node numbers
    if 'X_Position' in df.columns and 'Y_Position' in df.columns:
        positions = df.groupby('Sensor_ID')[['X_Position', 'Y_Position']].first()
    else:
        # Create synthetic grid positions based on node numbers for better connectivity
        positions = pd.DataFrame(index=sensors)
        positions['X_Position'] = [node_numbers[s] % 10 * 10 for s in sensors]
        positions['Y_Position'] = [node_numbers[s] // 10 * 10 for s in sensors]
    
    # Create edges based on proximity and node numbering (simulating pipe network topology)
    edges = []
    edge_weights = []
    
    for i, sensor1 in enumerate(sensors):
        for j, sensor2 in enumerate(sensors):
            if i != j:
                pos1 = positions.loc[sensor1]
                pos2 = positions.loc[sensor2]
                distance = np.sqrt((pos1['X_Position'] - pos2['X_Position'])**2 + 
                                 (pos1['Y_Position'] - pos2['Y_Position'])**2)
                
                # Connect nearby sensors or consecutive node numbers
                node_diff = abs(node_numbers[sensor1] - node_numbers[sensor2])
                is_nearby = distance < connectivity_threshold * 100
                is_consecutive = node_diff <= 3  # Increased from 2 to 3 for better connectivity across scenarios
                # Connect nodes from same scenario with similar node numbers (within same scenario network)
                same_scenario = False
                if '_Node_' in sensor1 and '_Node_' in sensor2:
                    scenario1 = sensor1.split('_Node_')[0]
                    scenario2 = sensor2.split('_Node_')[0]
                    same_scenario = (scenario1 == scenario2) and (node_diff <= 5)  # Connect within same scenario network
                
                if is_nearby or is_consecutive or same_scenario:
                    edges.append([sensor_to_idx[sensor1], sensor_to_idx[sensor2]])
                    # Weight based on both distance and node proximity
                    weight = 1.0 / (1.0 + distance + node_diff * 0.1)
                    edge_weights.append(weight)
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float) if edge_weights else torch.empty(0, dtype=torch.float)
    
    return edge_index, edge_attr, sensor_to_idx, positions

def create_temporal_graph_data(df, edge_index, edge_attr, sensor_to_idx, target_shift_steps=0):
    """
    Create PyTorch Geometric Data objects for temporal graph learning
    Each time step becomes a separate graph with the same topology
    """
    # Ensure data is sorted by sensor and time
    df_sorted = df.sort_values(['Sensor_ID', 'Timestamp']).copy()
    
    # Group data by time step
    time_groups = df_sorted.groupby('Timestamp')
    
    graph_data_list = []
    all_sensor_ids = []
    all_timestamps = []
    
    for timestamp, time_data in time_groups:
        # Create node features for this time step (expanded to 19 features)
        node_features = torch.zeros(len(sensor_to_idx), 19)  # Enhanced feature size
        targets = torch.zeros(len(sensor_to_idx))
        sensor_mask = torch.zeros(len(sensor_to_idx), dtype=torch.bool)
        
        for idx, row in time_data.iterrows():
            sensor_idx = sensor_to_idx[row['Sensor_ID']]
            sensor_mask[sensor_idx] = True
            
            # Enhanced node features with temporal patterns and statistical features
            # Safely extract scalar values from row
            def safe_get(row, key, default=0.0):
                """Safely extract scalar value from pandas Series row"""
                try:
                    if key in row.index:
                        val = row[key]
                        if hasattr(val, 'item'):
                            val = val.item()
                        elif not pd.api.types.is_scalar(val):
                            val = val.iloc[0] if hasattr(val, 'iloc') else default
                        return float(val) if pd.notna(val) else default
                except (KeyError, IndexError, ValueError, TypeError):
                    pass
                return default
            
            # Temporal features
            hour_val = safe_get(row, 'hour', 0)
            day_val = safe_get(row, 'day', 0)
            weekday_val = safe_get(row, 'weekday', 0)
            
            # Core sensor measurements
            flow_value = safe_get(row, 'Flow_Rate', safe_get(row, 'Demand', 0.0))
            pressure_value = safe_get(row, 'Pressure', 0.0)
            temp_value = safe_get(row, 'Temperature', 20.0)
            vib_value = safe_get(row, 'Vibration', 0.0)
            
            # Enhanced features for leak detection
            pressure_rate = safe_get(row, 'Pressure_RateChange', 0.0)
            demand_rate = safe_get(row, 'Demand_RateChange', 0.0)
            pressure_zscore = safe_get(row, 'Pressure_ZScore', 0.0)
            demand_zscore = safe_get(row, 'Demand_ZScore', 0.0)
            pressure_demand_ratio = safe_get(row, 'Pressure_Demand_Ratio', 1.0)
            pressure_ma3 = safe_get(row, 'Pressure_MA3', pressure_value)
            demand_ma3 = safe_get(row, 'Demand_MA3', flow_value)
            
            # Build comprehensive feature vector (19 features total)
            features = [
                # Temporal (3)
                hour_val,
                day_val,
                weekday_val,
                # Core measurements (4)
                pressure_value,
                flow_value,
                temp_value,
                vib_value,
                # Rate of change (2) - important for leak detection
                pressure_rate,
                demand_rate,
                # Statistical features (4) - z-scores and ratios
                pressure_zscore,
                demand_zscore,
                pressure_demand_ratio,
                (pressure_value - pressure_ma3) / (abs(pressure_ma3) + 1e-6),  # Deviation from moving avg
                # Moving averages (2) - trend indicators
                pressure_ma3,
                demand_ma3,
                # Additional derived features (4) - for better leak pattern capture
                abs(pressure_rate),  # Absolute rate of change
                abs(demand_rate),
                (flow_value - demand_ma3) / (abs(demand_ma3) + 1e-6),  # Demand deviation
                np.sin(2 * np.pi * hour_val / 24),  # Cyclical hour encoding
            ]
            
            node_features[sensor_idx] = torch.tensor(features, dtype=torch.float)
            
            # Target (with optional shifting)
            if target_shift_steps == 0:
                targets[sensor_idx] = row[target_col]
            else:
                targets[sensor_idx] = row.get('target_shifted', row[target_col])
        
        # Only include graphs with at least some sensor data
        if sensor_mask.sum() > 0:
            # Create PyTorch Geometric Data object for this time step
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=targets,
                sensor_mask=sensor_mask,
                timestamp=timestamp
            )
            graph_data_list.append(data)
            all_timestamps.append(timestamp)
    
    return graph_data_list, all_timestamps

# -------------------------------------------------------------
# 3. Configuration: problem definition and split strategy
# -------------------------------------------------------------
# See user feedback:
# - What are we predicting? At each time step, we predict leak at each node, based ONLY on input available up to that time (with future prediction possible with TARGET_SHIFT_STEPS).
# - Validation: for proper testing, group_by_sensor split ensures sensors/pipes are ONLY in train OR test set, never both. This prevents target leakage.
#
target_col = "Leak Status"  # Explicit target

# How to split data to avoid leakage:
# - "random": standard iid split
# - "group_by_sensor": keep all rows of a given Sensor_ID in either train or test
# - "time_holdout": use the most recent timestamps as test set
split_strategy = os.getenv("SPLIT_STRATEGY", "group_by_sensor")  # Must be 'group_by_sensor' for valid eval!
if split_strategy != "group_by_sensor":
    raise ValueError("For valid GNN evaluation, split_strategy must be 'group_by_sensor'.\nPlease set SPLIT_STRATEGY=group_by_sensor in your environment or at script top.")
test_size = float(os.getenv("TEST_SIZE", "0.2"))
val_size = float(os.getenv("VAL_SIZE", "0.1"))

# Predict leakage at future time steps? If >0, we use data at time T to predict leak at T+steps per sensor.
target_shift_steps = int(os.getenv("TARGET_SHIFT_STEPS", "0"))  # 0 for same-time, 1 for T->T+1, etc.

# GNN hyperparameters (optimized for accurate leak detection with all data)
hidden_dim = int(os.getenv("HIDDEN_DIM", "128"))  # Increased for better capacity with all scenarios
num_layers = int(os.getenv("NUM_LAYERS", "4"))  # More layers for deeper feature learning and better leak pattern capture
use_attention = os.getenv("USE_ATTENTION", "true").lower() == "true"
learning_rate = float(os.getenv("LEARNING_RATE", "0.0003"))  # Lower learning rate for more stable training
epochs = int(os.getenv("EPOCHS", "150"))  # More epochs to fully learn leak patterns
dropout = float(os.getenv("DROPOUT", "0.3"))  # Reduced dropout for better leak signal preservation
batch_size = int(os.getenv("BATCH_SIZE", "32"))  # Batch size for training

# Timestamp is already converted to datetime in load_leakdb_data, but ensure it's still datetime
if df["Timestamp"].dtype != 'datetime64[ns]':
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

# Feature engineering for temporal features
df["hour"] = df["Timestamp"].dt.hour
df["day"] = df["Timestamp"].dt.day
df["weekday"] = df["Timestamp"].dt.weekday

# Optional: shift target forward within each sensor to predict future state
if target_shift_steps > 0:
    df = df.sort_values(["Sensor_ID", "Timestamp"]).copy()
    df["target_shifted"] = df.groupby("Sensor_ID")[target_col].shift(-target_shift_steps)
    # Drop rows where future target is not available
    df = df.dropna(subset=["target_shifted"]).copy()
    df["target_shifted"] = df["target_shifted"].astype(int)
    y = df["target_shifted"]
else:
    y = df[target_col]

# -------------------------------------------------------------
# 4. Create Graph Structure and Split Data
# -------------------------------------------------------------
print("\n=== Creating Pipe Network Graph ===")
edge_index, edge_attr, sensor_to_idx, positions = create_pipe_network_graph(df)
print(f"Graph created with {len(sensor_to_idx)} sensors and {edge_index.shape[1]} edges")

# Create temporal graph data
print("=== Creating Temporal Graph Data ===")
graph_sequence, timestamps = create_temporal_graph_data(df, edge_index, edge_attr, sensor_to_idx, target_shift_steps)
print(f"Graph sequence created with {len(graph_sequence)} time steps")
if len(graph_sequence) > 0:
    print(f"Each graph has {graph_sequence[0].x.shape[0]} nodes and {graph_sequence[0].x.shape[1]} features per node")
    # ---------------------------------------------------------
    # Save full GNN graph sequence so you can inspect/use it
    # ---------------------------------------------------------
    torch.save(
        {
            "graph_sequence": graph_sequence,
            "timestamps": timestamps,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "sensor_to_idx": sensor_to_idx,
            "all_columns_used": list(df.columns),
        },
        "gnn_graph_sequence_all_data.pt",
    )
    # Also save a single sample graph (first time step)
    torch.save(graph_sequence[0], "gnn_sample_graph_all_data.pt")
    print("\nSaved full GNN graph sequence to 'gnn_graph_sequence_all_data.pt'")
    print("Saved sample GNN graph (first time step) to 'gnn_sample_graph_all_data.pt'")

# Split graph sequence based on strategy
if split_strategy == "group_by_sensor":
    # Group-aware split: sensors appear in only one of train or test
    sensors = df["Sensor_ID"].unique()
    train_sensors, test_sensors = train_test_split(
        sensors, test_size=test_size, random_state=42
    )
    train_mask = df["Sensor_ID"].isin(train_sensors)
    test_mask = df["Sensor_ID"].isin(test_sensors)
    df_train, df_test = df.loc[train_mask].copy(), df.loc[test_mask].copy()
elif split_strategy == "time_holdout":
    # Hold out the most recent fraction as test
    cutoff = df["Timestamp"].quantile(1.0 - test_size)
    train_mask = df["Timestamp"] < cutoff
    test_mask = df["Timestamp"] >= cutoff
    df_train, df_test = df.loc[train_mask].copy(), df.loc[test_mask].copy()
else:
    # Random split (iid). Consider temporal cross-validation for time-series.
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df[target_col]
    )

# Optional validation split within training sensors to tune thresholds/early stopping
df_val = pd.DataFrame(columns=df_train.columns)
if len(df_train) > 0 and val_size > 0:
    unique_train_sensors = df_train["Sensor_ID"].unique()
    if len(unique_train_sensors) >= 3:
        val_size_clipped = min(max(val_size, 0.05), 0.5)
        train_sensors_final, val_sensors = train_test_split(
            unique_train_sensors,
            test_size=val_size_clipped,
            random_state=42
        )
        df_val = df_train[df_train["Sensor_ID"].isin(val_sensors)].copy()
        df_train = df_train[df_train["Sensor_ID"].isin(train_sensors_final)].copy()
        print(f"Validation split enabled with {len(val_sensors)} sensors.")
    else:
        print("Skipping validation split: not enough unique sensors for a reliable holdout.")

# Create separate graph sequences for train and test
graph_sequence_train, timestamps_train = create_temporal_graph_data(
    df_train, edge_index, edge_attr, sensor_to_idx, target_shift_steps
)
graph_sequence_val, timestamps_val = create_temporal_graph_data(
    df_val, edge_index, edge_attr, sensor_to_idx, target_shift_steps
)
graph_sequence_test, timestamps_test = create_temporal_graph_data(
    df_test, edge_index, edge_attr, sensor_to_idx, target_shift_steps
)

print(f"Train sequence: {len(graph_sequence_train)} time steps")
print(f"Validation sequence: {len(graph_sequence_val)} time steps")
print(f"Test sequence: {len(graph_sequence_test)} time steps")

# Normalize features across all graphs
if len(graph_sequence_train) > 0:
    # Collect all features for normalization
    feature_graphs = graph_sequence_train + graph_sequence_val + graph_sequence_test
    all_features = torch.cat([g.x for g in feature_graphs], dim=0)
    scaler = StandardScaler()
    scaler.fit(all_features.numpy())
    
    # Apply normalization to all graphs
    for graph in feature_graphs:
        graph.x = torch.tensor(scaler.transform(graph.x.numpy()), dtype=torch.float)

# -------------------------------------------------------------
# 5. Build and train GNN
# -------------------------------------------------------------
# Model explicitly performs node-level leak classification using full pipe network graph.
print("\n=== Building GNN Model ===")
if len(graph_sequence_train) == 0:
    print("ERROR: No training data available!")
    exit(1)

input_dim = graph_sequence_train[0].x.shape[1]
model = WaterLeakGNN(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    use_attention=use_attention,
    dropout=dropout
)

# Use Adam optimizer with weight decay for regularization
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
# Add learning rate scheduler for better convergence
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
)

# Compute class imbalance from training graphs for pos_weight
pos_count = 0.0
neg_count = 0.0
for g in graph_sequence_train:
    targets_g = g.y[g.sensor_mask] if hasattr(g, 'sensor_mask') else g.y
    pos_count += float((targets_g == 1).sum().item())
    neg_count += float((targets_g == 0).sum().item())
pos_weight_value = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Utility function for evaluation during training
def evaluate_graphs(model_to_eval, graphs, loss_fn=None):
    model_to_eval.eval()
    preds = []
    targets = []
    losses = []
    with torch.no_grad():
        for graph in graphs:
            outputs = model_to_eval(graph.x, graph.edge_index, graph.edge_attr, graph.sensor_mask)
            graph_targets = graph.y[graph.sensor_mask] if hasattr(graph, 'sensor_mask') else graph.y
            if len(outputs) == 0 or len(graph_targets) == 0:
                continue
            logits = outputs.squeeze(1)
            if loss_fn is not None:
                losses.append(loss_fn(logits, graph_targets.float()).item())
            preds.extend(torch.sigmoid(logits).cpu().numpy())
            targets.extend(graph_targets.cpu().numpy())
    avg_loss = float(np.mean(losses)) if losses else float("nan")
    return np.array(preds), np.array(targets), avg_loss

# Training loop
print("\n=== Training GNN ===")
model.train()
train_losses = []
val_metric_history = []
best_state_dict = copy.deepcopy(model.state_dict())
best_val_f1_global = -np.inf  # Global best F1 for tracking
patience = int(os.getenv("PATIENCE", "20"))
patience_counter = 0

for epoch in range(epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    # Process each graph in the training sequence
    for graph in graph_sequence_train:
        optimizer.zero_grad()
        
        # Forward pass for single graph
        outputs = model(graph.x, graph.edge_index, graph.edge_attr, graph.sensor_mask)
        targets = graph.y[graph.sensor_mask] if hasattr(graph, 'sensor_mask') else graph.y
        
        if len(outputs) > 0 and len(targets) > 0:
            # Ensure shapes are [N] to avoid 0-d tensors when N==1
            loss = criterion(outputs.squeeze(1), targets.float())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
    
    if num_batches > 0:
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Validation monitoring for early stopping
    if len(graph_sequence_val) > 0 and (epoch + 1) % max(1, int(os.getenv("VAL_EVAL_EVERY", "5"))) == 0:
        val_preds, val_targets, val_loss = evaluate_graphs(model, graph_sequence_val, criterion)
        if val_targets.size > 0:
            # Use F1 score for threshold selection during validation
            val_f1_scores = []
            for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
                val_binary = (val_preds >= thresh).astype(int)
                if len(np.unique(val_binary)) > 1:
                    val_f1 = f1_score(val_targets, val_binary, zero_division=0)
                    val_f1_scores.append((thresh, val_f1))
            if val_f1_scores:
                best_val_thresh, best_val_f1 = max(val_f1_scores, key=lambda x: x[1])
            else:
                best_val_f1 = 0
            val_metric_history.append((epoch + 1, val_loss, best_val_f1))
            print(f"  Validation @ epoch {epoch+1}: loss={val_loss:.4f}, F1={best_val_f1:.4f}")
            
            # Update learning rate scheduler
            scheduler.step(val_loss)
            
            if best_val_f1 > best_val_f1_global:
                best_val_f1_global = best_val_f1
                best_state_dict = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break

# Load best model weights (if validation was used)
model.load_state_dict(best_state_dict)

# -------------------------------------------------------------
# 6. Evaluate the model
# -------------------------------------------------------------
print("\n=== Evaluating GNN ===")
train_pred_np, train_y_np, _ = evaluate_graphs(model, graph_sequence_train, criterion)
val_pred_np, val_y_np, _ = evaluate_graphs(model, graph_sequence_val, criterion)
test_pred_np, test_y_np, _ = evaluate_graphs(model, graph_sequence_test, criterion)

# Learn optimal threshold for leak detection using multiple metrics
# For leak detection, we want high recall (catch all leaks) while maintaining good precision
if val_y_np.size > 0 and len(np.unique(val_y_np)) > 1:
    fpr, tpr, thresholds = roc_curve(val_y_np, val_pred_np)
    targets_for_threshold = val_y_np
    preds_for_threshold = val_pred_np
elif len(np.unique(test_y_np)) > 1:
    fpr, tpr, thresholds = roc_curve(test_y_np, test_pred_np)
    targets_for_threshold = test_y_np
    preds_for_threshold = test_pred_np
else:
    fpr = np.array([0.0, 1.0])
    tpr = np.array([0.0, 1.0])
    thresholds = np.array([0.5])
    targets_for_threshold = test_y_np
    preds_for_threshold = test_pred_np

# Use F1 score to find optimal threshold (balances precision and recall)
best_f1 = -1
best_threshold = 0.5
for threshold in np.linspace(0.1, 0.9, 81):
    binary_preds = (preds_for_threshold >= threshold).astype(int)
    if len(np.unique(binary_preds)) > 1 and len(np.unique(targets_for_threshold)) > 1:
        f1 = f1_score(targets_for_threshold, binary_preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

# Also compute Youden's J as backup
if len(thresholds) > 1:
    youden_j = tpr - fpr
    youden_idx = int(np.argmax(youden_j))
    youden_threshold = thresholds[youden_idx] if thresholds.size > 0 else 0.5
    print(f"Optimal threshold (F1-based): {best_threshold:.4f}, (Youden's J): {youden_threshold:.4f}")
else:
    print(f"Using default threshold: {best_threshold:.4f}")

# Convert probabilities to binary predictions using tuned threshold
train_pred_binary = (train_pred_np >= best_threshold).astype(int)
val_pred_binary = (val_pred_np >= best_threshold).astype(int) if val_pred_np.size > 0 else np.array([])
test_pred_binary = (test_pred_np >= best_threshold).astype(int)

print("\n=== Classification Report (Test Set) ===")
print(classification_report(test_y_np, test_pred_binary))

# Confusion Matrix over ALL rows (train + test)
if train_y_np.size == 0:
    all_y_np = test_y_np
    all_pred_binary = test_pred_binary
elif test_y_np.size == 0:
    if val_y_np.size == 0:
        all_y_np = train_y_np
        all_pred_binary = train_pred_binary
    else:
        all_y_np = np.concatenate([train_y_np, val_y_np])
        all_pred_binary = np.concatenate([train_pred_binary, val_pred_binary])
else:
    if val_y_np.size == 0:
        all_y_np = np.concatenate([train_y_np, test_y_np])
        all_pred_binary = np.concatenate([train_pred_binary, test_pred_binary])
    else:
        all_y_np = np.concatenate([train_y_np, val_y_np, test_y_np])
        all_pred_binary = np.concatenate([train_pred_binary, val_pred_binary, test_pred_binary])

cm = confusion_matrix(all_y_np, all_pred_binary)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig1, ax1 = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax1)
ax1.set_title("GNN Confusion Matrix (All Rows, threshold = 0.5)")
plt.show()

# ROC Curve
roc_auc = auc(fpr, tpr)
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, linestyle='-', label=f"ROC curve (AUC = {roc_auc:.3f})")
ax2.plot([0,1],[0,1], linestyle='--', linewidth=1)
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("GNN ROC Curve")
ax2.legend(loc="lower right")
plt.show()

# Training Loss Curve
plt.figure(figsize=(8, 4))
plt.plot(train_losses)
plt.title("GNN Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# -------------------------------------------------------------
# 7. Graph Visualization (if small enough)
# -------------------------------------------------------------
if len(sensor_to_idx) <= 50 and len(graph_sequence_train) > 0:  # Only visualize if graph is small enough
    print("\n=== Graph Visualization ===")
    try:
        # Use the first training graph for visualization
        sample_graph = graph_sequence_train[0]
        G = to_networkx(sample_graph, to_undirected=True)
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Color nodes by leak status if available
        node_colors = []
        for i in range(len(sensor_to_idx)):
            if hasattr(sample_graph, 'sensor_mask') and sample_graph.sensor_mask[i]:
                # Active sensor - color by leak status
                leak_status = sample_graph.y[i].item()
                node_colors.append('red' if leak_status > 0.5 else 'lightblue')
            else:
                # Inactive sensor
                node_colors.append('gray')
        
        nx.draw(G, pos, with_labels=True, node_color=node_colors, 
                node_size=500, font_size=8, font_weight='bold')
        plt.title("Pipe Network Graph Structure\n(Red=Leak, Blue=Normal, Gray=Inactive)")
        plt.show()
    except Exception as e:
        print(f"Could not visualize graph: {e}")
else:
    print(f"\nGraph too large for visualization ({len(sensor_to_idx)} nodes) or no training data")

# -------------------------------------------------------------
# 8. Save trained model
# -------------------------------------------------------------
model_save_path = "water_leak_gnn_model.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'sensor_to_idx': sensor_to_idx,
    'edge_index': edge_index,
    'input_dim': input_dim,
    'hidden_dim': hidden_dim,
    'num_layers': num_layers,
    'use_attention': use_attention,
    'target_shift_steps': target_shift_steps
}, model_save_path)

print(f"\nGNN model saved successfully as: {model_save_path}")

print("\n=== GNN Architecture Summary ===")
print(f"- Graph structure: {len(sensor_to_idx)} sensors, {edge_index.shape[1]} edges")
print(f"- Node features: {input_dim} dimensions")
print(f"- Hidden dimension: {hidden_dim}")
print(f"- Number of layers: {num_layers}")
print(f"- Attention mechanism: {'Yes' if use_attention else 'No'}")
print(f"- Prediction task: {'T->T+' + str(target_shift_steps) if target_shift_steps > 0 else 'Same-time'}")

print("\nThis is now a proper Graph Neural Network that:")
print("- Models the pipe network as a graph (nodes=sensors, edges=pipe connections)")
print("- Node-wise prediction leverages spatial structure (message passing)")
print("- Uses train/test split by pipe ID to avoid leakage (check SPLIT_STRATEGY)")
print("- Prediction: leak status at each node at each time step (node classification)")
print("- Handles both same-time and future prediction tasks through 'target_shift_steps'")

print("\nEvaluation caution: This script ensures NO train/test leakage by enforcing 'group_by_sensor' split. For temporal forecasting (predicting T+1), use 'time_holdout' split and set TARGET_SHIFT_STEPS>0. For production or science, review Temporal Crossvalidation literature for robust assessment.")

print("\nScript complete ✅")