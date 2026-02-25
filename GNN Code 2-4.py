import pandas as pd

# Function to read data from various formats

def read_data(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV, JSON, Parquet, or Excel file.")

# Example usage
if __name__ == '__main__':
    data = read_data('path/to/your/file')
    print(data)
