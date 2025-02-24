import sqlite3
import pandas as pd

#Path to your SQLite database
db_path = "/mnt/f/datasets/pubtator.db"

# Connect to the database
conn = sqlite3.connect(db_path)

#nodes_df = pd.read_csv("data/exports/nodes.csv.gz", compression="gzip")
#print(nodes_df.columns.tolist())

query = "SELECT DISTINCT edge_type FROM edges_grouped;"
edge_types_df = pd.read_sql_query(query, conn)
print(edge_types_df)

# Close the database connection
conn.close()

#import torch
#print("CUDA available:", torch.cuda.is_available())
#print("CUDA version (PyTorch):", torch.version.cuda)



