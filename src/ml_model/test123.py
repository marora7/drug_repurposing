import sqlite3
import pandas as pd

# Path to your SQLite database
db_path = r"F:\datasets\pubtator.db"

# Connect to the database
conn = sqlite3.connect(db_path)

nodes_df = pd.read_csv("data/exports/nodes.csv.gz", compression="gzip")
print(nodes_df.columns.tolist())

# Close the database connection
conn.close()




