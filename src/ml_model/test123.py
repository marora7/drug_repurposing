import sqlite3
import pandas as pd

# Path to your SQLite database
db_path = r"F:\datasets\pubtator.db"

# Connect to the database
conn = sqlite3.connect(db_path)

# Execute a query to get unique edge_type values
query = "SELECT DISTINCT edge_type FROM edges_grouped"
unique_edge_types = pd.read_sql(query, conn)

print("Unique edge types in 'edges_grouped':")
print(unique_edge_types)

# Close the database connection
conn.close()
