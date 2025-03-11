import sqlite3

db_path = "data/processed/pubtator.db"

# Establish a database connection
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Define the update query
update_query = """
UPDATE diseases_grouping
SET 
    disease_type = 1, 
    disease_type_name = "Primary Treatable Diseases", 
    therapeutic_area = 15, 
    therapeutic_area_name = "Rare Diseases", 
    confidence = ?, 
    reasoning = ?
WHERE 
    disease_id = ? 
    AND disease_name = ?;
"""

# Define the parameters for safety
params = (0.8,
          "RHPD (Rhizomelic Pseudopolyarthritis with Hypoplastic Distal Phalanges, OMIM:208540) is a rare genetic disorder characterized by short limbs and abnormal bone development. It's classified as a primary treatable disease because while it has structural components, the underlying metabolic/genetic defect can be targeted with pharmacological interventions to manage symptoms and slow progression. Treatment includes medications to manage pain, inflammation, and bone metabolism. Given its genetic basis and rarity, it falls under the Rare Diseases therapeutic area.",
          "Disease:OMIM:208540", 
          "RHPD")


# Execute the update query
cursor.execute(update_query, params)

# Commit changes and close the connection
conn.commit()
conn.close()
