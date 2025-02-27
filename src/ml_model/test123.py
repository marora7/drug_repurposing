import sqlite3

# Database location
db_path = r"/mnt/f/drug_repurposing/data/processed/pubtator.db"

def check_and_delete_rows():
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check number of rows before deletion
        cursor.execute("SELECT COUNT(*) FROM treat_relations")
        initial_count = cursor.fetchone()[0]
        print(f"Number of rows before deletion: {initial_count}")
        
        # Delete all rows from the table
        cursor.execute("DELETE FROM treat_relations")
        conn.commit()
        print("All rows deleted successfully.")
        
        # Check if any rows remain after deletion
        cursor.execute("SELECT COUNT(*) FROM treat_relations")
        final_count = cursor.fetchone()[0]
        print(f"Number of rows after deletion: {final_count}")
        
        if final_count == 0:
            print("Verification successful: No rows remain in the table.")
        else:
            print(f"Warning: {final_count} rows still exist in the table.")
            
    except sqlite3.Error as e:
        print(f"SQLite error occurred: {e}")
    finally:
        # Close the connection
        conn.close()

if __name__ == "__main__":
    check_and_delete_rows()