import sqlite3

def delete_all_rows():
    # Connect to the SQLite database
    db_path = "data/processed/pubtator.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Execute DELETE query to remove all rows
        cursor.execute("DELETE FROM predictions_reasoning")
        
        # Commit the changes
        conn.commit()
        
        # Print the number of rows deleted
        print(f"Successfully deleted {cursor.rowcount} rows from predictions_reasoning table.")
    
    except sqlite3.Error as e:
        # Roll back any changes if an error occurs
        conn.rollback()
        print(f"Error occurred: {e}")
    
    finally:
        # Close the connection
        conn.close()

# Run the function
if __name__ == "__main__":
    delete_all_rows()