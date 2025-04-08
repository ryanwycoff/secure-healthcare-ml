import os
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Database connection settings
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", 5432)  # Default PostgreSQL port
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def get_db_connection():
    """
    Establish a connection to the PostgreSQL database.
    Returns a psycopg2 connection object.
    """
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        logger.info("Database connection established successfully.")
        return conn
    except Exception as e:
        logger.error(f"Error while connecting to database: {e}")
        raise


def close_db_connection(conn):
    """
    Close the database connection.
    """
    if conn:
        conn.close()
        logger.info("Database connection closed.")


def execute_query(query, params=None):
    """
    Executes a SELECT query and returns the result.
    Parameters:
        - query: SQL query string.
        - params: Parameters to pass into the query (default is None).
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        result = cursor.fetchall()
        cursor.close()
        return result
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise
    finally:
        if conn:
            close_db_connection(conn)


def execute_insert(query, params):
    """
    Executes an INSERT query into the database.
    Parameters:
        - query: SQL insert query string.
        - params: Parameters to pass into the query.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()  # Commit the transaction
        cursor.close()
        logger.info(f"Query executed successfully: {query}")
    except Exception as e:
        logger.error(f"Error executing insert query: {e}")
        if conn:
            conn.rollback()  # Rollback in case of error
        raise
    finally:
        if conn:
            close_db_connection(conn)


def create_table():
    """
    Creates a table in the database if it doesn't exist.
    Example table for storing patient data.
    """
    query = """
    CREATE TABLE IF NOT EXISTS patients (
        patient_id SERIAL PRIMARY KEY,
        first_name VARCHAR(100),
        last_name VARCHAR(100),
        dob DATE,
        gender VARCHAR(10),
        address TEXT,
        phone_number VARCHAR(20),
        email VARCHAR(100),
        diagnosis_code VARCHAR(10),
        diagnosis_description TEXT,
        treatment_code VARCHAR(10),
        treatment_description TEXT,
        medication_code VARCHAR(10),
        medication_description TEXT,
        visit_date DATE
    );
    """
    execute_insert(query, ())


def insert_patient_data(patient_data):
    """
    Insert patient data into the patients table.
    Parameters:
        - patient_data: A tuple containing the patient's data.
    """
    query = """
    INSERT INTO patients (
        first_name, last_name, dob, gender, address, phone_number, email,
        diagnosis_code, diagnosis_description, treatment_code, treatment_description,
        medication_code, medication_description, visit_date
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """
    execute_insert(query, patient_data)


def fetch_patient_data(patient_id):
    """
    Fetch patient data by patient_id.
    Parameters:
        - patient_id: The ID of the patient.
    """
    query = """
    SELECT * FROM patients WHERE patient_id = %s;
    """
    result = execute_query(query, (patient_id,))
    return result


# Example usage:
if __name__ == "__main__":
    # Create the table if it doesn't exist
    create_table()

    # Example of inserting patient data
    sample_patient = (
        'John', 'Doe', '1985-06-15', 'Male', '123 Main St, Springfield, IL', '555-1234',
        'johndoe@example.com', 'C34', 'Lung cancer', 'T01', 'Lung resection',
        'A01', 'Cisplatin', '2024-02-10'
    )
    insert_patient_data(sample_patient)

    # Fetch a patient's data by ID
    patient_id = 1
    patient = fetch_patient_data(patient_id)
    print(patient)
