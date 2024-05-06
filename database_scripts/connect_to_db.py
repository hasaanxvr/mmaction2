import psycopg2

# Replace these variables with your actual database credentials
dbname = 'activity_recognition_test'
user = 'farhan'
password = 'Xavor123'
host = 'localhost'  # Usually 'localhost' if the database is on your local machine
port = 5432  # Usually '5432' for PostgreSQL

try:
    # Establish a connection to the PostgreSQL database
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )

    # Create a cursor object
    cursor = conn.cursor()

    # Execute a SQL query
    cursor.execute("SELECT * from activities")

    # Fetch the result
    record = cursor.fetchone()
    print("You are connected to - ", record)

except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL database:", error)

finally:
    # Close the cursor and connection
    if conn:
        cursor.close()
        conn.close()
        print("PostgreSQL connection is closed")
