
import psycopg2
from datetime import datetime


data_to_insert = {
    'video_name': 'example_video.mp4',
    'timestamp': datetime.now(),
    'action_1': 'action1',
    'score_1': 0.75,
    'action_2': 'action2',
    'score_2': 0.85,
    'action_3': 'action3',
    'score_3': 0.65,
    'action_4': 'action4',
    'score_4': 0.92,
    'action_5': 'action5',
    'score_5': 0.78,
    'sit': 0.1,
    'stand': 0.1,
    'lie_score': 0.6,
    'walk_score': 0.75
}


insert_query = """
    INSERT INTO activities (
        video_name,
        timestamp,
        action_1,
        score_1,
        action_2,
        score_2,
        action_3,
        score_3,
        action_4,
        score_4,
        action_5,
        score_5,
        sit,
        stand,
        lie_score,
        walk_score
    ) VALUES (
        %(video_name)s,
        %(timestamp)s,
        %(action_1)s,
        %(score_1)s,
        %(action_2)s,
        %(score_2)s,
        %(action_3)s,
        %(score_3)s,
        %(action_4)s,
        %(score_4)s,
        %(action_5)s,
        %(score_5)s,
        %(sit_score)s,
        %(stand_score)s,
        %(lie_score)s,
        %(walk_score)s
    );
"""


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
    cursor.execute(insert_query, data_to_insert)

    conn.commit()
    

    print('Data Inserted Succesfully')

except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL database:", error)

finally:
    # Close the cursor and connection
    if conn:
        cursor.close()
        conn.close()
        print("PostgreSQL connection is closed")
