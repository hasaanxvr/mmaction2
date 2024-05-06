import psycopg2

# database credentials
dbname = 'activity_recognition_test'
user = 'farhan'
password = 'Xavor123'
host = 'localhost'  
port = 5432  


class database_connection:
    def __init__(self, dbname: str = dbname, user: str = user, password: str = password, host: str = host, port: int = port):
        print('Connecting to database...')
        try:
            self.connection = psycopg2.connect(
                                dbname=dbname,
                                user=user,
                                password=password,
                                host=host,
                                port=port)
            
            self.cursor = self.connection.cursor()

            print('Connection Succesful!')

        except (Exception, psycopg2.Error) as error:
            print("Error while connecting to PostgreSQL database:", error)
            self.connection = None
            

    def execute_query(self, query: str):
        try:
            self.cursor.execute(query)
        except Exception as e:
            print(e)

    
    def get_results(self):
        try:
            rows = self.cursor.fetchall()
            return rows
        except Exception as e:
            print(e)



    def write_to_database(self, data: dict):
        insert_query = """
            INSERT INTO activities
            VALUES (
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
                %(sit)s,
                %(stand)s,
                %(lie_score)s,
                %(walk_score)s
            );
            """
        
        self.cursor.execute(insert_query, data)


    
