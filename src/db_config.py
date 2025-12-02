import psycopg2
from psycopg2 import sql

class PostgresDB:
    def __init__(self, host="localhost", port=5432, dbname="bank_reviews", user="postgres", password="your_password"):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.conn = None

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.user,
                password=self.password
            )
            print("Database connection established.")
        except Exception as e:
            print("Error connecting to database:", e)

    def close(self):
        if self.conn:
            self.conn.close()
            print("Database connection closed.")

    def execute_query(self, query, params=None):
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            self.conn.commit()
