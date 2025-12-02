from src.db_config import PostgresDB

class DBSchema:
    def __init__(self, db: PostgresDB):
        self.db = db

    def create_tables(self):
        create_banks_table = """
        CREATE TABLE IF NOT EXISTS banks (
            bank_id SERIAL PRIMARY KEY,
            bank_name VARCHAR(255) NOT NULL,
            app_name VARCHAR(255)
        );
        """

        create_reviews_table = """
        CREATE TABLE IF NOT EXISTS reviews (
            review_id SERIAL PRIMARY KEY,
            bank_id INT REFERENCES banks(bank_id),
            review_text TEXT,
            rating INT,
            review_date DATE,
            sentiment_label VARCHAR(50),
            sentiment_score FLOAT,
            source VARCHAR(255)
        );
        """

        self.db.execute_query(create_banks_table)
        self.db.execute_query(create_reviews_table)
        print("Tables created successfully.")
