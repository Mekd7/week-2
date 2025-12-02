import pandas as pd
from src.db_config import PostgresDB

class DBInsert:
    def __init__(self, db: PostgresDB):
        self.db = db

    def insert_banks(self, banks: list):
        query = "INSERT INTO banks (bank_name, app_name) VALUES (%s, %s) ON CONFLICT DO NOTHING RETURNING bank_id, bank_name;"
        bank_ids = {}
        with self.db.conn.cursor() as cur:
            for bank_name, app_name in banks:
                cur.execute(query, (bank_name, app_name))
                result = cur.fetchone()
                if result:
                    bank_ids[bank_name] = result[0]
        self.db.conn.commit()
        return bank_ids

    def insert_reviews(self, df: pd.DataFrame, bank_ids: dict):
        insert_query = """
        INSERT INTO reviews (bank_id, review_text, rating, review_date, sentiment_label, sentiment_score, source)
        VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        with self.db.conn.cursor() as cur:
            for _, row in df.iterrows():
                bank_id = bank_ids[row['bank']]
                cur.execute(insert_query, (
                    bank_id,
                    row['review_text'],
                    row['rating'],
                    row['review_date'],
                    row['sentiment_label'],
                    row['sentiment_score'],
                    row['source']
                ))
        self.db.conn.commit()
        print("Reviews inserted successfully.")
