import pandas as pd

class ReviewPreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def remove_duplicates_and_missing(self):
        # Remove duplicates
        self.df = self.df.drop_duplicates(subset=['review'])
        # Drop rows with missing review or rating
        self.df = self.df.dropna(subset=['review', 'rating'])
        return self.df

    def normalize_dates(self):
        # Convert dates to YYYY-MM-DD format
        self.df['date'] = pd.to_datetime(self.df['date']).dt.strftime('%Y-%m-%d')
        return self.df

    def save_clean(self, filename="clean_reviews.csv"):
        # Save CSV with selected columns
        self.df.to_csv(filename, index=False, columns=['review','rating','date','bank','source'])
        print(f"Saved {filename}")
