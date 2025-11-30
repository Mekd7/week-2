from google_play_scraper import reviews
import pandas as pd

class ReviewScraper:
    def __init__(self, banks: dict):
        """
        banks: dictionary of {bank_name: app_id}
        Example:
        {
            "CBE": "com.cbe.mobile",
            "BOA": "com.boa.mobile",
            "Dashen": "com.dashen.mobile"
        }
        """
        self.banks = banks

    def scrape_reviews(self, count_per_bank=500):
        all_reviews = []
        for bank_name, app_id in self.banks.items():
            print(f"Scraping reviews for {bank_name}...")
            try:
                app_reviews, _ = reviews(
                    app_id,
                    lang='en',
                    country='et',
                    count=count_per_bank
                )
                for r in app_reviews:
                    all_reviews.append({
                        "review": r['content'],
                        "rating": r['score'],
                        "date": r['at'],
                        "bank": bank_name,
                        "source": "Google Play"
                    })
            except Exception as e:
                print(f"Error scraping {bank_name}: {e}")
        
        df = pd.DataFrame(all_reviews)
        print(f"Total reviews scraped: {len(df)}")
        return df

    def save_reviews(self, df, filename="raw_reviews.csv"):
        df.to_csv(filename, index=False)
        print(f"Saved {filename}")
