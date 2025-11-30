# Bank Mobile App Reviews Analysis

## Project Overview
Omega Consultancy is supporting banks to improve their mobile apps to enhance customer retention and satisfaction. This project focuses on analyzing user reviews from the Google Play Store for three banks in Ethiopia: CBE, BOA, and Dashen Bank.

The project includes:
- Scraping user reviews.
- Preprocessing and cleaning data.
- Performing sentiment analysis.
- Extracting key themes and keywords.
- Saving results to CSV files for further insights.

---

## Business Objective
As a Data Analyst, the goal is to:
1. Understand user satisfaction and pain points.
2. Identify key drivers of positive and negative sentiment.
3. Provide actionable recommendations to improve app experience.
4. Deliver a report with visualizations and insights.

---

## Tasks Completed

### Task 1: Data Collection and Preprocessing
- Scraped a minimum of 400 reviews per bank (1,200 total).
- Cleaned reviews by removing duplicates and handling missing data.
- Normalized dates to `YYYY-MM-DD` format.
- Saved cleaned data to `task2_with_sentiment.csv`.
- GitHub repository structured with frequent meaningful commits.

**Outcome:** Cleaned CSV dataset with reviews, ratings, dates, bank names, and sources.

---

### Task 2: Sentiment and Thematic Analysis
- Performed sentiment analysis using `distilbert-base-uncased-finetuned-sst-2-english` model (Hugging Face Transformers).  
- Aggregated sentiment by bank and rating.  
- Extracted top keywords for each bank using TF-IDF.  
- Grouped keywords into themes:
  - Dashen: `Transaction Performance`, `Account Access Issues`
  - BOA/CBE: Clustered for further labeling (e.g., login issues, UI, reliability)
