import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from collections import Counter
import re

class Visualizer:
    def __init__(self, df, bank_col='bank', sentiment_col='sentiment_score', rating_col='rating'):
        # Make a copy and ensure unique index
        self.df = df.copy().reset_index(drop=True)
        self.bank_col = bank_col
        self.sentiment_col = sentiment_col
        self.rating_col = rating_col
        
        print(f"✅ Visualizer initialized with {len(self.df)} rows")
        print(f"📊 Banks: {self.df[bank_col].unique().tolist()}")
        
    def plot_avg_sentiment(self):
        """Plot average sentiment score per bank"""
        print(f"\n📈 Plotting average sentiment...")
        
        # Work on a copy
        df = self.df.copy()
        
        # Check if sentiment column exists
        if self.sentiment_col not in df.columns:
            print(f"❌ Column '{self.sentiment_col}' not found!")
            return None
        
        print(f"📋 Processing column: '{self.sentiment_col}'")
        print(f"📊 Column dtype: {df[self.sentiment_col].dtype}")
        
        # Convert sentiment to numeric values
        def extract_numeric(value):
            try:
                # Handle NaN
                if pd.isna(value):
                    return np.nan
                
                # If it's already numeric
                if pd.api.types.is_number(value):
                    return float(value)
                
                # If it's a string
                if isinstance(value, str):
                    # Try direct conversion
                    try:
                        return float(value)
                    except:
                        # Extract numbers from string
                        numbers = re.findall(r'-?\d+\.?\d*', value)
                        if numbers:
                            return float(numbers[0])
                        return np.nan
                
                # If it's a list/tuple
                if isinstance(value, (list, tuple, np.ndarray)):
                    if len(value) == 0:
                        return np.nan
                    # Get first element
                    first_val = value[0]
                    # Recursively process
                    return extract_numeric(first_val)
                
                return np.nan
            except:
                return np.nan
        
        # Apply extraction
        print("🔧 Extracting numeric sentiment values...")
        numeric_sentiment = df[self.sentiment_col].apply(extract_numeric)
        
        # Check results
        print(f"✅ Extraction complete")
        print(f"📊 Valid values: {numeric_sentiment.notna().sum()}")
        print(f"📊 NaN values: {numeric_sentiment.isna().sum()}")
        
        if numeric_sentiment.notna().sum() == 0:
            print("❌ No valid sentiment values found!")
            return None
        
        # Add to dataframe
        df['numeric_sentiment'] = numeric_sentiment
        
        # Drop NaN values
        df_clean = df.dropna(subset=['numeric_sentiment'])
        
        if df_clean.empty:
            print("❌ No data after cleaning!")
            return None
        
        # Group by bank
        sentiment_by_bank = df_clean.groupby(self.bank_col)['numeric_sentiment'].agg([
            ('mean_sentiment', 'mean'),
            ('count', 'count'),
            ('std', 'std')
        ]).reset_index()
        
        # Sort for better visualization
        sentiment_by_bank = sentiment_by_bank.sort_values('mean_sentiment', ascending=False)
        
        print(f"\n📊 Aggregated sentiment by bank:")
        print(sentiment_by_bank.to_string())
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Create bar positions
        bars = range(len(sentiment_by_bank))
        
        # Plot bars
        colors = ['#2E86AB' if x >= 0 else '#A23B72' for x in sentiment_by_bank['mean_sentiment']]
        plt.bar(bars, sentiment_by_bank['mean_sentiment'], color=colors, edgecolor='black', alpha=0.8)
        
        # Add value labels
        for i, (_, row) in enumerate(sentiment_by_bank.iterrows()):
            plt.text(i, row['mean_sentiment'] + (0.02 if row['mean_sentiment'] >= 0 else -0.05),
                    f"{row['mean_sentiment']:.3f}\n(n={row['count']})",
                    ha='center', va='bottom' if row['mean_sentiment'] >= 0 else 'top',
                    fontsize=9)
        
        # Customize plot
        plt.title('Average Sentiment Score by Bank', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Bank', fontsize=12)
        plt.ylabel('Mean Sentiment Score', fontsize=12)
        plt.xticks(bars, sentiment_by_bank[self.bank_col], rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add sentiment interpretation
        plt.figtext(0.02, 0.02, 
                   "Sentiment Scale:\n> 0.5: Very Positive\n0 to 0.5: Positive\n-0.5 to 0: Negative\n< -0.5: Very Negative",
                   fontsize=9, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
        
        plt.tight_layout()
        plt.show()
        
        return sentiment_by_bank
    
    def plot_rating_distribution(self):
        """Plot rating distribution per bank"""
        if self.rating_col not in self.df.columns:
            print(f"⚠️ Rating column '{self.rating_col}' not found")
            return
        
        df = self.df.copy()
        
        # Clean rating values
        def clean_rating(val):
            try:
                if pd.isna(val):
                    return np.nan
                if pd.api.types.is_number(val):
                    return float(val)
                if isinstance(val, str):
                    numbers = re.findall(r'\d+\.?\d*', val)
                    if numbers:
                        return float(numbers[0])
                return np.nan
            except:
                return np.nan
        
        df['clean_rating'] = df[self.rating_col].apply(clean_rating)
        df = df.dropna(subset=['clean_rating'])
        
        if df.empty:
            print("❌ No valid ratings found")
            return
        
        # Create subplot for each bank
        banks = df[self.bank_col].unique()
        
        fig, axes = plt.subplots(1, len(banks), figsize=(5*len(banks), 5), sharey=True)
        if len(banks) == 1:
            axes = [axes]
        
        for ax, bank in zip(axes, banks):
            bank_data = df[df[self.bank_col] == bank]['clean_rating']
            
            # Create histogram
            ax.hist(bank_data, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Add mean line
            mean_val = bank_data.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            
            ax.set_title(f"{bank}\n(n={len(bank_data)})", fontsize=14)
            ax.set_xlabel('Rating')
            ax.legend()
            ax.grid(alpha=0.3)
        
        axes[0].set_ylabel('Frequency')
        plt.suptitle('Rating Distribution by Bank', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    
    def generate_wordclouds(self, text_col='review', max_words=100):
        """Generate word clouds for each bank"""
        df = self.df.copy()
        
        # Preprocess text
        def preprocess(text):
            if pd.isna(text):
                return ""
            text = str(text).lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        df['clean_text'] = df[text_col].apply(preprocess)
        
        banks = df[self.bank_col].unique()
        
        fig, axes = plt.subplots(1, len(banks), figsize=(5*len(banks), 5))
        if len(banks) == 1:
            axes = [axes]
        
        for ax, bank in zip(axes, banks):
            bank_texts = df[df[self.bank_col] == bank]['clean_text']
            all_text = ' '.join(bank_texts.tolist())
            
            if not all_text.strip():
                ax.text(0.5, 0.5, f"No text data\nfor {bank}", ha='center', va='center')
                ax.axis('off')
                continue
            
            # Generate wordcloud
            wordcloud = WordCloud(width=400, height=300, 
                                background_color='white',
                                max_words=max_words).generate(all_text)
            
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f"{bank}", fontsize=14)
            ax.axis('off')
        
        plt.suptitle('Word Clouds by Bank', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    
    def plot_sentiment_vs_rating(self):
        """Plot correlation between sentiment and rating"""
        if self.rating_col not in self.df.columns or self.sentiment_col not in self.df.columns:
            print("❌ Missing sentiment or rating column")
            return
        
        # This method would require both columns to be numeric
        # You can implement it similarly to plot_avg_sentiment
        print("⚠️ This method requires both sentiment and rating to be numeric")
        print("   Ensure sentiment_score and rating columns contain numeric values")

# Helper function for quick sentiment analysis
def quick_sentiment_analysis(df, text_col='review'):
    """Simple sentiment analysis as fallback"""
    from textblob import TextBlob
    
    def get_sentiment(text):
        if pd.isna(text):
            return 0.0
        try:
            blob = TextBlob(str(text))
            return blob.sentiment.polarity
        except:
            return 0.0
    
    return df[text_col].apply(get_sentiment)