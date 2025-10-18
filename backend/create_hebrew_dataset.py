"""
Hebrew Summarization Dataset Creation
Create training data for Hebrew summarization models
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from typing import List, Dict, Tuple, Optional
import time
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class HebrewDatasetCreator:
    """Create Hebrew summarization datasets from various sources."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def extract_article_from_url(self, url: str) -> Optional[Dict[str, str]]:
        """Extract article content from a URL."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Common article selectors
            article_selectors = [
                'article',
                '.article-body',
                '.post-content',
                '.entry-content',
                '[role="main"]',
                '.content'
            ]
            
            article_content = None
            for selector in article_selectors:
                article_element = soup.select_one(selector)
                if article_element:
                    article_content = article_element
                    break
            
            if not article_content:
                # Fallback to body
                article_content = soup.find('body')
            
            if not article_content:
                return None
            
            # Extract text
            paragraphs = article_content.find_all(['p', 'div'], string=True)
            text_content = []
            
            for p in paragraphs:
                text = p.get_text().strip()
                if len(text) > 50 and self._is_hebrew_text(text):
                    text_content.append(text)
            
            if len(text_content) < 3:
                return None
            
            # Create summary from first paragraph or extract existing summary
            summary = self._extract_summary(soup) or text_content[0]
            full_text = ' '.join(text_content)
            
            # Extract title
            title_element = soup.find('title') or soup.find('h1')
            title = title_element.get_text().strip() if title_element else ""
            
            return {
                'title': title,
                'text': full_text,
                'summary': summary,
                'url': url,
                'length': len(full_text)
            }
            
        except Exception as e:
            logger.error(f"Failed to extract from {url}: {e}")
            return None
    
    def _is_hebrew_text(self, text: str) -> bool:
        """Check if text is primarily Hebrew."""
        hebrew_chars = len(re.findall(r'[\u0590-\u05FF]', text))
        total_chars = len(re.findall(r'\w', text))
        return total_chars > 0 and hebrew_chars / total_chars > 0.5
    
    def _extract_summary(self, soup: BeautifulSoup) -> Optional[str]:
        """Try to extract existing summary from meta tags or article structure."""
        # Try meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            content = meta_desc['content'].strip()
            if self._is_hebrew_text(content) and len(content) > 30:
                return content
        
        # Try Open Graph description
        og_desc = soup.find('meta', attrs={'property': 'og:description'})
        if og_desc and og_desc.get('content'):
            content = og_desc['content'].strip()
            if self._is_hebrew_text(content) and len(content) > 30:
                return content
        
        # Try to find summary in article structure
        summary_selectors = [
            '.summary',
            '.excerpt',
            '.lead',
            '.intro',
            '.article-summary'
        ]
        
        for selector in summary_selectors:
            element = soup.select_one(selector)
            if element:
                content = element.get_text().strip()
                if self._is_hebrew_text(content) and len(content) > 30:
                    return content
        
        return None
    
    def create_dataset_from_urls(self, urls: List[str], output_file: str = "hebrew_summarization_dataset.csv") -> pd.DataFrame:
        """Create dataset from a list of URLs."""
        articles = []
        
        for i, url in enumerate(urls):
            print(f"Processing {i+1}/{len(urls)}: {url}")
            
            article = self.extract_article_from_url(url)
            if article:
                articles.append(article)
            
            # Be respectful to servers
            time.sleep(1)
        
        df = pd.DataFrame(articles)
        
        # Clean and filter
        df = self._clean_dataset(df)
        
        # Save to CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Dataset saved to {output_file} with {len(df)} articles")
        
        return df
    
    def _clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and filter the dataset."""
        # Remove duplicates
        df = df.drop_duplicates(subset=['text'])
        
        # Filter by length
        df = df[df['length'] > 200]  # Minimum article length
        df = df[df['length'] < 5000]  # Maximum article length
        
        # Filter by summary quality
        df = df[df['summary'].str.len() > 30]
        df = df[df['summary'].str.len() < df['text'].str.len() * 0.5]
        
        # Clean text
        df['text'] = df['text'].apply(self._clean_text)
        df['summary'] = df['summary'].apply(self._clean_text)
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """Clean text content."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common web artifacts
        text = re.sub(r'קרא עוד.*?(?=\.|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'לחץ כאן.*?(?=\.|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'פרסומת.*?(?=\.|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'ממומן.*?(?=\.|$)', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def create_synthetic_dataset(self, size: int = 1000) -> pd.DataFrame:
        """Create a synthetic dataset for testing."""
        # This would generate synthetic Hebrew text-summary pairs
        # For demonstration purposes
        
        templates = [
            {
                "text": "בינה מלאכותית מתפתחת במהירות רבה בשנים האחרונות. החברות הטכנולוגיות הגדולות משקיעות מיליארדי דולרים במחקר ופיתוח. הטכנולוגיה משנה תעשיות שלמות ומשפיעה על חיי היומיום שלנו. עם זאת, יש גם חששות מהשלכות אתיות וחברתיות.",
                "summary": "בינה מלאכותית מתפתחת במהירות ומשנה תעשיות, אך מעוררת חששות אתיים."
            },
            {
                "text": "שוק הנדלן בישראל עובר תנודות משמעותיות. מחירי הדירות עלו בעשורים האחרונים באופן דרמטי. הממשלה מנסה להתמודד עם המשבר באמצעות תוכניות שונות. צעירים רבים מתקשים לרכוש דירה ראשונה.",
                "summary": "מחירי הנדלן בישראל עלו דרמטית והממשלה מנסה להתמודד עם המשבר."
            }
        ]
        
        # Generate variations
        articles = []
        for i in range(size):
            template = templates[i % len(templates)]
            articles.append({
                'title': f'כותרת {i+1}',
                'text': template['text'],
                'summary': template['summary'],
                'url': f'https://example.com/article-{i+1}',
                'length': len(template['text'])
            })
        
        return pd.DataFrame(articles)

def main():
    """Main function to create dataset."""
    creator = HebrewDatasetCreator()
    
    # Example URLs (replace with actual Hebrew news sites)
    sample_urls = [
        # Add actual Hebrew news URLs here
        # "https://www.ynet.co.il/articles/...",
        # "https://www.haaretz.co.il/news/...",
    ]
    
    if sample_urls:
        # Create dataset from URLs
        dataset = creator.create_dataset_from_urls(sample_urls)
    else:
        # Create synthetic dataset for testing
        print("Creating synthetic dataset for testing...")
        dataset = creator.create_synthetic_dataset(100)
        dataset.to_csv("synthetic_hebrew_dataset.csv", index=False, encoding='utf-8')
    
    print(f"Dataset created with {len(dataset)} articles")
    print("\nDataset statistics:")
    print(f"Average text length: {dataset['length'].mean():.0f} characters")
    print(f"Average summary length: {dataset['summary'].str.len().mean():.0f} characters")
    print(f"Average compression ratio: {(dataset['summary'].str.len() / dataset['length']).mean():.2%}")

if __name__ == "__main__":
    main()