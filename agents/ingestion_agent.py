import pandas as pd
import json
import os
from pathlib import Path
from .base import BaseAgent, AgentResult

class IngestionAgent(BaseAgent):
    '''
    Ingestion Agent - Data collection and labeling.
    
    Handles:
    1. Loading existing data (from files)
    2. (Optional) Scraping new Reddit data
    3. (Optional) LLM annotation
    
    For research paper: primarily loads prepared labeled data.
    Can be extended to include real scraping/annotation.
    '''
    
    def __init__(self, 
                 data_path: str = None,
                 target_col: str = 'is_fraud',
                 feature_cols: list = None):
        super().__init__ ('IngestionAgent')
        self.data_path = data_path
        self.target_col = target_col
        self.feature_cols = feature_cols
        
    def run(self, state: dict) -> AgentResult:
        self.log('Ingestion: Loading data...')
        
        # Check if data provided directly
        if state.get('data') is not None:
            self.log('Using provided data from state')
            data = state['data']
            
        # Otherwise load from path
        elif self.data_path and os.path.exists(self.data_path):
            self.log(f'Loading from: {self.data_path}')
            if self.data_path.endswith('.csv'):
                data = pd.read_csv(self.data_path)
            elif self.data_path.endswith('.json'):
                data = pd.read_json(self.data_path)
            else:
                return AgentResult(
                    success=False,
                    message=f'Unsupported file format: {self.data_path}'
                )
        else:
            return AgentResult(
                success=False,
                message='No data provided and no valid data_path'
            )
        
        # Handle target column
        target_col = state.get('target_col', self.target_col)
        
        # Handle feature columns
        if self.feature_cols is None:
            self.feature_cols = [c for c in data.columns if c != target_col]
        
        # Clean data - remove uncertain labels (-1)
        if target_col in data.columns:
            original_len = len(data)
            data = data[data[target_col] != -1]
            removed = original_len - len(data)
            if removed > 0:
                self.log(f'Removed {removed} uncertain labels (-1)')
        
        # Basic preprocessing
        data = self._preprocess(data)
        
        self.log(f'Loaded {len(data)} rows, {len(self.feature_cols)} features')
        
        # Show class distribution
        if target_col in data.columns:
            counts = data[target_col].value_counts()
            self.log(f'Class distribution: {counts.to_dict()}')
        
        return AgentResult(
            success=True,
            data={
                'data': data,
                'target_col': target_col,
                'feature_cols': self.feature_cols,
                'original_rows': len(data)
            },
            message=f'Loaded {len(data)} samples',
            metrics={
                'rows': len(data),
                'features': len(self.feature_cols),
                'fraud_count': int((data[target_col] == 1).sum()) if target_col in data.columns else 0,
                'non_fraud_count': int((data[target_col] == 0).sum()) if target_col in data.columns else 0
            }
        )
    
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Basic preprocessing'''
        
        # Handle missing values in numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Handle missing values in categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna('unknown')
        
        return df
    
    def load_scraped_data(self, scraped_dir: str) -> pd.DataFrame:
        '''Load scraped and annotated data from directory'''
        
        # Look for latest annotation file
        annotation_files = list(Path(scraped_dir).glob('outputs/annotations_*.csv'))
        
        if not annotation_files:
            # Try alternative paths
            annotation_files = list(Path(scraped_dir).glob('Data labeling/outputs/annotations_*.csv'))
        
        if annotation_files:
            # Get most recent
            latest = max(annotation_files, key=lambda p: p.stat().st_mtime)
            self.log(f'Loading scraped data from: {latest}')
            return pd.read_csv(latest)
        
        return None


class ScraperAgent(BaseAgent):
    '''
    Scrapes Reddit for fraud-related posts.
    
    Note: Requires Reddit API credentials and rate limit handling.
    This is a placeholder - full implementation needs API keys.
    '''
    
    def __init__(self,
                 subreddits: list = None,
                 keywords: list = None,
                 time_window_days: int = 90):
        super().__init__('ScraperAgent')
        self.subreddits = subreddits or ['Scams', 'Fraud', 'PersonalFinance']
        self.keywords = keywords or [
            'scam', 'fraud', 'cheated', 'scammer', 'fake',
            'cryptocurrency scam', 'investment fraud', 'upi fraud'
        ]
        self.time_window_days = time_window_days
    
    def run(self, state: dict) -> AgentResult:
        self.log('Scraping Reddit for fraud posts...')
        
        # Placeholder - actual scraping requires Reddit API
        # This would use praw (Python Reddit API Wrapper)
        
        scraped_data = self._simulate_scraping()
        
        return AgentResult(
            success=True,
            data={
                'scraped_posts': scraped_data,
                'scraped_count': len(scraped_data)
            },
            message=f'Scraped {len(scraped_data)} posts',
            metrics={'posts_scraped': len(scraped_data)}
        )
    
    def _simulate_scraping(self):
        '''Placeholder - returns empty list'''
        # In real implementation:
        # import praw
        # reddit = praw.Reddit(...)
        # for post in reddit.subreddit('+'.join(self.subreddits)).search(...):
        #     ...
        
        self.log('Note: Real scraping requires Reddit API credentials')
        return []


class AnnotationAgent(BaseAgent):
    '''
    Annotates scraped posts using LLM (Ollama/Llama3).
    
    Analyzes post content and assigns:
    - Fraud label (0/1/-1)
    - Fraud type
    - Payment method
    - Psychological tactics
    '''
    
    def __init__(self,
                 model_name: str = 'llama3',
                 ollama_url: str = 'http://localhost:11434'):
        super().__init__('AnnotationAgent')
        self.model_name = model_name
        self.ollama_url = ollama_url
    
    def run(self, state: dict) -> AgentResult:
        posts = state.get('scraped_posts', [])
        
        if not posts:
            return AgentResult(
                success=True,
                data={'annotations': [], 'annotated_count': 0},
                message='No posts to annotate'
            )
        
        self.log(f'Annotating {len(posts)} posts with LLM...')
        
        annotations = []
        
        # In real implementation, would call Ollama API
        # for post in posts:
        #     annotation = self._annotate(post)
        #     annotations.append(annotation)
        
        self.log('Note: Real annotation requires Ollama running locally')
        
        return AgentResult(
            success=True,
            data={
                'annotations': annotations,
                'annotated_count': len(annotations)
            },
            message=f'Annotated {len(annotations)} posts',
            metrics={'posts_annotated': len(annotations)}
        )
    
    def _annotate(self, post: dict) -> dict:
        '''Call LLM to annotate a single post'''
        
        prompt = f'''
        You are an expert fraud analyst. Analyze this Reddit post and extract:
        
        Post Title: {post.get('title', '')}
        Post Body: {post.get('body', '')}
        
        Output JSON with:
        - is_fraud: 1 (fraud), 0 (non-fraud), -1 (uncertain)
        - fraud_type: category of fraud
        - payment_method: how payment was requested
        - psychological_tactics: urgency, fear, authority, reward scores (0-1)
        - confidence: your confidence in the label (0-1)
        '''
        
        # Would make API call to Ollama here
        # return json.loads(response)
        
        return {}