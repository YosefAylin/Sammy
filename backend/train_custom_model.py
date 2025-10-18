"""
Custom Hebrew Summarization Model Training
Fine-tune existing models on Hebrew summarization data
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset, load_dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class HebrewSummarizationTrainer:
    """Train custom Hebrew summarization models."""
    
    def __init__(self, base_model: str = "google/mt5-small"):
        self.base_model = base_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        
        # Training parameters
        self.max_input_length = 512
        self.max_target_length = 150
        
    def prepare_hebrew_dataset(self, data_path: Optional[str] = None) -> Dataset:
        """Prepare Hebrew summarization dataset."""
        
        if data_path and Path(data_path).exists():
            # Load custom dataset
            df = pd.read_csv(data_path)
            texts = df['text'].tolist()
            summaries = df['summary'].tolist()
        else:
            # Create synthetic dataset or use existing Hebrew data
            texts, summaries = self._create_synthetic_dataset()
        
        # Clean and preprocess
        processed_data = []
        for text, summary in zip(texts, summaries):
            if self._is_valid_pair(text, summary):
                processed_data.append({
                    'text': self._preprocess_text(text),
                    'summary': self._preprocess_text(summary)
                })
        
        return Dataset.from_list(processed_data)
    
    def _create_synthetic_dataset(self) -> Tuple[List[str], List[str]]:
        """Create a synthetic Hebrew dataset for demonstration."""
        # This is a placeholder - in practice, you'd use real Hebrew news articles
        sample_data = [
            {
                "text": """
                בינה מלאכותית היא תחום מדעי המתמחה בפיתוח מערכות מחשב המסוגלות לבצע משימות הדורשות בדרך כלל אינטליגנציה אנושית. 
                התחום כולל למידת מכונה, עיבוד שפה טבעית, ראייה ממוחשבת ועוד. בשנים האחרונות חלה התפתחות מהירה בתחום, 
                במיוחד בזכות רשתות נוירונים עמוקות. טכנולוגיות אלו משנות את פני התעשייה והחברה. עם זאת, הן מעוררות גם שאלות 
                אתיות וחברתיות חשובות. חשוב לפתח בינה מלאכותית באופן אחראי ובטוח. המחקר בתחום ממשיך להתקדם במהירות רבה.
                """,
                "summary": "בינה מלאכותית מתפתחת במהירות ומשנה את התעשייה, אך מעוררת שאלות אתיות חשובות."
            },
            {
                "text": """
                שינויי האקלים הם אחד האתגרים הגדולים ביותר של המאה ה-21. הטמפרטורות הגלובליות עולות, הקרחונים נמסים, 
                ורמת הים עולה. השפעות אלו נגרמות בעיקר מפעילות אנושית, במיוחד שריפת דלקים פוסיליים. המדינות ברחבי העולם 
                מנסות להתמודד עם האתגר באמצעות הסכמים בינלאומיים כמו הסכם פריז. יש צורך במעבר לאנרגיות מתחדשות ובהפחתת 
                פליטות גזי חממה. הזמן לפעולה מתקצר, והשלכות חוסר המעש יהיו חמורות.
                """,
                "summary": "שינויי האקלים מהווים אתגר גלובלי הדורש מעבר לאנרגיות מתחדשות והפחתת פליטות."
            }
        ]
        
        texts = [item["text"] for item in sample_data]
        summaries = [item["summary"] for item in sample_data]
        
        # Duplicate for training (in practice, you'd have thousands of examples)
        texts = texts * 100
        summaries = summaries * 100
        
        return texts, summaries
    
    def _is_valid_pair(self, text: str, summary: str) -> bool:
        """Validate text-summary pairs."""
        return (
            len(text.strip()) > 100 and
            len(summary.strip()) > 20 and
            len(summary) < len(text) * 0.8 and
            self._has_hebrew_content(text) and
            self._has_hebrew_content(summary)
        )
    
    def _has_hebrew_content(self, text: str) -> bool:
        """Check if text contains Hebrew characters."""
        hebrew_chars = len(re.findall(r'[\u0590-\u05FF]', text))
        return hebrew_chars > len(text) * 0.3
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for training."""
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove common noise
        noise_patterns = [
            r'קרא עוד.*?(?=\.|$)',
            r'לחץ כאן.*?(?=\.|$)',
            r'פרסומת.*?(?=\.|$)'
        ]
        
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    def tokenize_function(self, examples):
        """Tokenize examples for training."""
        # Add task prefix for T5-style models
        if "t5" in self.base_model.lower():
            inputs = [f"summarize: {text}" for text in examples["text"]]
        else:
            inputs = examples["text"]
        
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_input_length,
            truncation=True,
            padding=True
        )
        
        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples["summary"],
                max_length=self.max_target_length,
                truncation=True,
                padding=True
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def train_model(self, 
                   dataset: Dataset,
                   output_dir: str = "./hebrew_summarizer",
                   num_epochs: int = 3,
                   batch_size: int = 4,
                   learning_rate: float = 5e-5) -> None:
        """Train the summarization model."""
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Tokenize datasets
        train_dataset = Dataset.from_dict({
            'text': [dataset[i]['text'] for i in train_dataset.indices],
            'summary': [dataset[i]['summary'] for i in train_dataset.indices]
        })
        
        val_dataset = Dataset.from_dict({
            'text': [dataset[i]['text'] for i in val_dataset.indices],
            'summary': [dataset[i]['summary'] for i in val_dataset.indices]
        })
        
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        val_dataset = val_dataset.map(self.tokenize_function, batched=True)
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
    
    def evaluate_model(self, test_texts: List[str], test_summaries: List[str]) -> Dict[str, float]:
        """Evaluate the trained model."""
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for text, reference in zip(test_texts, test_summaries):
            # Generate summary
            if "t5" in self.base_model.lower():
                input_text = f"summarize: {text}"
            else:
                input_text = text
                
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=self.max_input_length,
                truncation=True
            )
            
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=self.max_target_length,
                    min_length=30,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
            
            generated_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Calculate ROUGE scores
            rouge_scores = scorer.score(reference, generated_summary)
            
            for metric in scores:
                scores[metric].append(rouge_scores[metric].fmeasure)
        
        # Average scores
        avg_scores = {metric: np.mean(scores[metric]) for metric in scores}
        
        return avg_scores

def create_training_data_from_web():
    """Helper function to create training data from Hebrew websites."""
    # This would scrape Hebrew news sites and create article-summary pairs
    # For demonstration purposes only
    
    sample_urls = [
        "https://www.ynet.co.il",
        "https://www.haaretz.co.il",
        "https://www.mako.co.il"
    ]
    
    # Implementation would involve:
    # 1. Scraping articles
    # 2. Extracting first paragraph as summary
    # 3. Using full article as text
    # 4. Cleaning and validating data
    
    print("This function would scrape Hebrew websites to create training data")
    print("For legal and ethical reasons, implement with proper permissions")

def main():
    """Main training function."""
    # Initialize trainer
    trainer = HebrewSummarizationTrainer("google/mt5-small")
    
    # Prepare dataset
    dataset = trainer.prepare_hebrew_dataset()
    
    # Train model
    trainer.train_model(
        dataset=dataset,
        output_dir="./models/hebrew_summarizer_v1",
        num_epochs=5,
        batch_size=2,  # Small batch size for demo
        learning_rate=3e-5
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()