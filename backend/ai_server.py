#!/usr/bin/env python3
"""
AI-Powered Sammy Backend Server
Using AlephBERT for Hebrew text understanding and advanced summarization
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import re
import logging
from functools import lru_cache
import hashlib
from typing import List, Dict, Optional, Tuple
import time
from model_manager import ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS for Chrome extensions
CORS(app, origins=["chrome-extension://*", "moz-extension://*"])

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

class AIHebrewSummarizer:
    """AI-powered Hebrew summarizer using AlephBERT and advanced techniques."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model manager
        self.model_manager = ModelManager()
        
        # Initialize models
        self.models_loaded = False
        self.alephbert_tokenizer = None
        self.alephbert_model = None
        self.abstractive_tokenizer = None
        self.abstractive_model = None
        
        # Hebrew linguistic features
        self.hebrew_stopwords = {
            'של', 'את', 'על', 'אל', 'עם', 'בין', 'לפני', 'אחרי', 'תחת', 'מעל',
            'זה', 'זו', 'זאת', 'אלה', 'אלו', 'הוא', 'היא', 'הם', 'הן', 'אני',
            'אתה', 'את', 'אנחנו', 'אתם', 'אתן', 'כל', 'כמה', 'איזה', 'איך',
            'מה', 'מי', 'איפה', 'מתי', 'למה', 'כן', 'לא', 'גם', 'רק', 'עוד'
        }
        
        self.connectors = [
            'עם זאת', 'לכן', 'בנוסף', 'אך', 'אולם', 'לפיכך', 'למרות זאת',
            'ולכן', 'כמו כן', 'כתוצאה מכך', 'מאידך', 'מצד שני', 'לפחות',
            'בעקבות זאת', 'באופן דומה', 'לסיכום', 'באופן כללי', 'למעשה'
        ]
        
        # Cache for embeddings
        self.embedding_cache = {}
        
    def load_models(self):
        """Load AI models on first use."""
        if self.models_loaded:
            return
            
        try:
            logger.info("Loading AlephBERT model...")
            # Get AlephBERT path (download if needed)
            alephbert_path = self.model_manager.get_model_path('alephbert')
            self.alephbert_tokenizer = AutoTokenizer.from_pretrained(alephbert_path)
            self.alephbert_model = AutoModel.from_pretrained(alephbert_path).to(self.device)
            self.alephbert_model.eval()
            
            logger.info("Loading multilingual summarization model...")
            # Get mT5 path (download if needed)
            mt5_path = self.model_manager.get_model_path('mt5')
            self.abstractive_tokenizer = AutoTokenizer.from_pretrained(mt5_path)
            self.abstractive_model = AutoModelForSeq2SeqLM.from_pretrained(mt5_path).to(self.device)
            self.abstractive_model.eval()
            
            self.models_loaded = True
            logger.info("✅ All AI models loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise
    
    def preprocess_hebrew_text(self, text: str) -> str:
        """Clean and preprocess Hebrew text."""
        # Remove web artifacts
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        
        # Remove common noise
        noise_patterns = [
            r'קרא עוד.*?(?=\.|$)',
            r'לחץ כאן.*?(?=\.|$)',
            r'פרסומת.*?(?=\.|$)',
            r'ממומן.*?(?=\.|$)',
            r'מידע נוסף.*?(?=\.|$)',
            r'תגובות.*?(?=\.|$)'
        ]
        
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def split_sentences(self, text: str) -> List[str]:
        """Advanced Hebrew sentence splitting."""
        # Handle Hebrew-specific punctuation
        text = re.sub(r'(\w+)\.(\w+)', r'\1.\2', text)  # Handle abbreviations
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[א-ת])|(?<=[.!?])\s+(?=[A-Z])', text)
        
        processed_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Filter sentences
            if (20 <= len(sentence) <= 500 and 
                not self._is_noise_sentence(sentence) and
                self._has_hebrew_content(sentence)):
                processed_sentences.append(sentence)
        
        return processed_sentences
    
    def _is_noise_sentence(self, sentence: str) -> bool:
        """Detect noise sentences."""
        noise_patterns = [
            r'^\d+\.\s*$',
            r'^[^\u0590-\u05FF]*$',
            r'(קרא עוד|לחץ כאן|מידע נוסף|פרסומת|ממומן)',
            r'^\W+$'
        ]
        return any(re.search(pattern, sentence) for pattern in noise_patterns)
    
    def _has_hebrew_content(self, text: str) -> bool:
        """Check if text has sufficient Hebrew content."""
        hebrew_chars = len(re.findall(r'[\u0590-\u05FF]', text))
        return hebrew_chars > len(text) * 0.3
    
    @lru_cache(maxsize=128)
    def get_sentence_embeddings(self, sentences_tuple: Tuple[str, ...]) -> np.ndarray:
        """Get AlephBERT embeddings for sentences with caching."""
        sentences = list(sentences_tuple)
        
        # Create cache key
        cache_key = hashlib.md5(''.join(sentences).encode()).hexdigest()
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        embeddings = []
        batch_size = 8
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            inputs = self.alephbert_tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
                return_attention_mask=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.alephbert_model(**inputs)
                # Use attention-weighted pooling
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                token_embeddings = outputs.last_hidden_state
                weighted_embeddings = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
                embeddings.append(weighted_embeddings.cpu().numpy())
        
        result = np.vstack(embeddings)
        self.embedding_cache[cache_key] = result
        return result
    
    def calculate_sentence_scores(self, sentences: List[str], embeddings: np.ndarray) -> List[Tuple[float, str, int]]:
        """Calculate comprehensive sentence scores."""
        scores = []
        
        # TextRank with embeddings
        similarity_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(similarity_matrix, 0)
        
        # Adaptive threshold
        threshold = np.percentile(similarity_matrix.flatten(), 70)
        similarity_matrix = np.where(similarity_matrix < threshold, 0, similarity_matrix)
        
        # PageRank
        graph = nx.from_numpy_array(similarity_matrix)
        pagerank_scores = nx.pagerank(graph, alpha=0.85, max_iter=100)
        
        for i, sentence in enumerate(sentences):
            score = 0
            
            # 1. PageRank score (40%)
            score += pagerank_scores.get(i, 0) * 0.4
            
            # 2. Position score (25%)
            position_score = (len(sentences) - i) / len(sentences)
            score += position_score * 0.25
            
            # 3. Length score (15%)
            optimal_length = 150
            length_score = 1 - abs(len(sentence) - optimal_length) / optimal_length
            score += max(0, length_score) * 0.15
            
            # 4. Hebrew content score (10%)
            hebrew_chars = len(re.findall(r'[\u0590-\u05FF]', sentence))
            hebrew_ratio = hebrew_chars / len(sentence) if len(sentence) > 0 else 0
            score += hebrew_ratio * 0.1
            
            # 5. Content quality score (10%)
            content_score = 0
            if re.search(r'\d', sentence):  # Contains numbers
                content_score += 0.3
            if sentence.endswith('?'):  # Question
                content_score += 0.2
            if any(word in sentence for word in ['חשוב', 'עיקרי', 'מרכזי']):  # Important words
                content_score += 0.5
            score += min(content_score, 1.0) * 0.1
            
            scores.append((score, sentence, i))
        
        return sorted(scores, key=lambda x: x[0], reverse=True)
    
    def extractive_summarize(self, text: str, top_n: int = 6) -> Dict[str, any]:
        """Advanced extractive summarization using AlephBERT."""
        start_time = time.time()
        
        try:
            # Preprocess and split
            clean_text = self.preprocess_hebrew_text(text)
            sentences = self.split_sentences(clean_text)
            
            if len(sentences) < 3:
                return {
                    "summary": clean_text,
                    "metadata": {
                        "method": "no_summarization_needed",
                        "original_sentences": len(sentences),
                        "processing_time": time.time() - start_time
                    }
                }
            
            # Get embeddings
            embeddings = self.get_sentence_embeddings(tuple(sentences))
            
            # Score sentences
            scored_sentences = self.calculate_sentence_scores(sentences, embeddings)
            
            # Select top sentences
            adaptive_top_n = min(top_n, max(3, len(sentences) // 3))
            selected = scored_sentences[:adaptive_top_n]
            
            # Order by original position
            selected.sort(key=lambda x: x[2])
            
            # Smart ordering (avoid connectors at start)
            sentences_only = [s[1] for s in selected]
            start_sentence = next(
                (s for s in sentences_only if not any(s.startswith(c) for c in self.connectors)),
                sentences_only[0]
            )
            
            if start_sentence in sentences_only:
                sentences_only.remove(start_sentence)
                final_sentences = [start_sentence] + sentences_only
            else:
                final_sentences = sentences_only
            
            summary = ' '.join(final_sentences)
            
            return {
                "summary": summary,
                "metadata": {
                    "method": "alephbert_extractive",
                    "original_sentences": len(sentences),
                    "summary_sentences": len(final_sentences),
                    "compression_ratio": len(final_sentences) / len(sentences),
                    "processing_time": time.time() - start_time,
                    "model": "AlephBERT + Advanced TextRank"
                }
            }
            
        except Exception as e:
            logger.error(f"Extractive summarization failed: {e}")
            return self._fallback_summary(text, start_time)
    
    def abstractive_summarize(self, text: str, max_length: int = 150) -> Dict[str, any]:
        """Abstractive summarization using mT5."""
        start_time = time.time()
        
        try:
            clean_text = self.preprocess_hebrew_text(text)
            
            # Prepare input for mT5
            input_text = f"summarize: {clean_text}"
            
            inputs = self.abstractive_tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                summary_ids = self.abstractive_model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    min_length=30,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            summary = self.abstractive_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            return {
                "summary": summary,
                "metadata": {
                    "method": "mt5_abstractive",
                    "original_length": len(clean_text),
                    "summary_length": len(summary),
                    "compression_ratio": len(summary) / len(clean_text),
                    "processing_time": time.time() - start_time,
                    "model": "mT5-small"
                }
            }
            
        except Exception as e:
            logger.error(f"Abstractive summarization failed: {e}")
            return self._fallback_summary(text, start_time)
    
    def _fallback_summary(self, text: str, start_time: float) -> Dict[str, any]:
        """Fallback to simple extraction if AI fails."""
        sentences = self.split_sentences(text)
        fallback = '. '.join(sentences[:3]) + '.'
        
        return {
            "summary": fallback,
            "metadata": {
                "method": "fallback_extraction",
                "processing_time": time.time() - start_time,
                "error": "AI model failed, used fallback"
            }
        }

# Initialize AI summarizer
ai_summarizer = AIHebrewSummarizer()

@app.route("/summarize", methods=["POST", "OPTIONS"])
def summarize_api():
    """AI-powered summarization API."""
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
    
    try:
        # Load models on first request
        ai_summarizer.load_models()
        
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Empty text provided"}), 400
        
        # Parameters
        method = data.get("method", "extractive")  # "extractive" or "abstractive"
        top_n = data.get("top_n", 6)
        max_length = data.get("max_length", 150)
        
        # Choose summarization method
        if method == "abstractive":
            result = ai_summarizer.abstractive_summarize(text, max_length)
        else:
            result = ai_summarizer.extractive_summarize(text, top_n)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": f"Summarization failed: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check with model status."""
    return jsonify({
        "status": "healthy",
        "models_loaded": ai_summarizer.models_loaded,
        "device": str(ai_summarizer.device),
        "methods": ["extractive", "abstractive"],
        "version": "AI-powered v3.0"
    })

@app.route("/models/status", methods=["GET"])
def models_status():
    """Get detailed model status."""
    return jsonify({
        "models": ai_summarizer.model_manager.list_models(),
        "models_loaded": ai_summarizer.models_loaded,
        "device": str(ai_summarizer.device)
    })

@app.route("/models/download", methods=["POST"])
def download_models():
    """Download all required models."""
    try:
        success = ai_summarizer.model_manager.download_all_models()
        if success:
            return jsonify({"status": "All models downloaded successfully"})
        else:
            return jsonify({"error": "Failed to download some models"}), 500
    except Exception as e:
        return jsonify({"error": f"Download failed: {str(e)}"}), 500

@app.route("/load_models", methods=["POST"])
def load_models():
    """Preload models endpoint."""
    try:
        ai_summarizer.load_models()
        return jsonify({"status": "Models loaded successfully"})
    except Exception as e:
        return jsonify({"error": f"Failed to load models: {str(e)}"}), 500

if __name__ == "__main__":
    print("🤖 Starting AI-Powered Sammy Hebrew Summarizer...")
    print("📍 Server running on: http://localhost:5002")
    print("🧠 Models: AlephBERT + mT5")
    print("⚡ Methods: Extractive & Abstractive")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 50)
    
    app.run(host='0.0.0.0', port=5002, debug=True)