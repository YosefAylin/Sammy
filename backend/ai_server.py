#!/usr/bin/env python3
"""
AI-Powered Sammy Backend Server
Using AlephBERT for Hebrew text understanding and advanced summarization
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModel
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
            
            logger.info("✅ AlephBERT model loaded successfully!")
            
            self.models_loaded = True
            logger.info("✅ All AI models loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise
    
    def preprocess_hebrew_text(self, text: str) -> str:
        """Clean and preprocess Hebrew text, removing metadata and noise."""
        # Remove web artifacts
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        
        # Light noise patterns including betting content
        noise_patterns = [
            # Clear UI elements
            r'\bקרא עוד\b',
            r'\bלחץ כאן\b',
            r'\bממומן\b',
            r'\bTaboola\b',
            r'\boutbrain\b',
            
            # Betting and odds content
            r'\bיתרון\b.*?\d+\.\d+',
            r'\bרגיל\b.*?\d+\.\d+',
            r'\bמעל/מתחת\b',
            r'\d+\.\d+X\d+\.\d+',
            r'מתוך \d+ משחקים',
            
            # Standalone URLs and emails
            r'\bwww\.\S+',
            r'\bhttp\S+',
            r'\S+@\S+\.\S+'
        ]
        
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove multiple spaces again after cleaning
        text = re.sub(r'\s+', ' ', text)
        
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
        """Detect noise sentences including metadata."""
        noise_patterns = [
            # Only clear noise patterns
            r'^\d+\.\s*$',  # Numbers only
            r'^\W+$',       # Symbols only
            r'^.{1,15}$',   # Very short
            
            # Clear UI elements (exact matches)
            r'^(קרא עוד|לחץ כאן|שתף|לייק)$',
            
            # Standalone external services
            r'^(taboola|outbrain|sponsored)$',
            
            # Standalone URLs/emails
            r'^\S+@\S+\.\S+$',
            r'^(www\.|http)',
            
            # Repeated patterns
            r'^(.+?)\1+$'
        ]
        return any(re.search(pattern, sentence, re.IGNORECASE) for pattern in noise_patterns)
    
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
        batch_size = 4  # Reduced for faster processing
        
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
    
    def comprehensive_extractive_summarize(self, text: str, target_ratio: float = 0.35, title: str = "") -> Dict[str, any]:
        """
        Comprehensive extractive summarization following the guidelines:
        - Captures main ideas from opening, middle, and closing sections
        - Extracts essential content avoiding repetitions
        - Creates flowing, coherent summary representing entire text
        - Maintains 25-40% of original length
        """
        start_time = time.time()
        
        # Debug logging
        logger.info(f"Starting comprehensive_extractive_summarize with target_ratio={target_ratio:.2f}")
        
        try:
            # 1. Preprocess and analyze text structure
            clean_text = self.preprocess_hebrew_text(text)
            sentences = self.split_sentences(clean_text)
            
            # Quick processing for short texts
            if len(sentences) < 5:
                return self._create_short_summary(clean_text, sentences, start_time)
            
            # Ultra-short summary for very low target ratio
            if target_ratio <= 0.25:
                return self._create_ultra_short_summary(sentences, target_ratio, start_time, title)
            
            # Fast processing for medium texts
            if len(sentences) < 15:
                return self._fast_extractive_summarize(sentences, target_ratio, start_time, title)
            
            # 2. Identify central theme from title and opening sentences
            central_theme = self._identify_central_theme(sentences[:3], title)
            logger.info(f"Identified central theme keywords: {central_theme}")
            
            # 3. Identify sections (opening, middle, closing)
            sections = self._identify_text_sections(sentences)
            
            # 4. Get embeddings for all sentences
            embeddings = self.get_sentence_embeddings(tuple(sentences))
            
            # 5. Score sentences with theme-aware algorithm
            scored_sentences = self._theme_aware_sentence_scoring(
                sentences, embeddings, sections, central_theme
            )
            
            # 6. Select sentences ensuring representation from all sections
            selected_sentences = self._select_representative_sentences(
                scored_sentences, sections, target_ratio
            )
            
            # 7. Ensure central theme appears at the beginning
            selected_sentences = self._prioritize_theme_in_opening(selected_sentences, central_theme)
            
            # 8. Reorder and create flowing narrative
            final_summary = self._create_flowing_summary(selected_sentences, sentences)
            
            # 8. Post-process for coherence
            polished_summary = self._polish_summary_flow(final_summary)
            
            return {
                "summary": polished_summary,
                "metadata": {
                    "method": "comprehensive_extractive",
                    "original_sentences": len(sentences),
                    "summary_sentences": len(selected_sentences),
                    "compression_ratio": len(selected_sentences) / len(sentences),
                    "sections_represented": {
                        "opening": len([s for s in selected_sentences if s[2] in sections['opening']]),
                        "middle": len([s for s in selected_sentences if s[2] in sections['middle']]),
                        "closing": len([s for s in selected_sentences if s[2] in sections['closing']])
                    },
                    "processing_time": time.time() - start_time,
                    "model": "AlephBERT + Comprehensive Algorithm"
                }
            }
            
        except Exception as e:
            logger.error(f"Comprehensive summarization failed: {e}")
            return self._fallback_summary(text, start_time)
    
    def _identify_text_sections(self, sentences: List[str]) -> Dict[str, List[int]]:
        """Identify opening, middle, and closing sections."""
        total = len(sentences)
        opening_end = max(2, int(total * 0.25))
        closing_start = min(total - 2, int(total * 0.75))
        
        return {
            'opening': list(range(0, opening_end)),
            'middle': list(range(opening_end, closing_start)),
            'closing': list(range(closing_start, total))
        }
    
    def _comprehensive_sentence_scoring(self, sentences: List[str], embeddings: np.ndarray, 
                                      sections: Dict[str, List[int]]) -> List[Tuple[float, str, int]]:
        """Enhanced scoring ensuring representation from all sections."""
        # Base TextRank scoring
        base_scores = self.calculate_sentence_scores(sentences, embeddings)
        
        # Section-aware enhancement
        enhanced_scores = []
        
        for score, sentence, idx in base_scores:
            enhanced_score = score
            
            # 1. Section representation bonus
            if idx in sections['opening']:
                enhanced_score *= 1.2  # Boost opening sentences
            elif idx in sections['closing']:
                enhanced_score *= 1.15  # Boost closing sentences
            
            # 2. Key insight detection
            if self._contains_key_insights(sentence):
                enhanced_score *= 1.3
            
            # 3. Transition and connector detection
            if self._is_transition_sentence(sentence):
                enhanced_score *= 1.1
            
            # 4. Avoid redundancy penalty
            redundancy_penalty = self._calculate_redundancy_penalty(
                sentence, [s[1] for s in enhanced_scores]
            )
            enhanced_score *= (1 - redundancy_penalty)
            
            enhanced_scores.append((enhanced_score, sentence, idx))
        
        return sorted(enhanced_scores, key=lambda x: x[0], reverse=True)
    
    def _contains_key_insights(self, sentence: str) -> bool:
        """Detect sentences containing key insights or main ideas."""
        key_indicators = [
            r'(עיקר|מרכזי|חשוב|בעיקר|בעיקרון)',
            r'(לכן|לפיכך|כתוצאה|בעקבות)',
            r'(לסיכום|בסופו של דבר|לבסוף)',
            r'(הסיבה|הגורם|הבעיה|הפתרון)',
            r'(מחקר|ממצא|תוצאה|מסקנה)',
            r'(החלטה|החליט|קבע|קובע)'
        ]
        return any(re.search(pattern, sentence, re.IGNORECASE) for pattern in key_indicators)
    
    def _is_transition_sentence(self, sentence: str) -> bool:
        """Detect sentences that provide narrative flow."""
        transition_patterns = [
            r'^(עם זאת|אולם|אך|למרות זאת)',
            r'^(בנוסף|כמו כן|יתר על כן)',
            r'^(מאידך|מצד שני|לעומת זאת)',
            r'(באופן כללי|בדרך כלל|למעשה)'
        ]
        return any(re.search(pattern, sentence, re.IGNORECASE) for pattern in transition_patterns)
    
    def _calculate_redundancy_penalty(self, sentence: str, existing_sentences: List[str]) -> float:
        """Calculate penalty for redundant content."""
        if not existing_sentences:
            return 0.0
        
        # Simple word overlap check
        sentence_words = set(sentence.lower().split())
        max_overlap = 0
        
        for existing in existing_sentences:
            existing_words = set(existing.lower().split())
            overlap = len(sentence_words & existing_words) / len(sentence_words | existing_words)
            max_overlap = max(max_overlap, overlap)
        
        return min(max_overlap * 0.5, 0.3)  # Cap penalty at 30%
    
    def _select_representative_sentences(self, scored_sentences: List[Tuple[float, str, int]], 
                                       sections: Dict[str, List[int]], 
                                       target_ratio: float) -> List[Tuple[float, str, int]]:
        """Select sentences ensuring representation from all sections."""
        total_sentences = len(scored_sentences)
        target_count = max(2, int(total_sentences * target_ratio))
        
        # Debug logging
        logger.info(f"Selection: total_sentences={total_sentences}, target_ratio={target_ratio:.2f}, target_count={target_count}")
        
        # Flexible section representation based on target count
        if target_count <= 3:
            # For small summaries, prioritize best sentences regardless of section
            return scored_sentences[:target_count]
        
        # For larger summaries, ensure some section diversity
        min_opening = max(1, target_count // 5)
        min_closing = max(1, target_count // 6)
        min_middle = max(1, target_count - min_opening - min_closing)
        
        selected = []
        section_counts = {'opening': 0, 'middle': 0, 'closing': 0}
        
        # First pass: ensure minimum representation
        for score, sentence, idx in scored_sentences:
            current_section = None
            if idx in sections['opening']:
                current_section = 'opening'
            elif idx in sections['middle']:
                current_section = 'middle'
            elif idx in sections['closing']:
                current_section = 'closing'
            
            if current_section:
                min_needed = {'opening': min_opening, 'middle': min_middle, 'closing': min_closing}
                if section_counts[current_section] < min_needed[current_section]:
                    selected.append((score, sentence, idx))
                    section_counts[current_section] += 1
                    
                    if len(selected) >= target_count:
                        break
        
        # Second pass: fill remaining slots with highest scores
        remaining_slots = target_count - len(selected)
        selected_indices = {s[2] for s in selected}
        
        for score, sentence, idx in scored_sentences:
            if remaining_slots <= 0:
                break
            if idx not in selected_indices:
                selected.append((score, sentence, idx))
                remaining_slots -= 1
        
        # Debug logging
        logger.info(f"Selected {len(selected)} sentences (target was {target_count})")
        
        return selected
    
    def _prioritize_theme_in_opening(self, selected_sentences: List[Tuple[float, str, int]], 
                                   theme_keywords: List[str]) -> List[Tuple[float, str, int]]:
        """Ensure sentences with central theme appear early in the summary."""
        if not theme_keywords:
            return selected_sentences
        
        theme_sentences = []
        other_sentences = []
        
        for score, sentence, idx in selected_sentences:
            sentence_lower = sentence.lower()
            has_theme = any(keyword.lower() in sentence_lower for keyword in theme_keywords)
            
            if has_theme:
                theme_sentences.append((score, sentence, idx))
            else:
                other_sentences.append((score, sentence, idx))
        
        # Sort theme sentences by original position (to maintain flow)
        theme_sentences.sort(key=lambda x: x[2])
        other_sentences.sort(key=lambda x: x[2])
        
        # Combine: theme sentences first, then others
        return theme_sentences + other_sentences
    
    def _create_ultra_short_summary(self, sentences: List[str], target_ratio: float, 
                                   start_time: float, title: str = "") -> Dict[str, any]:
        """Create short summary (3-5 sentences) focusing on main theme."""
        central_theme = self._identify_central_theme(sentences[:3], title)
        
        # For short summary, we want 3-5 sentences based on text length
        base_count = int(len(sentences) * target_ratio)
        target_count = min(5, max(3, base_count))
        
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Theme relevance score
            theme_score = 0
            for keyword in central_theme:
                if keyword.lower() in sentence_lower:
                    theme_score += 1.5  # Good weight for theme
            
            # Position score (distribute across text)
            if i < 3:
                pos_score = 2.5  # Opening is important
            elif i >= len(sentences) - 3:
                pos_score = 2.0  # Closing is also important
            else:
                pos_score = 1.5  # Middle content
            
            # Length score (prefer informative sentences)
            optimal_length = 120
            len_score = 1.0 - abs(len(sentence) - optimal_length) / optimal_length
            len_score = max(0.3, len_score) * 1.5
            
            # Hebrew content score
            hebrew_ratio = len([c for c in sentence if '\u0590' <= c <= '\u05FF']) / len(sentence)
            
            # Information density (avoid very simple sentences)
            word_count = len(sentence.split())
            info_score = min(1.5, word_count / 15) if word_count > 5 else 0.5
            
            total_score = theme_score + pos_score + len_score + hebrew_ratio + info_score
            scored_sentences.append((total_score, sentence, i))
        
        # Select top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        selected = scored_sentences[:target_count]
        
        # Sort by original position for flow
        selected.sort(key=lambda x: x[2])
        summary = ' '.join([s[1] for s in selected])
        
        return {
            "summary": summary,
            "metadata": {
                "method": "ultra_short_extractive",
                "original_sentences": len(sentences),
                "summary_sentences": len(selected),
                "compression_ratio": len(selected) / len(sentences),
                "central_theme": central_theme,
                "processing_time": time.time() - start_time,
                "note": f"Short summary with {len(selected)} sentences focusing on main theme"
            }
        }
    
    def _identify_central_theme(self, opening_sentences: List[str], title: str = "") -> List[str]:
        """Identify central theme keywords from title and opening sentences."""
        theme_keywords = []
        
        # Combine title and opening sentences (title gets priority)
        combined_text = f"{title} {' '.join(opening_sentences)}".lower()
        
        # Hebrew keywords that often indicate main themes
        theme_indicators = [
            # Political/social themes
            'משטרה', 'ממשלה', 'כנסת', 'בית משפט', 'חוק', 'מדיניות',
            'הפגנה', 'מחאה', 'עצרת', 'שביתה',
            
            # Economic themes  
            'כלכלה', 'משק', 'תקציב', 'מס', 'שכר', 'מחירים', 'יוקר',
            
            # Security themes
            'ביטחון', 'צבא', 'מלחמה', 'טרור', 'איום', 'פיגוע',
            
            # Social themes
            'חברה', 'חינוך', 'בריאות', 'דיור', 'תחבורה', 'סביבה',
            
            # Sports themes
            'כדורגל', 'ספורט', 'משחק', 'קבוצה', 'שחקן', 'מאמן', 'ליגה'
        ]
        
        # Find theme keywords in combined text
        for keyword in theme_indicators:
            if keyword in combined_text:
                theme_keywords.append(keyword)
        
        # Extract key entities (proper nouns) - prioritize title
        import re
        title_entities = re.findall(r'\b[א-ת][א-ת]+(?:\s+[א-ת][א-ת]+)*\b', title.lower()) if title else []
        text_entities = re.findall(r'\b[א-ת][א-ת]+(?:\s+[א-ת][א-ת]+)*\b', combined_text)
        
        # Add significant entities (prioritize title entities)
        all_entities = title_entities + text_entities
        for entity in all_entities:
            if len(entity) > 3 and entity not in theme_keywords:
                theme_keywords.append(entity)
        
        return theme_keywords[:7]  # Return top 7 theme elements
    
    def _theme_aware_sentence_scoring(self, sentences: List[str], embeddings: np.ndarray, 
                                    sections: Dict[str, List[int]], theme_keywords: List[str]) -> List[Tuple[float, str, int]]:
        """Score sentences with awareness of central theme."""
        scores = []
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Base score from embeddings (calculate similarity to all sentences)
            sentence_embedding = embeddings[i]
            similarities = np.dot(embeddings, sentence_embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(sentence_embedding)
            )
            base_score = np.mean(similarities)
            
            # Theme relevance score
            theme_score = 0
            for keyword in theme_keywords:
                if keyword.lower() in sentence_lower:
                    theme_score += 1
            
            # Normalize theme score
            theme_score = min(theme_score / max(len(theme_keywords), 1), 1.0)
            
            # Position score (opening and closing are important)
            if i in sections['opening']:
                position_score = 1.0
            elif i in sections['closing']:
                position_score = 0.8
            else:
                position_score = 0.6
            
            # Length score (prefer medium-length sentences)
            length_score = min(len(sentence) / 150, 1.0)
            
            # Combined score with theme emphasis
            final_score = (base_score * 0.4 + 
                          theme_score * 0.4 + 
                          position_score * 0.15 + 
                          length_score * 0.05)
            
            scores.append((final_score, sentence, i))
        
        return sorted(scores, key=lambda x: x[0], reverse=True)
    
    def _fast_extractive_summarize(self, sentences: List[str], target_ratio: float, start_time: float, title: str = "") -> Dict[str, any]:
        """Fast summarization for medium-length texts with theme awareness."""
        target_count = max(2, int(len(sentences) * target_ratio))
        
        # Identify theme from title and first sentences
        theme_keywords = self._identify_central_theme(sentences[:2], title)
        
        # Theme-aware scoring
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Position score (beginning and end are important)
            pos_score = 1.0 if i < 2 or i >= len(sentences) - 2 else 0.5
            
            # Length score (prefer medium-length sentences)
            len_score = min(1.0, len(sentence) / 100)
            
            # Hebrew content score
            hebrew_ratio = len([c for c in sentence if '\u0590' <= c <= '\u05FF']) / len(sentence)
            
            # Theme relevance score
            theme_score = 0
            for keyword in theme_keywords:
                if keyword.lower() in sentence_lower:
                    theme_score += 0.5
            
            total_score = pos_score + len_score + hebrew_ratio + theme_score
            scored_sentences.append((total_score, sentence, i))
        
        # Select top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        selected = scored_sentences[:target_count]
        
        # Reorder by original position
        selected.sort(key=lambda x: x[2])
        summary = ' '.join([s[1] for s in selected])
        
        return {
            "summary": summary,
            "metadata": {
                "method": "fast_extractive_with_theme",
                "original_sentences": len(sentences),
                "summary_sentences": len(selected),
                "compression_ratio": len(selected) / len(sentences),
                "central_theme": theme_keywords,
                "processing_time": time.time() - start_time
            }
        }
    
    def _create_flowing_summary(self, selected_sentences: List[Tuple[float, str, int]], 
                               all_sentences: List[str]) -> str:
        """Create a flowing, coherent summary maintaining narrative order."""
        # Sort by original position to maintain narrative flow
        ordered_sentences = sorted(selected_sentences, key=lambda x: x[2])
        
        # Extract just the sentences
        summary_sentences = [s[1] for s in ordered_sentences]
        
        # Add smooth transitions where needed
        flowing_sentences = []
        for i, sentence in enumerate(summary_sentences):
            # Clean up sentence start if it's a connector without context
            if i == 0 and any(sentence.startswith(c) for c in self.connectors):
                # Remove leading connector for first sentence
                for connector in self.connectors:
                    if sentence.startswith(connector):
                        sentence = sentence[len(connector):].strip()
                        if sentence and sentence[0].islower():
                            sentence = sentence[0].upper() + sentence[1:]
                        break
            
            flowing_sentences.append(sentence)
        
        return ' '.join(flowing_sentences)
    
    def _polish_summary_flow(self, summary: str) -> str:
        """Final polish for summary coherence and readability."""
        # Remove double spaces
        summary = re.sub(r'\s+', ' ', summary)
        
        # Ensure proper sentence endings
        summary = re.sub(r'([^.!?])\s*$', r'\1.', summary)
        
        # Fix spacing around punctuation
        summary = re.sub(r'\s+([.!?])', r'\1', summary)
        
        return summary.strip()
    
    def _create_short_summary(self, text: str, sentences: List[str], start_time: float) -> Dict[str, any]:
        """Handle short texts that don't need complex summarization."""
        return {
            "summary": text,
            "metadata": {
                "method": "no_summarization_needed",
                "original_sentences": len(sentences),
                "processing_time": time.time() - start_time,
                "note": "Text too short for comprehensive summarization"
            }
        }
    
    def extractive_summarize(self, text: str, top_n: int = 6) -> Dict[str, any]:
        """Main extractive summarization entry point - now uses comprehensive method."""
        start_time = time.time()
        
        try:
            # Preprocess and split
            clean_text = self.preprocess_hebrew_text(text)
            sentences = self.split_sentences(clean_text)
            
            logger.info(f"Extractive request: top_n={top_n}, total_sentences={len(sentences)}")
            
            if len(sentences) < 3:
                return {
                    "summary": clean_text,
                    "metadata": {
                        "method": "no_summarization_needed",
                        "original_sentences": len(sentences),
                        "processing_time": time.time() - start_time
                    }
                }
            
            # For small requests or short texts, use simple selection
            if top_n <= 3 or len(sentences) <= top_n + 2:
                return self._simple_extractive_summarize(sentences, top_n, start_time)
            
            # For larger requests, use comprehensive method
            target_ratio = min(0.8, max(0.15, top_n / len(sentences)))
            logger.info(f"Using comprehensive method: target_ratio={target_ratio:.2f}")
            return self.comprehensive_extractive_summarize(text, target_ratio)
            
        except Exception as e:
            logger.error(f"Extractive summarization failed: {e}")
            return self._fallback_summary(text, start_time)
    
    def comprehensive_abstractive_summarize(self, text: str, target_ratio: float = 0.3, title: str = "") -> Dict[str, any]:
        """
        Hebrew-optimized abstractive summarization:
        - Uses AlephBERT for understanding + intelligent rewriting
        - Uses native Hebrew models for better quality
        - Creates flowing, natural summaries
        - Deterministic results for same input
        """
        start_time = time.time()
        
        # Set seed for consistent results
        import torch
        import numpy as np
        text_hash = hash(text + str(target_ratio)) % 2**32
        torch.manual_seed(text_hash)
        np.random.seed(text_hash)
        
        try:
            clean_text = self.preprocess_hebrew_text(text)
            sentences = self.split_sentences(clean_text)
            
            if len(sentences) < 3:
                return self._create_short_summary(clean_text, sentences, start_time)
            
            # Use AlephBERT-based intelligent extraction with rewriting
            result = self._hebrew_optimized_abstractive(sentences, target_ratio, start_time)
            

            
            return result
            
        except Exception as e:
            logger.error(f"Abstractive summarization failed: {e}")
            # Fall back to comprehensive extractive instead of simple fallback
            logger.info("Falling back to comprehensive extractive summarization")
            return self.comprehensive_extractive_summarize(text, target_ratio, title)
    
    def _hebrew_optimized_abstractive(self, sentences: List[str], target_ratio: float, start_time: float) -> Dict[str, any]:
        """Hebrew-optimized abstractive using AlephBERT understanding."""
        try:
            # Get embeddings and score sentences
            embeddings = self.get_sentence_embeddings(tuple(sentences))
            scored_sentences = self.calculate_sentence_scores(sentences, embeddings)
            
            # Select key sentences for rewriting
            target_count = max(2, int(len(sentences) * target_ratio))
            selected = scored_sentences[:target_count + 2]  # Get a few extra for rewriting
            
            # Create abstractive-style summary by intelligent rewriting
            summary = self._create_abstractive_style_summary(selected, sentences)
            
            return {
                "summary": summary,
                "metadata": {
                    "method": "hebrew_optimized_abstractive",
                    "original_sentences": len(sentences),
                    "summary_sentences": len(summary.split('.')),
                    "compression_ratio": len(summary) / len(' '.join(sentences)),
                    "processing_time": time.time() - start_time,
                    "model": "AlephBERT + Hebrew Rewriting"
                }
            }
            
        except Exception as e:
            logger.error(f"Hebrew abstractive failed: {e}")
            return self._fallback_summary(' '.join(sentences), start_time)
    
    def _create_abstractive_style_summary(self, selected_sentences: List[Tuple[float, str, int]], 
                                        all_sentences: List[str]) -> str:
        """Create abstractive-style summary by intelligent rewriting and paraphrasing."""
        # Extract and paraphrase key information
        key_concepts = []
        for score, sentence, idx in selected_sentences:
            paraphrased = self._paraphrase_sentence(sentence)
            if paraphrased and paraphrased != sentence:
                key_concepts.append(paraphrased)
            else:
                # Extract key concepts if paraphrasing fails
                concepts = self._extract_key_concepts(sentence)
                if concepts:
                    key_concepts.append(concepts)
        
        if not key_concepts:
            return selected_sentences[0][1] if selected_sentences else ""
        
        # Create new flowing narrative
        if len(key_concepts) == 1:
            return self._improve_hebrew_flow(key_concepts[0])
        
        # Combine multiple concepts into new sentences
        combined_summary = self._synthesize_new_summary(key_concepts)
        
        return self._improve_hebrew_flow(combined_summary)
    
    def _paraphrase_sentence(self, sentence: str) -> str:
        """Paraphrase a sentence to create abstractive-style content."""
        # Hebrew paraphrasing patterns
        paraphrase_patterns = [
            # Change "זהו X" to "מדובר ב-X"
            (r'^זהו\s+(.+)', r'מדובר ב\1'),
            # Change "הוא מכיל" to "כולל"
            (r'הוא מכיל\s+(.+)', r'כולל \1'),
            # Change "המטרה היא" to "נועד"
            (r'המטרה היא\s+(.+)', r'נועד \1'),
            # Change "צריך לבדוק" to "נבדק"
            (r'צריך לבדוק\s+(.+)', r'נבדק \1'),
            # Change "הוא צריך" to "נדרש"
            (r'הוא צריך\s+(.+)', r'נדרש \1'),
        ]
        
        paraphrased = sentence
        for pattern, replacement in paraphrase_patterns:
            paraphrased = re.sub(pattern, replacement, paraphrased, flags=re.IGNORECASE)
        
        # If no paraphrasing occurred, try concept extraction
        if paraphrased == sentence:
            return self._extract_key_concepts(sentence)
        
        return paraphrased
    
    def _synthesize_new_summary(self, concepts: List[str]) -> str:
        """Synthesize new summary from multiple concepts."""
        if not concepts:
            return ""
        
        if len(concepts) == 1:
            return concepts[0]
        
        # Create new synthetic sentences
        synthesis_templates = [
            "המאמר עוסק ב{} ו{}",
            "הנושא כולל {} בנוסף ל{}",
            "המחקר מתמקד ב{} תוך התייחסות ל{}",
            "הדיון נסוב סביב {} ו{}"
        ]
        
        # Try to combine first two concepts
        if len(concepts) >= 2:
            concept1 = self._clean_concept_for_synthesis(concepts[0])
            concept2 = self._clean_concept_for_synthesis(concepts[1])
            
            if concept1 and concept2:
                template = synthesis_templates[0]  # Use first template
                synthesized = template.format(concept1, concept2)
                
                # Add remaining concepts if any
                if len(concepts) > 2:
                    remaining = [self._clean_concept_for_synthesis(c) for c in concepts[2:]]
                    remaining = [c for c in remaining if c]  # Filter empty
                    if remaining:
                        synthesized += ". " + "כמו כן, " + ", ".join(remaining[:2])
                
                return synthesized
        
        # Fallback: just connect with connectors
        return self._connect_ideas(concepts)
    
    def _clean_concept_for_synthesis(self, concept: str) -> str:
        """Clean concept for use in synthesis templates."""
        # Remove sentence endings and clean up
        cleaned = concept.strip('.,!?;:')
        
        # Remove common sentence starters
        starters = ['זהו', 'הוא', 'היא', 'זה', 'המטרה', 'הנושא']
        words = cleaned.split()
        
        if words and words[0] in starters:
            cleaned = ' '.join(words[1:])
        
        # Keep only if substantial
        return cleaned if len(cleaned) > 10 else ""
    
    def _extract_key_concepts(self, sentence: str) -> str:
        """Extract key concepts from a sentence."""
        # Remove common filler words and focus on key content
        words = sentence.split()
        
        # Keep important words, remove common fillers
        important_words = []
        skip_words = {'זה', 'זו', 'זאת', 'הוא', 'היא', 'הם', 'הן', 'כי', 'אם', 'או', 'גם', 'רק'}
        
        for word in words:
            clean_word = word.strip('.,!?;:')
            if len(clean_word) > 2 and clean_word not in skip_words:
                important_words.append(word)
        
        # Reconstruct with key concepts
        if len(important_words) >= 3:
            return ' '.join(important_words)
        else:
            return sentence  # Keep original if too short
    
    def _connect_ideas(self, ideas: List[str]) -> str:
        """Connect multiple ideas into flowing text."""
        if not ideas:
            return ""
        
        if len(ideas) == 1:
            return ideas[0]
        
        # Simple connection with Hebrew connectors
        connectors = ['כמו כן', 'בנוסף', 'יתר על כן', 'למעשה']
        
        connected = ideas[0]
        for i, idea in enumerate(ideas[1:], 1):
            if i < len(connectors):
                connected += f". {connectors[i-1]}, {idea}"
            else:
                connected += f". {idea}"
        
        return connected
    
    def _improve_hebrew_flow(self, text: str) -> str:
        """Improve Hebrew text flow and readability."""
        # Basic flow improvements
        text = re.sub(r'\s+', ' ', text)  # Clean spaces
        text = re.sub(r'\.+', '.', text)  # Fix multiple periods
        text = text.strip()
        
        # Ensure proper ending
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    

    

    
    def _is_reasonable_summary(self, summary: str, original: str) -> bool:
        """Check if summary is reasonable quality - strict validation."""
        if not summary or len(summary) < 50:  # Minimum meaningful length
            return False
        
        # Check for Hebrew content (must be majority Hebrew)
        hebrew_chars = len(re.findall(r'[\u0590-\u05FF]', summary))
        total_chars = len(re.sub(r'\s', '', summary))
        if total_chars == 0 or hebrew_chars < total_chars * 0.6:
            return False
        
        # Check for artifacts and gibberish
        artifacts = ['<extra_id', 'summarize:', 'summary:', '▁', 'undefined', 'null', 
                    'NaN', '###', '***', '...', '???']
        if any(artifact in summary.lower() for artifact in artifacts):
            return False
        
        # Check for excessive repetitions (stricter)
        words = summary.split()
        if len(words) > 5:
            # Count repeated 2-word phrases
            repeated_phrases = 0
            for i in range(len(words) - 3):
                phrase = f"{words[i]} {words[i+1]}"
                if phrase in ' '.join(words[i+2:]):
                    repeated_phrases += 1
            
            # If more than 20% of phrases are repeated, it's bad quality
            if repeated_phrases > len(words) * 0.2:
                return False
        
        # Check for incomplete sentences and brackets
        if (summary.count('(') != summary.count(')') or 
            summary.count('[') != summary.count(']') or
            summary.count('{') != summary.count('}')):
            return False
        
        # Check sentence structure
        sentences = [s.strip() for s in summary.split('.') if s.strip()]
        if len(sentences) == 0:
            return False
        
        # Check for very short or very long sentences
        for sentence in sentences:
            words_in_sentence = len(sentence.split())
            if words_in_sentence < 3 or words_in_sentence > 50:
                return False
        
        # Check for nonsensical character patterns
        if re.search(r'[a-zA-Z]{10,}', summary):  # Long English sequences
            return False
        
        if re.search(r'[\d\W]{5,}', summary):  # Long sequences of numbers/symbols
            return False
        
        return True
    
    def _extract_key_points_from_sections(self, sentences: List[str], 
                                         sections: Dict[str, List[int]]) -> Dict[str, List[str]]:
        """Extract key points from each section of the text."""
        key_points = {'opening': [], 'middle': [], 'closing': []}
        
        for section_name, indices in sections.items():
            section_sentences = [sentences[i] for i in indices]
            
            # Score sentences within section
            if len(section_sentences) > 0:
                # Simple scoring based on content indicators
                scored = []
                for sentence in section_sentences:
                    score = 0
                    
                    # Key content indicators
                    if self._contains_key_insights(sentence):
                        score += 2
                    if re.search(r'\d+', sentence):  # Contains numbers/data
                        score += 1
                    if len(sentence.split()) > 10:  # Substantial length
                        score += 1
                    if sentence.endswith('?'):  # Questions are often important
                        score += 1
                    
                    scored.append((score, sentence))
                
                # Select top sentences from section
                scored.sort(key=lambda x: x[0], reverse=True)
                section_limit = max(1, len(section_sentences) // 3)
                key_points[section_name] = [s[1] for s in scored[:section_limit]]
        
        return key_points
    
    def _create_structured_input(self, key_points: Dict[str, List[str]], full_text: str) -> str:
        """Create structured input that guides the model to comprehensive summarization."""
        # Create a structured prompt that emphasizes comprehensive coverage
        structured_parts = []
        
        # Add instruction for comprehensive summarization
        structured_parts.append("summarize comprehensively from beginning to end:")
        
        # Add key points from each section
        if key_points['opening']:
            structured_parts.append("Opening: " + " ".join(key_points['opening'][:2]))
        
        if key_points['middle']:
            structured_parts.append("Main content: " + " ".join(key_points['middle'][:3]))
        
        if key_points['closing']:
            structured_parts.append("Conclusion: " + " ".join(key_points['closing'][:2]))
        
        # Combine with truncated full text if space allows
        structured_input = " ".join(structured_parts)
        
        # If input is too long, prioritize the structured parts
        if len(structured_input.split()) > 400:
            # Use only the structured key points
            return structured_input
        else:
            # Add some of the full text for context
            remaining_space = 400 - len(structured_input.split())
            full_text_words = full_text.split()[:remaining_space]
            return structured_input + " Full context: " + " ".join(full_text_words)
    
    def _polish_abstractive_summary(self, raw_summary: str, key_points: Dict[str, List[str]]) -> str:
        """Polish the abstractive summary for better flow and coherence."""
        # Remove any instruction artifacts
        summary = re.sub(r'^(summarize|summary|תקציר)[:.]?\s*', '', raw_summary, flags=re.IGNORECASE)
        
        # Ensure proper capitalization
        if summary and summary[0].islower():
            summary = summary[0].upper() + summary[1:]
        
        # Clean up spacing and punctuation
        summary = re.sub(r'\s+', ' ', summary)
        summary = re.sub(r'\s+([.!?])', r'\1', summary)
        
        # Ensure proper ending
        if summary and not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        return summary.strip()
    
    def abstractive_summarize(self, text: str, max_length: int = 150) -> Dict[str, any]:
        """Main abstractive summarization entry point - now uses comprehensive method."""
        # Convert max_length to target_ratio for consistency
        # For abstractive, we use character-based estimation
        estimated_chars = len(text)
        if estimated_chars > 0:
            # Estimate target ratio based on desired output length
            target_ratio = min(0.5, max(0.2, (max_length * 6) / estimated_chars))  # ~6 chars per word
        else:
            target_ratio = 0.3
        
        return self.comprehensive_abstractive_summarize(text, target_ratio)
    
    def _simple_extractive_summarize(self, sentences: List[str], top_n: int, start_time: float) -> Dict[str, any]:
        """Simple extractive summarization that directly respects top_n."""
        try:
            # Get embeddings
            embeddings = self.get_sentence_embeddings(tuple(sentences))
            
            # Score sentences using the existing method
            scored_sentences = self.calculate_sentence_scores(sentences, embeddings)
            
            # Select exactly top_n sentences
            selected = scored_sentences[:top_n]
            
            # Order by original position
            selected.sort(key=lambda x: x[2])
            
            # Create summary
            summary_sentences = [s[1] for s in selected]
            summary = ' '.join(summary_sentences)
            
            return {
                "summary": summary,
                "metadata": {
                    "method": "simple_extractive",
                    "original_sentences": len(sentences),
                    "summary_sentences": len(selected),
                    "compression_ratio": len(selected) / len(sentences),
                    "processing_time": time.time() - start_time,
                    "model": "AlephBERT + Simple Selection"
                }
            }
            
        except Exception as e:
            logger.error(f"Simple extractive failed: {e}")
            return self._fallback_summary(' '.join(sentences), start_time)
    
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
        title = data.get("title", "").strip()
        if not text:
            return jsonify({"error": "Empty text provided"}), 400
        
        # Parameters - support both old (top_n) and new (target_ratio) formats
        method = data.get("method", "extractive")  # "extractive" or "abstractive"
        
        # Handle percentage-based input (new format)
        if "target_ratio" in data:
            target_ratio = float(data.get("target_ratio", 0.35))
            max_length = data.get("max_length", 150)
        else:
            # Backward compatibility with sentence-based input (old format)
            top_n = data.get("top_n", 6)
            max_length = data.get("max_length", 150)
            # Convert top_n to target_ratio for internal use
            sentences = ai_summarizer.split_sentences(ai_summarizer.preprocess_hebrew_text(text))
            target_ratio = min(0.8, max(0.15, top_n / len(sentences))) if len(sentences) > 0 else 0.35
        
        # Log parameters for debugging
        logger.info(f"Summarization request: method={method}, target_ratio={target_ratio:.2f}, max_length={max_length}, text_length={len(text)}")
        
        # Choose summarization method
        if method == "abstractive":
            result = ai_summarizer.comprehensive_abstractive_summarize(text, target_ratio, title)
        else:
            result = ai_summarizer.comprehensive_extractive_summarize(text, target_ratio, title)
        
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
    print("🧠 Models: AlephBERT Hebrew NLP")
    print("⚡ Methods: Extractive & Hybrid Abstractive")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 50)
    
    app.run(host='0.0.0.0', port=5002, debug=True)