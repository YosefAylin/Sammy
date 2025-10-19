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
    
    def comprehensive_extractive_summarize(self, text: str, target_ratio: float = 0.35) -> Dict[str, any]:
        """
        Comprehensive extractive summarization following the guidelines:
        - Captures main ideas from opening, middle, and closing sections
        - Extracts essential content avoiding repetitions
        - Creates flowing, coherent summary representing entire text
        - Maintains 25-40% of original length
        """
        start_time = time.time()
        
        try:
            # 1. Preprocess and analyze text structure
            clean_text = self.preprocess_hebrew_text(text)
            sentences = self.split_sentences(clean_text)
            
            if len(sentences) < 5:
                return self._create_short_summary(clean_text, sentences, start_time)
            
            # 2. Identify sections (opening, middle, closing)
            sections = self._identify_text_sections(sentences)
            
            # 3. Get embeddings for all sentences
            embeddings = self.get_sentence_embeddings(tuple(sentences))
            
            # 4. Score sentences with section-aware algorithm
            scored_sentences = self._comprehensive_sentence_scoring(
                sentences, embeddings, sections
            )
            
            # 5. Select sentences ensuring representation from all sections
            selected_sentences = self._select_representative_sentences(
                scored_sentences, sections, target_ratio
            )
            
            # 6. Reorder and create flowing narrative
            final_summary = self._create_flowing_summary(selected_sentences, sentences)
            
            # 7. Post-process for coherence
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
        
        return selected
    
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
        # Convert top_n to target_ratio based on estimated sentence count
        sentences = self.split_sentences(self.preprocess_hebrew_text(text))
        if len(sentences) > 0:
            # Be more responsive to user's top_n preference
            target_ratio = min(0.8, max(0.15, top_n / len(sentences)))
            # Ensure we don't go below user's request if text is short
            if len(sentences) <= top_n:
                target_ratio = 1.0
        else:
            target_ratio = 0.35
        
        logger.info(f"Extractive: sentences={len(sentences)}, top_n={top_n}, target_ratio={target_ratio:.2f}")
        return self.comprehensive_extractive_summarize(text, target_ratio)
    
    def comprehensive_abstractive_summarize(self, text: str, target_ratio: float = 0.3) -> Dict[str, any]:
        """
        Comprehensive abstractive summarization following guidelines:
        - Analyzes entire text from beginning to end
        - Identifies key points from all sections
        - Generates flowing, coherent summary in own words
        - Maintains 25-40% compression ratio
        """
        start_time = time.time()
        
        try:
            clean_text = self.preprocess_hebrew_text(text)
            
            # 1. Analyze text structure and extract key points
            sentences = self.split_sentences(clean_text)
            if len(sentences) < 3:
                return self._create_short_summary(clean_text, sentences, start_time)
            
            # 2. Identify sections and key content
            sections = self._identify_text_sections(sentences)
            key_points = self._extract_key_points_from_sections(sentences, sections)
            
            # 3. Create structured input for better abstractive generation
            structured_input = self._create_structured_input(key_points, clean_text)
            
            # 4. Calculate target length based on compression ratio
            target_length = max(50, min(200, int(len(clean_text.split()) * target_ratio)))
            
            # 5. Generate summary with enhanced parameters
            inputs = self.abstractive_tokenizer(
                structured_input,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                summary_ids = self.abstractive_model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=target_length,
                    min_length=max(30, target_length // 3),
                    length_penalty=1.5,  # Encourage appropriate length
                    num_beams=6,  # More beams for better quality
                    early_stopping=True,
                    do_sample=True,  # Add some creativity
                    temperature=0.7,  # Controlled randomness
                    top_p=0.9,  # Nucleus sampling
                    repetition_penalty=1.2  # Avoid repetition
                )
            
            raw_summary = self.abstractive_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # 6. Post-process for coherence and flow
            polished_summary = self._polish_abstractive_summary(raw_summary, key_points)
            
            return {
                "summary": polished_summary,
                "metadata": {
                    "method": "comprehensive_abstractive",
                    "original_length": len(clean_text),
                    "summary_length": len(polished_summary),
                    "compression_ratio": len(polished_summary) / len(clean_text),
                    "sections_analyzed": {
                        "opening_points": len(key_points.get('opening', [])),
                        "middle_points": len(key_points.get('middle', [])),
                        "closing_points": len(key_points.get('closing', []))
                    },
                    "processing_time": time.time() - start_time,
                    "model": "mT5-small + Comprehensive Analysis"
                }
            }
            
        except Exception as e:
            logger.error(f"Comprehensive abstractive summarization failed: {e}")
            return self._fallback_summary(text, start_time)
    
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
        
        # Log parameters for debugging
        logger.info(f"Summarization request: method={method}, top_n={top_n}, max_length={max_length}, text_length={len(text)}")
        
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