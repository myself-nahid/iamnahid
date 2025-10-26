"""
Improved Input Validation System - Hardened against Gibberish
- Less aggressive filtering on valid input
- Stricter rejection of nonsensical input
- Contextual validation
- Multi-language support preparation
"""

import re
from typing import Tuple, List, Dict
import unicodedata


class ImprovedInputValidator:
    """Enhanced input validator that's less aggressive and more contextual"""

    def __init__(self):
        # ... (the __init__ method with its word lists remains the same) ...
        # Expanded common words including technical terms
        self.common_words = {
            # Question words
            'what', 'who', 'where', 'when', 'why', 'how', 'which', 'can', 'could', 'should',
            # Action words
            'tell', 'explain', 'describe', 'show', 'list', 'give', 'ask', 'query', 'present',
            'can', 'could', 'would', 'should', 'will', 'do', 'does', 'did', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
            # Common words
            'hello', 'hi', 'hey', 'greetings', 'help', 'about', 'your', 'my',
            'you', 'me', 'us', 'we', 'they', 'it', 'its', 'this', 'that', 'these', 'those',
            'with', 'for', 'from', 'the', 'a', 'an', 'to', 'of', 'in', 'on', 'at', 'by', 'up', 'down', 'out', 'off',
            'and', 'or', 'but', 'so', 'if', 'then', 'else', 'as', 'than', 'more', 'less', 'most', 'least', 'many', 'much',
            # Technical terms (AI/ML) - expanded list
            'ai', 'ml', 'nlp', 'cv', 'llm', 'bert', 'gpt', 'model', 'neural', 'network',
            'deep', 'learning', 'machine', 'python', 'tensorflow', 'pytorch', 'api', 'cloud',
            'aws', 'gcp', 'azure', 'docker', 'kubernetes', 'data', 'algorithm', 'train', 'deploy',
            'finetuning', 'transformer', 'cnn', 'rnn', 'lstm', 'gru', 'reinforcement', 'computer',
            'vision', 'natural', 'language', 'processing', 'ops', 'mlops', 'etl', 'sql', 'nosql',
            'database', 'architecture', 'system', 'solution', 'development', 'engineer', 'specialist',
            'framework', 'library', 'feature', 'functionality', 'problem', 'challenge', 'impact', 'result',
            # Portfolio terms - expanded list
            'project', 'skill', 'expertise', 'experience', 'work', 'portfolio', 'education', 'degree',
            'university', 'certification', 'achievements', 'domain', 'industry', 'career', 'role',
            'accomplishment', 'development', 'built', 'created', 'developed', 'implemented', 'utilized',
            'demonstrate', 'showcase'
        }
        
        # Common bigrams in English
        self.common_bigrams = {
            'th', 'he', 'in', 'er', 'an', 're', 'on', 'at', 'en', 'nd', 'ed', 'es', 'or', 'ti', 
            'ar', 'te', 'ng', 'al', 'it', 'as', 'is', 'ou', 'io', 'le', 'ea', 'ra', 'co', 'ro', 
            'll', 'ha', 'sh', 'ch', 'st', 'de', 'nt', 'to', 'of', 'and', 'for', 'wit', 'fro', 'abo'
        }
        
        # Technical abbreviations that might look like gibberish
        self.technical_abbrevs = {
            'ai', 'ml', 'nlp', 'cv', 'cnn', 'rnn', 'lstm', 'gru', 'gan', 'vae', 'rl', 'dqn',
            'api', 'aws', 'gcp', 'ci', 'cd', 'etl', 'sql', 'nosql', 'db', 'orm',
            'iot', 'gpu', 'cpu', 'ram', 'ssd', 'http', 'https', 'rest', 'rpc',
            'json', 'xml', 'html', 'css', 'js', 'ts', 'py', 'cpp', 'java', 'go', 'rust',
            'gpt', 'bert', 'yolo', 'res', 'net', 'seq', 'to', 'seq', 'transformer',
            'sagemaker', 'azureml', 'gcpai', 'kubernetes', 'k8s', 'docker', 'mlops',
            'scikit', 'torch', 'tf', 'keras', 'conda', 'pip', 'venv'
        }

    def is_valid_message(self, message: str) -> Tuple[bool, str, Dict]:
        """
        Hardened validation logic to better detect gibberish.
        """
        metadata = {
            "length": len(message),
            "word_count": 0,
            "known_word_ratio": 0.0,
            "gibberish_score": 0.0
        }

        # --- Step 1: Basic Sanitization and Pre-checks ---
        message = ' '.join(message.strip().split())
        if len(message) < 2:
            return False, "too_short", metadata

        clean_text = ''.join(char for char in message if char.isalpha())
        if not clean_text:
            return False, "no_letters", metadata

        words = message.split()
        words_lower = [w.lower().strip('.,!?;:\"\'') for w in words]
        metadata["word_count"] = len(words)

        # --- Step 2: High-Confidence Validation (Known Words) ---
        # Check for a reasonable ratio of known words. This is a strong signal of validity.
        known_words = [w for w in words_lower if w in self.common_words or w in self.technical_abbrevs]
        known_word_ratio = len(known_words) / len(words) if words else 0
        metadata["known_word_ratio"] = known_word_ratio

        if known_word_ratio > 0.3:  # If over 30% of words are known, it's very likely valid.
            return True, "valid_known_words", metadata

        # --- Step 3: Deep Gibberish Analysis (for unknown phrases) ---
        # If the query consists of unknown words, it needs to pass stricter heuristic checks.
        gibberish_score = self._calculate_gibberish_score(words_lower)
        metadata["gibberish_score"] = gibberish_score

        # If the score is high, it's definitely gibberish.
        if gibberish_score > 0.7:
            return False, "likely_gibberish", metadata

        # Check character distribution. A strange distribution is a red flag.
        if not self._has_reasonable_char_distribution(clean_text):
            return False, "unusual_distribution", metadata

        # If the gibberish score is moderate, check for word patterns as a tie-breaker.
        if gibberish_score > 0.4:
            # This is now a fallback, not a primary validation method.
            if not self._has_valid_word_pattern(words_lower):
                return False, "no_valid_patterns", metadata

        # --- Step 4: Final Decision ---
        # If it passes all the above checks, we can consider it valid.
        return True, "valid", metadata
    
    # ... (the rest of the helper methods like _calculate_gibberish_score, sanitize_input, etc., remain the same) ...
    def _calculate_gibberish_score(self, words: List[str]) -> float:
        """Calculate probability that text is gibberish"""
        if not words:
            return 1.0

        total_score = 0.0
        scored_words = 0

        for word in words:
            word_lower = word.lower().strip('.,!?;:\"\'')

            if len(word_lower) < 2:
                continue

            scored_words += 1
            word_score = 0.0

            # Check 1: Repeated characters (e.g., "aaaa")
            if len(word_lower) > 2 and len(set(word_lower)) <= 2:
                word_score += 0.4

            # Check 2: No vowels in word > 3 chars (excluding known abbreviations)
            if len(word_lower) > 3 and word_lower not in self.technical_abbrevs:
                vowels = set('aeiou')
                if not any(c in vowels for c in word_lower):
                    word_score += 0.3

            # Check 3: Lacks common bigrams
            if len(word_lower) > 4:
                has_common_bigram = any(
                    word_lower[i:i+2] in self.common_bigrams 
                    for i in range(len(word_lower) - 1)
                )
                if not has_common_bigram:
                    word_score += 0.2

            # Check 4: Excessive character repetition
            if len(word_lower) > 3:
                char_freq = {}
                for char in word_lower:
                    char_freq[char] = char_freq.get(char, 0) + 1
                
                max_freq = max(char_freq.values())
                if max_freq / len(word_lower) > 0.5:
                    word_score += 0.3

            total_score += min(word_score, 1.0)

        if scored_words == 0:
            return 0.0

        return total_score / scored_words

    def _has_reasonable_char_distribution(self, text: str) -> bool:
        """Check if character distribution looks natural"""
        text_lower = text.lower()
        
        # Check vowel ratio (more lenient range)
        vowels = set('aeiou')
        vowel_count = sum(1 for c in text_lower if c in vowels)
        total_alpha = len(text_lower)
        
        if total_alpha == 0:
            return False
        
        vowel_ratio = vowel_count / total_alpha
        
        # Natural language typically has 35-45% vowels. Allow 15-65% range to be more inclusive.
        if vowel_ratio < 0.15 or vowel_ratio > 0.65:
            return False
        
        # Check for keyboard mashing patterns (consecutive consonants)
        # Allow up to 6 consecutive consonants (e.g., "strengths")
        max_consecutive_consonants = 0
        current_consecutive = 0
        
        for char in text_lower:
            if char.isalpha() and char not in vowels:
                current_consecutive += 1
                max_consecutive_consonants = max(max_consecutive_consonants, current_consecutive)
            else:
                current_consecutive = 0
        
        if max_consecutive_consonants > 7:
            return False
        
        return True

    def _has_valid_word_pattern(self, words: List[str]) -> bool:
        """Check if at least one word looks valid. Stricter than before."""
        valid_pattern_count = 0
        for word in words:
            if len(word) < 3:
                continue
            
            # Check if word has at least two common bigrams
            bigram_matches = 0
            for i in range(len(word) - 1):
                if word[i:i+2] in self.common_bigrams:
                    bigram_matches += 1
            if bigram_matches >= 2:
                valid_pattern_count += 1
        
        # Require at least one word with a decent pattern
        return valid_pattern_count > 0

    def sanitize_input(self, message: str) -> str:
        """Sanitize user input to prevent basic injection and formatting issues."""
        # Remove potential script tags
        message = re.sub(r'<script[^>]*>.*?</script>', '', message, flags=re.IGNORECASE | re.DOTALL)

        # Remove other HTML tags
        message = re.sub(r'<[^>]+>', '', message)

        # Remove null bytes, which can cause issues
        message = message.replace('\x00', '')

        # Normalize unicode to handle different character representations consistently
        message = unicodedata.normalize('NFKC', message)

        # Limit overall length to prevent overload
        max_length = 1000  # Sync with models.py
        if len(message) > max_length:
            message = message[:max_length]

        # Standardize whitespace to single spaces
        message = ' '.join(message.split())

        return message.strip()

    def get_validation_feedback(self, is_valid: bool, reason: str, metadata: Dict) -> str:
        """Get user-friendly feedback message for invalid input."""
        if is_valid:
            return ""

        feedback_messages = {
            "too_short": "Your message is a bit short. Could you please provide more detail in your question?",
            "no_letters": "I can only process text-based questions. How can I help you learn about my AI/ML expertise?",
            "single_char": "Could you please ask a complete question? I can answer questions about my skills, projects, or experience.",
            "likely_gibberish": "I didn't quite understand that. Could you please rephrase your question about my AI/ML work?",
            "unusual_distribution": "That doesn't look like a valid question. Try asking about my experience, projects, or technical skills!",
            "no_valid_patterns": "I'm sorry, I couldn't process that input. Please ask me about my skills, projects, or experience."
        }

        default_msg = "I'm sorry, I didn't understand that. You can ask me about my skills, projects, or AI/ML experience."

        return feedback_messages.get(reason, default_msg)