"""
Improved Input Validation System
- Less aggressive filtering
- Better handling of technical terms
- Contextual validation
- Multi-language support preparation
"""

import re
from typing import Tuple, List, Dict
import unicodedata


class ImprovedInputValidator:
    """Enhanced input validator that's less aggressive and more contextual"""
    
    def __init__(self):
        # Expanded common words including technical terms
        self.common_words = {
            # Question words
            'what', 'who', 'where', 'when', 'why', 'how', 'which',
            # Action words
            'tell', 'explain', 'describe', 'show', 'list', 'give',
            'can', 'could', 'would', 'should', 'will', 'do', 'does',
            # Common words
            'hello', 'hi', 'hey', 'greetings', 'help', 'about', 'your',
            'you', 'me', 'my', 'have', 'has', 'this', 'that', 'with',
            'for', 'from', 'the', 'a', 'an', 'these', 'those', 'are', 'is',
            # Technical terms (AI/ML)
            'ai', 'ml', 'nlp', 'cv', 'llm', 'bert', 'gpt', 'model',
            'neural', 'network', 'deep', 'learning', 'machine', 'python',
            'tensorflow', 'pytorch', 'api', 'cloud', 'aws', 'gcp', 'azure',
            'docker', 'kubernetes', 'data', 'algorithm', 'train', 'deploy',
            # Portfolio terms
            'project', 'skill', 'experience', 'work', 'portfolio',
            'education', 'degree', 'university', 'certification'
        }
        
        # Common bigrams in English
        self.common_bigrams = {
            'th', 'he', 'in', 'er', 'an', 're', 'on', 'at', 'en', 'nd',
            'ed', 'es', 'or', 'ti', 'ar', 'te', 'ng', 'al', 'it', 'as',
            'is', 'ou', 'io', 'le', 'ea', 'ra', 'co', 'ro', 'll', 'ha'
        }
        
        # Technical abbreviations that might look like gibberish
        self.technical_abbrevs = {
            'ai', 'ml', 'nlp', 'cv', 'cnn', 'rnn', 'lstm', 'gru',
            'api', 'aws', 'gcp', 'ci', 'cd', 'etl', 'sql', 'nosql',
            'iot', 'gpu', 'cpu', 'ram', 'ssd', 'http', 'https', 'rest',
            'json', 'xml', 'html', 'css', 'js', 'ts', 'py', 'cpp',
            'gpt', 'bert', 'yolo', 'gan', 'vae', 'rl', 'dqn'
        }
    
    def is_valid_message(self, message: str) -> Tuple[bool, str, Dict]:
        """
        Enhanced validation with detailed feedback
        
        Returns:
            (is_valid, reason, metadata)
        """
        metadata = {
            "length": len(message),
            "word_count": len(message.split()),
            "has_technical_terms": False,
            "language_detected": "unknown"
        }
        
        # Basic checks
        if len(message.strip()) < 2:
            return False, "too_short", metadata
        
        # Remove extra whitespace
        message = ' '.join(message.split())
        metadata["length"] = len(message)
        
        # Check for all non-alphabetic (e.g., just numbers or symbols)
        clean_text = ''.join(char for char in message if char.isalpha())
        if not clean_text:
            return False, "no_letters", metadata
        
        # Extract words
        words = message.split()
        metadata["word_count"] = len(words)
        
        # Single character messages (unless common ones like "hi")
        if len(words) == 1 and len(words[0]) == 1:
            if words[0].lower() not in ['i', 'a']:
                return False, "single_char", metadata
        
        # Check for technical terms
        message_lower = message.lower()
        has_tech_term = any(term in message_lower for term in self.technical_abbrevs)
        metadata["has_technical_terms"] = has_tech_term
        
        # Check for common words
        words_lower = [w.lower().strip('.,!?;:\"\'') for w in words]
        has_common_word = any(word in self.common_words for word in words_lower)
        
        # If has technical terms, be more lenient
        if has_tech_term:
            return True, "valid_technical", metadata
        
        # If has common words, validate
        if has_common_word:
            return True, "valid_common", metadata
        
        # For messages without obvious markers, do deeper analysis
        gibberish_score = self._calculate_gibberish_score(words)
        metadata["gibberish_score"] = gibberish_score
        
        # More lenient threshold
        if gibberish_score > 0.85:  # Increased from 0.7
            return False, "likely_gibberish", metadata
        
        # Check for reasonable character distribution
        if len(clean_text) > 5:
            if not self._has_reasonable_char_distribution(clean_text):
                return False, "unusual_distribution", metadata
        
        # Check for at least one valid-looking word
        if not self._has_valid_word_pattern(words_lower):
            return False, "no_valid_patterns", metadata
        
        # Passed all checks
        return True, "valid", metadata
    
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
        
        # Natural language typically has 35-45% vowels
        # Allow 15-65% range to be more inclusive
        if vowel_ratio < 0.15 or vowel_ratio > 0.65:
            return False
        
        # Check for keyboard mashing patterns
        # Allow up to 6 consecutive consonants (some words like "strength")
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
        """Check if at least one word looks valid"""
        for word in words:
            if len(word) < 2:
                continue
            
            # Check if word is in common words or technical terms
            if word in self.common_words or word in self.technical_abbrevs:
                return True
            
            # Check if word has common bigrams
            if len(word) >= 3:
                for i in range(len(word) - 1):
                    if word[i:i+2] in self.common_bigrams:
                        return True
        
        return False
    
    def sanitize_input(self, message: str) -> str:
        """Sanitize user input"""
        # Remove potential script tags
        message = re.sub(r'<script[^>]*>.*?</script>', '', message, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove HTML tags
        message = re.sub(r'<[^>]+>', '', message)
        
        # Remove null bytes
        message = message.replace('\x00', '')
        
        # Normalize unicode
        message = unicodedata.normalize('NFKC', message)
        
        # Limit length
        max_length = 1000  # Increased from 500
        if len(message) > max_length:
            message = message[:max_length]
        
        # Remove excessive whitespace but preserve single spaces
        message = ' '.join(message.split())
        
        return message.strip()
    
    def get_validation_feedback(self, is_valid: bool, reason: str, metadata: Dict) -> str:
        """Get user-friendly feedback message"""
        if is_valid:
            return ""
        
        feedback_messages = {
            "too_short": "Could you provide a bit more detail in your question?",
            "no_letters": "I need a text-based question. How can I help you learn about my AI/ML expertise?",
            "single_char": "Could you ask a complete question? Feel free to ask about my skills, projects, or experience.",
            "likely_gibberish": "I didn't quite understand that. Could you rephrase your question about my AI/ML work?",
            "unusual_distribution": "That doesn't look like a valid question. Try asking about my experience, projects, or technical skills!",
            "no_valid_patterns": "I couldn't process that input. Could you ask me about my skills, projects, or experience?"
        }
        
        default_msg = "I didn't understand that. Could you ask me about my skills, projects, or AI/ML experience?"
        
        return feedback_messages.get(reason, default_msg)


# Example usage
if __name__ == "__main__":
    validator = ImprovedInputValidator()
    
    # Test cases
    test_messages = [
        "What are your AI skills?",  # Valid
        "Tell me about NLP",  # Valid - technical term
        "asdfghjkl",  # Invalid - gibberish
        "hi",  # Valid - common word
        "Can you explain your CNN experience?",  # Valid - technical
        "xyzabc",  # Invalid - no patterns
        "What's your experience with MLOps?",  # Valid
        "qqqqqqq",  # Invalid - repeated chars
        "How do you deploy models to AWS?",  # Valid
        ""  # Invalid - empty
    ]
    
    print("Input Validation Tests:\n" + "="*50)
    for msg in test_messages:
        is_valid, reason, metadata = validator.is_valid_message(msg)
        status = "✓ VALID" if is_valid else "✗ INVALID"
        print(f"\n{status}: '{msg}'")
        print(f"  Reason: {reason}")
        print(f"  Metadata: {metadata}")
        if not is_valid:
            feedback = validator.get_validation_feedback(is_valid, reason, metadata)
            print(f"  Feedback: {feedback}")