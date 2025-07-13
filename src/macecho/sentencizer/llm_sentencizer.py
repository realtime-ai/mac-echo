from .base import BaseSentencizer
from typing import List, Optional, Set


class LLMSentencizer(BaseSentencizer):
    """
    LLM streaming output sentence splitter
    
    Designed specifically for LLM streaming output, treats newlines as sentence endings.
    Suitable for scenarios where LLM outputs newlines at the end of each sentence.
    """
    
    def __init__(self,
                 end_punctuations: Optional[Set[str]] = None,
                 pause_punctuations: Optional[Set[str]] = None,
                 newline_as_separator: bool = True,
                 strip_newlines: bool = True):
        """
        Initialize LLM sentence splitter
        
        Args:
            end_punctuations: Set of ending punctuation marks
            pause_punctuations: Set of pause punctuation marks  
            newline_as_separator: Whether to use newlines as sentence separators
            strip_newlines: Whether to remove newlines when returning sentences
        """
        # Pass separator to base class if using newlines
        separator = '\n' if newline_as_separator else None
        super().__init__(end_punctuations, pause_punctuations, separator)
        
        self.newline_as_separator = newline_as_separator
        self.strip_newlines = strip_newlines
    
    def _is_sentence_complete(self, text: str) -> bool:
        """
        Check if text forms a complete sentence
        
        For LLM output, if newline separation is enabled, newlines indicate sentence end;
        otherwise use traditional punctuation marks.
        """
        if not text:
            return False
        
        # If newline separation enabled, check for newlines
        if self.newline_as_separator and '\n' in text:
            return True
            
        # Check if ends with ending punctuation
        return text.rstrip()[-1] in self.end_punctuations if text.rstrip() else False
    
    def _extract_sentences(self, text: str) -> List[str]:
        """
        Extract complete sentences from text
        
        If newline separation enabled, split by newlines;
        otherwise split by punctuation marks.
        """
        sentences = []
        
        if self.newline_as_separator:
            # Split sentences by newlines
            lines = text.split('\n')
            
            # Process all lines except the last one (they are complete sentences)
            for line in lines[:-1]:
                sentence = line.strip() if self.strip_newlines else line
                if sentence:  # Only add non-empty sentences
                    sentences.append(sentence)
            
            # Update buffer to last line (possibly incomplete)
            last_line = lines[-1] if lines else ""
            remaining_text = last_line
        else:
            # Use traditional punctuation splitting
            start_pos = 0
            remaining_text = text
            
            for i, char in enumerate(text):
                if char in self.end_punctuations:
                    sentence = text[start_pos:i+1].strip()
                    if sentence:
                        sentences.append(sentence)
                    start_pos = i + 1
            
            # Keep unprocessed part
            remaining_text = text[start_pos:]
        
        # Update buffer
        self.buffer = remaining_text
        
        return sentences
    
    def process_chunk(self, text_chunk: str) -> List[str]:
        """
        Process a text chunk and return all complete sentences
        
        Override base class method to optimize LLM streaming processing
        """
        if not text_chunk:
            return []
        
        # Add new text to buffer
        self.buffer += text_chunk
        
        # Extract complete sentences (this automatically updates buffer)
        sentences = self._extract_sentences(self.buffer)
        
        return sentences


# Test code
if __name__ == "__main__":
    print("=== LLM Sentencizer Test ===")
    
    # Create LLMSentencizer instance
    sentencizer = LLMSentencizer()
    
    # Simulate LLM streaming output (each sentence followed by newline)
    llm_stream_text = "This is the first sentence.\nThis is the second sentence!\nThis is the third sentence?\n"
    
    print(f"Simulated LLM output: {repr(llm_stream_text)}")
    print("Streaming processing result:")
    
    # Simulate chunked input
    chunk_size = 5
    sentences_count = 0
    
    for i in range(0, len(llm_stream_text), chunk_size):
        chunk = llm_stream_text[i:i+chunk_size]
        print(f"Input chunk: {repr(chunk)}")
        
        sentences = sentencizer.process_chunk(chunk)
        for sentence in sentences:
            sentences_count += 1
            print(f"  Sentence {sentences_count}: '{sentence}'")
    
    # Process remaining content
    remaining = sentencizer.finish()
    for sentence in remaining:
        sentences_count += 1
        print(f"  Remaining sentence: '{sentence}'")
    
    print(f"\nTotal extracted {sentences_count} sentences")
    
    # Test mixed mode (newlines + punctuation)
    print("\n=== Mixed Mode Test ===")
    sentencizer.reset()
    
    mixed_text = "Sentence 1. Sentence 2!\nSentence 3 after newline? Sentence 4.\n"
    print(f"Mixed text: {repr(mixed_text)}")
    
    # Process all at once
    sentences = sentencizer.process_chunk(mixed_text)
    remaining = sentencizer.finish()
    
    all_sentences = sentences + remaining
    print("Extracted sentences:")
    for i, sentence in enumerate(all_sentences, 1):
        print(f"  {i}: '{sentence}'")
    
    # Test keeping newlines
    print("\n=== Keep Newlines Test ===")
    sentencizer_keep_newlines = LLMSentencizer(strip_newlines=False)
    
    test_text = "Keep newline sentence 1.\nKeep newline sentence 2!\n"
    sentences = sentencizer_keep_newlines.process_chunk(test_text)
    remaining = sentencizer_keep_newlines.finish()
    
    all_sentences = sentences + remaining
    print("Results with newlines kept:")
    for i, sentence in enumerate(all_sentences, 1):
        print(f"  {i}: {repr(sentence)}")
    
    # Test traditional punctuation mode
    print("\n=== Traditional Punctuation Mode Test ===")
    traditional_sentencizer = LLMSentencizer(newline_as_separator=False)
    
    traditional_text = "Traditional sentence 1. Traditional sentence 2! Traditional sentence 3?"
    sentences = traditional_sentencizer.process_chunk(traditional_text)
    remaining = traditional_sentencizer.finish()
    
    all_sentences = sentences + remaining
    print("Traditional mode result:")
    for i, sentence in enumerate(all_sentences, 1):
        print(f"  {i}: '{sentence}'")