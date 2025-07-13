#!/usr/bin/env python3
"""
Example demonstrating LLMSentencizer usage for streaming LLM output
"""

import sys
import os

# fmt: off
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from macecho.sentencizer import LLMSentencizer


def simulate_llm_streaming():
    """Simulate LLM streaming output with newlines after sentences"""
    llm_response = """Hello there! How can I help you today?
I'm an AI assistant designed to be helpful, harmless, and honest.
I can assist with a wide variety of tasks including:
- Answering questions
- Writing and editing text  
- Code analysis and programming help
- Research and information lookup
What would you like to work on?
"""

    # Simulate streaming by yielding small chunks
    chunk_size = 12  # Small chunks to simulate real streaming
    for i in range(0, len(llm_response), chunk_size):
        yield llm_response[i:i+chunk_size]


def main():
    print("=== LLM Sentencizer Demo ===\n")

    # Create sentencizer for LLM output (newline-separated sentences)
    sentencizer = LLMSentencizer(
        newline_as_separator=True,
        strip_newlines=True
    )

    print("Processing streaming LLM output...\n")

    sentence_count = 0

    # Process streaming chunks
    for chunk in simulate_llm_streaming():
        print(f"Received chunk: {repr(chunk)}")

        # Extract complete sentences from this chunk
        sentences = sentencizer.process_chunk(chunk)

        # Output any complete sentences found
        for sentence in sentences:
            sentence_count += 1
            print(f"📝 Sentence {sentence_count}: {sentence}")
            print()  # Empty line for readability

    # Handle any remaining content
    remaining_sentences = sentencizer.finish()
    for sentence in remaining_sentences:
        sentence_count += 1
        print(f"📝 Final sentence {sentence_count}: {sentence}")

    print(f"\n✅ Processed {sentence_count} sentences total")


if __name__ == "__main__":
    main()