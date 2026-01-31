from typing import List


def chunk_text(text: str, min_words: int = 500, max_words: int = 800, overlap_words: int = 100) -> List[str]:
    """
    Text chunker that creates chunks of 500-800 words.
    
    Design goals:
    - Create chunks in the 500-800 word range as specified
    - Maintain overlap to preserve context across boundaries
    - Work with whitespace-separated words
    
    Args:
        text: The text to chunk
        min_words: Minimum words per chunk (default 500)
        max_words: Maximum words per chunk (default 800)
        overlap_words: Number of words to overlap between chunks (default 100)
    
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    # Split text into words
    words = text.split()
    
    # If text is smaller than max_words, return as single chunk
    if len(words) <= max_words:
        return [" ".join(words)]

    chunks: List[str] = []
    start = 0
    
    while start < len(words):
        # Try to create a chunk of max_words, but at least min_words
        end = min(start + max_words, len(words))
        
        # If we're near the end and the remaining chunk would be too small,
        # merge it with the previous chunk if possible
        if end == len(words) and (end - start) < min_words and chunks:
            # Merge with last chunk
            last_chunk_words = chunks[-1].split()
            chunks[-1] = " ".join(last_chunk_words + words[start:end])
            break
        
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        
        if end == len(words):
            break
        
        # Step forward with overlap to preserve context
        start = end - overlap_words
        # Ensure we don't go backwards
        if start <= chunks[-1].count(" "):
            start = end

    return chunks


