"""
text_cleaner.py
---------------
Utility functions for cleaning raw text extracted from PDF documents.

Purpose : Remove noise introduced by PDF extraction (headers, footers,
          excessive whitespace, page numbers) to produce clean text
          suitable for LLM processing.
Input   : Raw string from PyMuPDF
Output  : Cleaned, normalized string
"""

import re


def remove_excessive_whitespace(text: str) -> str:
    """
    Replace multiple consecutive blank lines with a single blank line
    and strip leading/trailing whitespace from each line.

    Args:
        text: Raw extracted text.

    Returns:
        Text with normalized whitespace.
    """
    # Strip each line
    lines = [line.strip() for line in text.splitlines()]
    # Collapse 3+ consecutive empty lines into 2
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return cleaned.strip()


def remove_page_numbers(text: str) -> str:
    """
    Remove standalone page numbers (e.g., '- 1 -', 'Page 1 of 10', '1').

    Args:
        text: Input text.

    Returns:
        Text with page number patterns removed.
    """
    # Patterns like: "Page 1 of 10", "- 1 -", or a lone digit on its own line
    text = re.sub(r"(?im)^-?\s*page\s+\d+\s*(of\s+\d+)?\s*-?$", "", text)
    text = re.sub(r"(?m)^-\s*\d+\s*-$", "", text)
    text = re.sub(r"(?m)^\d+$", "", text)
    return text


def remove_special_characters(text: str) -> str:
    """
    Remove non-printable / control characters while keeping standard
    punctuation, newlines, and Unicode letters/digits.

    Args:
        text: Input text.

    Returns:
        Text with control characters stripped.
    """
    # Keep printable characters + newlines + tabs
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)
    return text


def normalize_spaces(text: str) -> str:
    """
    Replace multiple consecutive spaces (not newlines) with a single space.

    Args:
        text: Input text.

    Returns:
        Text with collapsed horizontal whitespace.
    """
    # Collapse horizontal whitespace only (preserve newlines)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def clean_text(raw_text: str) -> str:
    """
    Master cleaning pipeline. Applies all cleaning steps in order.

    Args:
        raw_text: Raw text string extracted from a PDF page.

    Returns:
        Fully cleaned and normalized text string.
    """
    text = remove_special_characters(raw_text)
    text = remove_page_numbers(text)
    text = normalize_spaces(text)
    text = remove_excessive_whitespace(text)
    return text
