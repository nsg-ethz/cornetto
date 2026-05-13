"""
Custom exception classes for parser failure modes.
These enable granular tracking of different failure categories.
"""

from enum import Enum
from typing import Optional, Dict, Any


class FailureCategory(str, Enum):
    """High-level failure categories for benchmark results."""
    
    # Parsing/Format failures
    YAML_SYNTAX = "yaml_syntax"           # Invalid YAML syntax
    YAML_EMPTY = "yaml_empty"             # Empty or None YAML response
    YAML_STRUCTURE = "yaml_structure"     # Valid YAML but wrong structure (missing keys, etc.)
    
    # Search/Replace algorithm failures
    SEARCH_NOT_FOUND = "search_not_found"       # Search block not found in config
    SEARCH_MULTIPLE = "search_multiple_matches" # Search block matches multiple times
    SEARCH_EMPTY = "search_empty_block"         # Search block is empty after normalization
    REPLACEMENT_INVALID = "replacement_invalid" # Invalid replacement entry structure
    
    # Config/Data errors
    MISSING_CONFIG = "missing_config"     # Referenced config file doesn't exist
    EMPTY_RESULT = "empty_result"         # No configurations generated
    
    # API/Network errors
    API_ERROR = "api_error"               # General API error
    API_QUOTA = "api_quota"               # Rate limit or quota exceeded
    API_TIMEOUT = "api_timeout"           # Request timeout
    CONTEXT_OVERFLOW = "context_overflow" # Context length exceeded
    
    # Other
    UNKNOWN = "unknown"                   # Unclassified error
    NONE = "none"                         # No failure (success)
    PARSE_RETRY = "parse_retry"           # Parsing failed but retried successfully


class ParserError(Exception):
    """Base exception for parser errors with failure categorization."""
    
    def __init__(
        self, 
        message: str, 
        category: FailureCategory,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.category = category
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to a dictionary for logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "category": self.category.value,
            "details": self.details,
        }


class YAMLSyntaxError(ParserError):
    """Raised when YAML syntax is invalid."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        details = {}
        if original_error:
            details["original_error"] = str(original_error)
            details["original_type"] = type(original_error).__name__
        super().__init__(message, FailureCategory.YAML_SYNTAX, details)


class YAMLEmptyError(ParserError):
    """Raised when YAML content is empty or None."""
    
    def __init__(self, message: str = "Parsed YAML is empty or None"):
        super().__init__(message, FailureCategory.YAML_EMPTY)


class YAMLStructureError(ParserError):
    """Raised when YAML structure is invalid (missing keys, wrong types)."""
    
    def __init__(self, message: str, missing_key: Optional[str] = None):
        details = {}
        if missing_key:
            details["missing_key"] = missing_key
        super().__init__(message, FailureCategory.YAML_STRUCTURE, details)


class SearchNotFoundError(ParserError):
    """Raised when search block is not found in config."""
    
    def __init__(
        self, 
        message: str, 
        filename: str, 
        search_block: Optional[str] = None,
        strategies_tried: Optional[list] = None
    ):
        details = {
            "filename": filename,
            "strategies_tried": strategies_tried or ["exact", "whitespace_normalized", "regex_tolerant", "difflib_similarity"],
        }
        if search_block:
            # Truncate for logging
            details["search_block_preview"] = search_block[:200] + "..." if len(search_block) > 200 else search_block
        super().__init__(message, FailureCategory.SEARCH_NOT_FOUND, details)


class SearchMultipleMatchesError(ParserError):
    """Raised when search block matches multiple times."""
    
    def __init__(self, message: str, filename: str, match_count: Optional[int] = None):
        details = {"filename": filename}
        if match_count:
            details["match_count"] = match_count
        super().__init__(message, FailureCategory.SEARCH_MULTIPLE, details)


class SearchEmptyBlockError(ParserError):
    """Raised when search block is empty after normalization."""
    
    def __init__(self, message: str, filename: str):
        super().__init__(message, FailureCategory.SEARCH_EMPTY, {"filename": filename})


class ReplacementInvalidError(ParserError):
    """Raised when replacement entry has invalid structure."""
    
    def __init__(self, message: str, filename: str, entry: Optional[Any] = None):
        details = {"filename": filename}
        if entry:
            details["entry_preview"] = str(entry)[:200]
        super().__init__(message, FailureCategory.REPLACEMENT_INVALID, details)


class MissingConfigError(ParserError):
    """Raised when a referenced config file doesn't exist."""
    
    def __init__(self, message: str, filename: str):
        super().__init__(message, FailureCategory.MISSING_CONFIG, {"filename": filename})


class EmptyResultError(ParserError):
    """Raised when no configurations were generated."""
    
    def __init__(self, message: str = "No configurations were generated"):
        super().__init__(message, FailureCategory.EMPTY_RESULT)


def classify_error(error: Exception) -> FailureCategory:
    """
    Classify an exception into a failure category.
    
    Args:
        error: The exception to classify.
        
    Returns:
        The appropriate FailureCategory.
    """
    if isinstance(error, ParserError):
        return error.category
    
    error_str = str(error).lower()
    
    # Search/replace errors - check FIRST as they're often more specific
    # and error messages may be wrapped with additional context
    if "invalid replacement" in error_str or "missing 'replace'" in error_str:
        return FailureCategory.REPLACEMENT_INVALID
    
    if "search block" in error_str or ("not found" in error_str and "search" in error_str):
        if "multiple" in error_str:
            return FailureCategory.SEARCH_MULTIPLE
        if "empty" in error_str:
            return FailureCategory.SEARCH_EMPTY
        return FailureCategory.SEARCH_NOT_FOUND
    
    if "empty search" in error_str or "empty block" in error_str:
        return FailureCategory.SEARCH_EMPTY
    
    # YAML errors
    if "yaml" in error_str:
        if "empty" in error_str or "none" in error_str:
            return FailureCategory.YAML_EMPTY
        if "structure" in error_str or "expected" in error_str:
            return FailureCategory.YAML_STRUCTURE
        return FailureCategory.YAML_SYNTAX
    
    if "replacement" in error_str and "invalid" in error_str:
        return FailureCategory.REPLACEMENT_INVALID
    
    # Config errors
    if "missing" in error_str and "config" in error_str:
        return FailureCategory.MISSING_CONFIG
    
    if "empty" in error_str and ("result" in error_str or "config" in error_str):
        return FailureCategory.EMPTY_RESULT
    
    # API errors
    if any(x in error_str for x in ["rate limit", "quota", "429"]):
        return FailureCategory.API_QUOTA
    
    if any(x in error_str for x in ["timeout", "timed out"]):
        return FailureCategory.API_TIMEOUT
    
    if any(x in error_str for x in ["context", "overflow", "too long", "maximum context"]):
        return FailureCategory.CONTEXT_OVERFLOW
    
    if any(x in error_str for x in ["api", "request failed", "connection"]):
        return FailureCategory.API_ERROR
    
    return FailureCategory.UNKNOWN


def format_failure_mode(
    category: FailureCategory,
    fuzzy_match_used: bool = False,
    fuzzy_match_count: int = 0
) -> Dict[str, Any]:
    """
    Format failure mode information for result logging.
    
    Args:
        category: The failure category.
        fuzzy_match_used: Whether fuzzy matching was used.
        fuzzy_match_count: Number of fuzzy matches applied.
        
    Returns:
        Dictionary with failure mode details.
    """
    return {
        "failure_mode": category.value,
        "failure_category_group": _get_category_group(category),
        "fuzzy_match_used": fuzzy_match_used,
        "fuzzy_match_count": fuzzy_match_count,
    }


def _get_category_group(category: FailureCategory) -> str:
    """Get the high-level group for a failure category."""
    yaml_categories = {
        FailureCategory.YAML_SYNTAX,
        FailureCategory.YAML_EMPTY,
        FailureCategory.YAML_STRUCTURE,
    }
    search_replace_categories = {
        FailureCategory.SEARCH_NOT_FOUND,
        FailureCategory.SEARCH_MULTIPLE,
        FailureCategory.SEARCH_EMPTY,
        FailureCategory.REPLACEMENT_INVALID,
    }
    config_categories = {
        FailureCategory.MISSING_CONFIG,
        FailureCategory.EMPTY_RESULT,
    }
    api_categories = {
        FailureCategory.API_ERROR,
        FailureCategory.API_QUOTA,
        FailureCategory.API_TIMEOUT,
        FailureCategory.CONTEXT_OVERFLOW,
    }
    
    if category in yaml_categories:
        return "yaml_parsing"
    if category in search_replace_categories:
        return "search_replace"
    if category in config_categories:
        return "config_data"
    if category in api_categories:
        return "api_network"
    if category == FailureCategory.NONE:
        return "success"
    if category == FailureCategory.PARSE_RETRY:
        return "recovered"
    return "other"
