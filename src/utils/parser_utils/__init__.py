"""
Parser utilities for configuration output parsing.
"""

from .errors import (
    ParserError,
    FailureCategory,
    YAMLSyntaxError,
    YAMLEmptyError,
    YAMLStructureError,
    SearchNotFoundError,
    SearchMultipleMatchesError,
    SearchEmptyBlockError,
    ReplacementInvalidError,
    MissingConfigError,
    EmptyResultError,
    classify_error,
    format_failure_mode,
)

from .parsers import (
    full_config_parser,
    diff_patch_parser,
    search_replace_parser,
    apply_unified_diff,
)

from .parser_registry import create_parser

__all__ = [
    # Error classes
    "ParserError",
    "FailureCategory",
    "YAMLSyntaxError",
    "YAMLEmptyError",
    "YAMLStructureError",
    "SearchNotFoundError",
    "SearchMultipleMatchesError",
    "SearchEmptyBlockError",
    "ReplacementInvalidError",
    "MissingConfigError",
    "EmptyResultError",
    # Helper functions
    "classify_error",
    "format_failure_mode",
    # Parsers
    "full_config_parser",
    "diff_patch_parser",
    "search_replace_parser",
    "apply_unified_diff",
    "create_parser",
]