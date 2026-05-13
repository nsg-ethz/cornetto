"""
Configuration parsers for different output formats.
Supports outputs both in full configuration and unified diff patch formats.
"""

import json
import csv
import io
import os
import subprocess
import tempfile
import re
import yaml
import logging
from typing import Dict, Any, Tuple, List

from .errors import (
    ParserError,
    YAMLSyntaxError,
    YAMLEmptyError,
    YAMLStructureError,
    SearchNotFoundError,
    SearchMultipleMatchesError,
    SearchEmptyBlockError,
    ReplacementInvalidError,
    MissingConfigError,
    EmptyResultError,
    FailureCategory,
)

logger = logging.getLogger(__name__)


# =========================================================================== #
#                         Model Output Pre-Processing                         #
# =========================================================================== #
def _strip_model_wrappers(content: str) -> str:
    """
    Strip model-specific wrapper tokens (thinking tags, channel tags,
    harmony format tokens) from raw model output before parsing.

    This is a shared pre-processing step that all parsers call before
    attempting any format-specific parsing to handle reasoning model
    outputs that wrap structured content inside control tokens.

    Handles:
      - <think>...</think> tags (DeepSeek-R1, QwQ, Gemma 3 QAT, etc.)
      - <|channel>thought\\n...<channel|> (Gemma 4)
      - Harmony format: <|channel|>analysis...<|end|> / <|channel|>final...<|return|> (GPT-OSS)

    Args:
        content (str): Raw model output string.

    Returns:
        Cleaned content with model-specific wrappers removed.
    """
    if not content:
        return content

    # ----------------------------------------------------------------------#
    # 1. Strip <think>...</think> blocks (DeepSeek-R1, QwQ, Gemma 3, etc.)  #
    # ----------------------------------------------------------------------#
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    # Handle unclosed <think> tags (model may have been cut off)
    content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
    
    # If there is only a closing think tag (happens with Qwen for example)
    if '</think>' in content:
        content = content.split('</think>')[-1].strip()

    # ----------------------------------------------------------------------#
    # 2. Strip Gemma 4 thought channel blocks                               #
    #    Format: <|channel>thought\n...<channel|>                           #
    # ----------------------------------------------------------------------#
    content = re.sub(
        r'<\|channel>thought\n.*?<channel\|>', '', content, flags=re.DOTALL
    )
    # Handle unclosed Gemma 4 thought blocks
    content = re.sub(
        r'<\|channel>thought\n.*', '', content, flags=re.DOTALL
    )

    # ----------------------------------------------------------------------#
    # 3. Strip Harmony (GPT-OSS) analysis/commentary channel blocks,        #
    #    keep only the final channel content                                #
    #    Format: <|start|>assistant<|channel|>analysis<|message|>...<|end|> #
    #            <|start|>assistant<|channel|>final<|message|>...<|return|> #
    # ----------------------------------------------------------------------#
    # Remove analysis and commentary blocks entirely
    content = re.sub(
        r'<\|start\|>assistant<\|channel\|>(?:analysis|commentary)<\|message\|>.*?<\|end\|>',
        '', content, flags=re.DOTALL
    )

    # Extract final channel content from harmony if present
    final_match = re.search(
        r'<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)',
        content, flags=re.DOTALL
    )
    if final_match:
        content = final_match.group(1)

    # Clean up any remaining orphaned harmony control tokens
    content = re.sub(
        r'<\|(?:start|end|return|channel|message)\|>[^\n]*', '', content
    )

    return content.strip()


# =========================================================================== #
#                          Full Configuration Parser                          #
# =========================================================================== #
def full_config_parser(
    content: str,
    faulty_configs: Dict[str, str] = None
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Parse response content containing full router configurations.
    
    Args:
        content (str): Raw response content string.
        faulty_configs (Dict[str, str]): Original faulty configurations.
        
    Returns:
        Tuple of parsed configs and metadata.
    """
    # Strip model-specific wrappers (thinking tags, channel tags, etc.)
    content = _strip_model_wrappers(content)

    # Strip markdown code block markers if present
    content = re.sub(r'^```\s*yaml\s*\n', '', content)
    content = re.sub(r'\n```\s*$', '', content)

    # Parse YAML
    try:
        response_yaml = yaml.safe_load(content)
        if not response_yaml:
            raise ValueError("Parsed YAML is empty or None")

        # Extract configs and metadata
        try:
            fixed_configs = response_yaml.get("configs", {})
            fix_metadata = response_yaml.get("metadata", {})
        except Exception as e:
            logger.error(f"Failed to extract configs and metadata from YAML: {str(e)}")
            raise

    except Exception as e:
        logger.warning(f"Failed to directly parse model response as YAML, using manual parser now...")
    
        # Split content into sections first to avoid YAML parsing issues
        sections_match = re.search(r'(configs:.*?)metadata:(.*)', content, re.DOTALL)
        if not sections_match:
            raise ValueError("Could not find both configs and metadata sections")
        
        configs_section = sections_match.group(1).strip()
        metadata_section = "metadata:" + sections_match.group(2)
        
        # Process configs section using regex to extract each router config
        fixed_configs = {}
        
        # Pattern to match router.cfg: | or router.cfg: No change needed
        router_pattern = re.compile(
            r'(\S+\.cfg):\s*(?:(\|\s*\n)(.*?)(?=\n\s*\S+\.cfg:|$)|(\s*No change needed))', re.DOTALL
        )
        
        for match in router_pattern.finditer(configs_section):
            filename = match.group(1)
            
            # Handle "No change needed" case
            if match.group(4) and "No change needed" in match.group(4):
                fixed_configs[filename] = "No change needed"
                continue
                
            # Process router config
            if match.group(3):
                config_content = match.group(3)
                lines = config_content.split('\n')
                processed_lines = []
                
                for line in lines:
                    if line.strip():
                        stripped = line.lstrip()
                        if line.startswith(' ') and stripped[0].islower():
                            processed_lines.append(' ' + stripped)
                        else:
                            processed_lines.append(stripped)
                    else:
                        processed_lines.append('')  
                fixed_configs[filename] = '\n'.join(processed_lines)
        
        if not fixed_configs:
            logger.error("Manual parsing failed... Could not extract any router configurations")
            raise ValueError("Complete parsing failure... No router configurations found")

        fix_metadata = {}
        try:
            metadata_yaml = yaml.safe_load(metadata_section)
            fix_metadata = metadata_yaml.get("metadata", {})
        except Exception as e:
            # Fallback to regex if YAML parsing fails for metadata
            logger.warning(f"Failed to parse metadata as YAML: {str(e)}")
            problem_match = re.search(
                r'problem_diagnosis:\s*\|(.*?)(?=\n\s*proposed_fix:|$)', metadata_section, re.DOTALL
            )
            fix_match = re.search(r'proposed_fix:\s*\|(.*?)$', metadata_section, re.DOTALL)
            
            if problem_match:
                fix_metadata["problem_diagnosis"] = problem_match.group(1).strip()
            if fix_match:
                fix_metadata["proposed_fix"] = fix_match.group(1).strip()
    
    # Handle "No change needed" cases
    all_configs = {}
    for filename, config_text in fixed_configs.items():
        if config_text != "No change needed":
            all_configs[filename] = config_text
        else:
            # Use faulty config for "No change needed" case
            if faulty_configs and filename in faulty_configs:
                all_configs[filename] = faulty_configs[filename]
            else:
                logger.warning(f"No faulty config found for unchanged file: {filename}")
                all_configs[filename] = "No change needed"

    return all_configs, fix_metadata


# =========================================================================== #
#                              Diff Patch Parser                              #
# =========================================================================== #
def diff_patch_parser(
    content: str,
    faulty_configs: Dict[str, str] = None
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Parse response content containing unified diff patches and apply them.
    
    Args:
        content (str): Raw response content string containing patches.
        faulty_configs (Dict[str, str]): Original faulty configurations.
        
    Returns:
        Tuple of fixed configs (with patches applied) and metadata.
    """
    if not faulty_configs:
        raise ValueError("faulty_configs must be provided for patch parsing")
    
    # Strip model-specific wrappers (thinking tags, channel tags, etc.)
    content = _strip_model_wrappers(content)

    # Strip markdown code block markers if present
    content = re.sub(r'^```\s*yaml\s*\n', '', content)
    content = re.sub(r'\n```\s*$', '', content)

    # Parse YAML
    try:
        response_yaml = yaml.safe_load(content)
        if not response_yaml:
            raise ValueError("Parsed YAML is empty or None")

        patches_dict = response_yaml.get("patches", {})
        fix_metadata = response_yaml.get("metadata", {}) or {}
        
        if not patches_dict:
            raise ValueError("No patches section found in response")

    except Exception as e:
        logger.warning(f"Failed to parse model response as YAML: {str(e)}")
        logger.error("Manual parsing for patch format not implemented")
        raise ValueError(f"Could not parse patches from response: {str(e)}")
    
    # Apply patches to faulty configs
    fixed_configs = {}
    patch_failures = []
    
    for filename, patch_content in patches_dict.items():        
        # Apply the patch
        if filename in faulty_configs:
            try:
                fixed_config = apply_unified_diff(
                    faulty_configs[filename], 
                    patch_content
                )
                fixed_configs[filename] = fixed_config
                logger.info(f"Successfully applied patch to {filename}")
            except Exception as e:
                logger.error(f"Failed to apply patch to {filename}: {str(e)}")
                # Fall back to original faulty config on patch failure
                fixed_configs[filename] = faulty_configs[filename]
                patch_failures.append(filename)
        else:
            logger.error(f"Cannot apply patch to {filename}: original config not found")
            raise ValueError(f"Missing original config for {filename}")
    
    # Ensure all routers are present
    for filename in faulty_configs:
        if filename not in fixed_configs:
            # Router not mentioned in patches means no change
            fixed_configs[filename] = faulty_configs[filename]
            logger.info(f"Router {filename} not in patches, keeping original config")

    # Validate that we have all configs
    if not fixed_configs:
        logger.error("No configurations were generated after applying patches")
        raise ValueError("Patch application resulted in empty configs")
    
    if patch_failures:
        fix_metadata["patch_failures"] = patch_failures
        fix_metadata["patch_fail_count"] = len(patch_failures)
    else:
        fix_metadata.setdefault("patch_fail_count", 0)
        fix_metadata.setdefault("patch_failures", [])

    return fixed_configs, fix_metadata


# =========================================================================== #
#                        Search-and-Replace Parser                            #
# =========================================================================== #
def search_replace_parser(
    content: str,
    faulty_configs: Dict[str, str]
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Parse response content containing search-and-replace instructions and apply them.
    
    Args:
        content (str): Raw response content string containing replacements.
        faulty_configs (Dict[str, str]): Original faulty configurations.
        
    Returns:
        Tuple of fixed configs (after applying replacements) and metadata.
    """
    if not faulty_configs:
        raise ValueError("faulty_configs must be provided for search-and-replace parsing")

    # Strip model-specific wrappers (thinking tags, channel tags, etc.)
    content = _strip_model_wrappers(content)

    # Strip markdown code block markers if present
    content = re.sub(r'^```\s*yaml\s*\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'\n```\s*$', '', content)
    # Also handle code blocks that might appear in the middle
    content = re.sub(r'```\s*yaml\s*\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'```\s*\n', '', content, flags=re.MULTILINE)

    def fix_yaml_block_scalars(text):
        """Fix inconsistent indentation in YAML literal block scalars."""
        lines = text.split('\n')
        result = []
        in_block = False
        block_indent = 0
        first_content_line = True
        
        for i, line in enumerate(lines):
            # Check if this starts a new block scalar
            match = re.match(r'^(\s*)-?\s*(search|replace):\s*\|', line)
            if match:
                result.append(line)
                in_block = True
                first_content_line = True
                # Calculate expected content indentation
                key_indent = len(match.group(1))
                block_indent = key_indent + 8  # Standard 8-space indent for content
                continue
            
            # Check if we're exiting the block
            if in_block and line.strip():
                # Exit if we see a YAML key at same or lower indentation level
                if re.match(r'^\s*(-\s+)?(search|replace|routers|metadata|replacements):', line):
                    in_block = False
                    first_content_line = True
            
            # Process content lines within block
            if in_block:
                if line.strip():  # Non-empty line
                    # Detect actual indentation on first content line
                    if first_content_line:
                        actual_indent = len(line) - len(line.lstrip())
                        if actual_indent > 0:
                            block_indent = actual_indent
                        first_content_line = False
                    
                    # Normalize indentation
                    stripped = line.lstrip()
                    result.append(' ' * block_indent + stripped)
                else:
                    # Preserve empty lines
                    result.append(line)
            else:
                result.append(line)
        
        return '\n'.join(result)

    # Try parsing directly first
    try:
        response_yaml = yaml.safe_load(content)
        if not response_yaml:
            raise YAMLEmptyError("Parsed YAML is empty or None")

        replacements_dict = response_yaml.get("replacements", {})
        fix_metadata = response_yaml.get("metadata", {})

        if replacements_dict is None:
            replacements_dict = {}

        if not isinstance(replacements_dict, dict):
            raise YAMLStructureError("replacements section is not a mapping", missing_key="replacements")

        logger.info("Successfully parsed YAML on first attempt")

    except (yaml.YAMLError, YAMLEmptyError, YAMLStructureError) as yaml_err:
        # Apply progressive fixes
        logger.warning(f"Initial YAML parse failed: {str(yaml_err)}")
        
        # Strategy 1: Fix block scalar indentation
        logger.info("Attempting fix: normalize block scalar indentation")
        fixed_content = fix_yaml_block_scalars(content)
        
        try:
            response_yaml = yaml.safe_load(fixed_content)
            if not response_yaml:
                raise YAMLEmptyError("Parsed YAML is empty or None")
            
            logger.info("Successfully parsed YAML after fixing block scalar indentation")
            
            replacements_dict = response_yaml.get("replacements", {})
            fix_metadata = response_yaml.get("metadata", {})

            if replacements_dict is None:
                replacements_dict = {}

            if not isinstance(replacements_dict, dict):
                raise YAMLStructureError("replacements section is not a mapping", missing_key="replacements")
                
        except Exception as retry_err:
            # Strategy 2: Try removing problematic metadata sections
            logger.warning(f"Block scalar fix failed: {str(retry_err)}")
            logger.info("Attempting fix: extract only replacements section")
            
            try:
                # Extract just the replacements section using regex
                replacements_match = re.search(
                    r'^replacements:\s*\n((?:  .*\n)*)',
                    fixed_content,
                    re.MULTILINE
                )
                
                if replacements_match:
                    replacements_section = "replacements:\n" + replacements_match.group(1)
                    minimal_yaml = yaml.safe_load(replacements_section)
                    
                    if minimal_yaml and isinstance(minimal_yaml.get("replacements"), dict):
                        logger.info("Successfully extracted and parsed replacements section")
                        replacements_dict = minimal_yaml["replacements"]
                        fix_metadata = {}
                    else:
                        raise YAMLStructureError("Extracted replacements is invalid")
                else:
                    raise YAMLSyntaxError("Could not find replacements section")
                    
            except Exception as final_err:
                logger.error(f"All YAML parsing strategies failed: {str(final_err)}")
                logger.debug(f"Content attempted to parse:\n{fixed_content[:500]}...")
                raise YAMLSyntaxError(
                    f"YAML parsing failed after multiple strategies: {str(yaml_err)}",
                    original_error=yaml_err
                )
    
    except Exception as e:
        logger.error(f"Unexpected error parsing model response: {str(e)}")
        raise YAMLSyntaxError(f"Failed to parse model response: {str(e)}", original_error=e)

    fixed_configs: Dict[str, str] = {}
    operations_count = 0
    replacements_applied = 0
    fuzzy_matches = 0
    match_strategies_used: List[str] = []

    def _apply_single_replacement(
        current_config: str,
        search_block: str,
        replace_block: str,
        filename: str,
    ) -> Tuple[str, bool, str]:
        """
        Try exact match first; then fall back to a tolerant regex that ignores
        indentation, multiple blank lines, and lines that are just '!'.
        If that fails, try difflib similarity matching.
        Returns updated config, whether fuzzy matching was used, and match strategy used.
        """
        # Strategy 1: Exact match
        occurrences = current_config.count(search_block)
        if occurrences == 1:
            return current_config.replace(search_block, replace_block, 1), False, "exact"
        if occurrences > 1:
            raise SearchMultipleMatchesError(
                f"Search block matches multiple times in {filename}",
                filename=filename,
                match_count=occurrences
            )

        # Strategy 2: Whitespace-normalized match (strip trailing spaces from each line)
        def normalize_whitespace(text):
            return '\n'.join(line.rstrip() for line in text.split('\n'))
        
        norm_config = normalize_whitespace(current_config)
        norm_search = normalize_whitespace(search_block)
        if norm_config.count(norm_search) == 1:
            # Find position in normalized, apply to original
            idx = norm_config.find(norm_search)
            # Map back to original positions
            orig_idx = len(current_config) - len(current_config.lstrip()) + idx
            # This is approximate; use the normalized approach
            logger.info(f"Using whitespace-normalized match for {filename}")
            return norm_config.replace(norm_search, replace_block, 1), True, "whitespace_normalized"

        # Strategy 3: Flexible regex fallback (tolerates indentation, blank lines, '!' separators)
        def _line_pattern(line: str) -> str:
            stripped = line.strip()
            if stripped == "" or stripped == "!":
                return ""
            escaped = re.escape(stripped)
            # Replace escaped whitespace sequences with flexible whitespace pattern
            # re.escape turns " " into "\ " and "\t" into "\\\t"
            escaped = escaped.replace("\\ ", r"\s+").replace("\\\t", r"\s+")
            return r"[ \t]*" + escaped

        pieces = []
        for ln in search_block.splitlines():
            pat = _line_pattern(ln)
            if pat:
                pieces.append(pat)
        if not pieces:
            raise SearchEmptyBlockError(
                f"Search block is empty after normalization for {filename}",
                filename=filename
            )

        spacer = r"(?:\s*!\s*\n|\s*\n)*"
        flexible_pattern = spacer + spacer.join(pieces) + spacer
        regex = re.compile(flexible_pattern, re.MULTILINE | re.DOTALL)
        matches = list(regex.finditer(current_config))
        if len(matches) == 1:
            start, end = matches[0].span()
            updated = current_config[:start] + replace_block + current_config[end:]
            return updated, True, "regex_tolerant"
        if len(matches) > 1:
            raise SearchMultipleMatchesError(
                f"Search block matches multiple times in {filename} after tolerant match",
                filename=filename,
                match_count=len(matches)
            )

        # Strategy 4: Difflib sequence matching - find most similar block
        import difflib
        search_lines = [l.strip() for l in search_block.strip().splitlines() if l.strip() and l.strip() != '!']
        config_lines = current_config.splitlines()
        
        if search_lines:
            best_ratio = 0
            best_start = -1
            best_end = -1
            window_size = len(search_lines)
            
            # Slide window across config to find best match
            for i in range(len(config_lines) - window_size + 1):
                window = config_lines[i:i + window_size + 2]  # +2 for flexibility
                window_stripped = [l.strip() for l in window if l.strip() and l.strip() != '!']
                
                if len(window_stripped) >= len(search_lines):
                    # Compare using difflib
                    matcher = difflib.SequenceMatcher(None, search_lines, window_stripped[:len(search_lines)])
                    ratio = matcher.ratio()
                    
                    if ratio > best_ratio and ratio >= 0.8:  # 80% similarity threshold
                        best_ratio = ratio
                        best_start = i
                        # Find actual end by matching line count
                        matched_lines = 0
                        best_end = i
                        for j in range(i, min(i + window_size + 5, len(config_lines))):
                            if config_lines[j].strip() and config_lines[j].strip() != '!':
                                matched_lines += 1
                            best_end = j + 1
                            if matched_lines >= len(search_lines):
                                break
            
            if best_ratio >= 0.8:
                logger.info(f"Using difflib similarity match ({best_ratio:.1%}) for {filename}")
                before = '\n'.join(config_lines[:best_start])
                after = '\n'.join(config_lines[best_end:])
                if before:
                    before += '\n'
                if after:
                    after = '\n' + after
                return before + replace_block + after, True, f"difflib_similarity_{best_ratio:.0%}"

        raise SearchNotFoundError(
            f"Search block not found in {filename} (tried exact, whitespace-normalized, "
            f"regex-tolerant, and difflib similarity matching)",
            filename=filename,
            search_block=search_block,
            strategies_tried=["exact", "whitespace_normalized", "regex_tolerant", "difflib_similarity"]
        )

    for filename, operations in replacements_dict.items():
        if filename not in faulty_configs:
            raise MissingConfigError(
                f"Missing original config for {filename}",
                filename=filename
            )

        current_config = faulty_configs[filename]

        # Normalize "no change" cases
        if operations in (None, [], "No change needed"):
            fixed_configs[filename] = current_config
            continue

        if not isinstance(operations, list):
            raise ReplacementInvalidError(
                f"Replacements for {filename} must be a list",
                filename=filename,
                entry=operations
            )

        for op in operations:
            operations_count += 1
            if not isinstance(op, dict) or "search" not in op or "replace" not in op:
                raise ReplacementInvalidError(
                    f"Invalid replacement entry for {filename}: {op}",
                    filename=filename,
                    entry=op
                )

            search_block = op.get("search") or ""
            replace_block = op.get("replace") or ""

            # Skip if search block is empty/None
            if not isinstance(search_block, str) or not search_block.strip():
                logger.warning(f"Empty/None search block for {filename}, skipping operation")
                continue

            # Coerce replace to string
            if not isinstance(replace_block, str):
                replace_block = ""

            current_config, used_fuzzy, strategy = _apply_single_replacement(
                current_config, search_block, replace_block, filename
            )
            replacements_applied += 1
            match_strategies_used.append(strategy)
            if used_fuzzy:
                fuzzy_matches += 1

        fixed_configs[filename] = current_config

    # Ensure all routers are present
    for filename, original in faulty_configs.items():
        if filename not in fixed_configs:
            fixed_configs[filename] = original

    if not fixed_configs:
        logger.error("No configurations were generated after applying replacements")
        raise EmptyResultError("Replacement application resulted in empty configs")

    fix_metadata = fix_metadata or {}
    fix_metadata.setdefault("replacements_count", operations_count)
    fix_metadata.setdefault("replacements_applied", replacements_applied)
    fix_metadata.setdefault("replacements_fuzzy_applied", fuzzy_matches)
    fix_metadata.setdefault("fuzzy_match_used", fuzzy_matches > 0)
    fix_metadata.setdefault("match_strategies_used", match_strategies_used)

    return fixed_configs, fix_metadata

def apply_unified_diff(
    original_config: str, 
    patch_text: str
) -> str:
    """
    Apply a unified diff patch to original config using the system 'patch' utility.
    
    Args:
        original_config (str): Original configuration text.
        patch_text (str): Unified diff patch text.
        
    Returns:
        Patched configuration text.        
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write original config to temp file
        original_file = os.path.join(tmpdir, "config.orig")
        with open(original_file, 'w') as f:
            f.write(original_config)
        
        # Write patch to temp file
        patch_file = os.path.join(tmpdir, "changes.patch")
        with open(patch_file, 'w') as f:
            f.write(patch_text)
        
        # Apply patch using system utility
        try:
            result = subprocess.run(
                ['patch', '-u', '-o', '-', original_file, patch_file],
                capture_output=True,
                text=True,
                check=True,
                cwd=tmpdir
            )
            return result.stdout
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Patch application failed with exit code {e.returncode}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            raise


# =========================================================================== #
#                          JSON Specification Parser                           #
# =========================================================================== #
def json_spec_parser(
    content: str,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Parse response content containing JSON-formatted network specifications.
    
    Expects a JSON object with a "specifications" key containing a list of
    CSV-like strings or dict objects in the format:
    Type, Source_Node, Destination_Prefix, waypoint_node, num_routes, Status

    Args:
        content (str): Raw response content string.

    Returns:
        Tuple of parsed specification dicts and metadata.
    """
    FIELDS = ["type", "source_node", "destination_prefix", "waypoint_node", "num_routes", "status"]
    VALID_TYPES = {"reachability", "waypointing", "isolation", "load_balancing"}
    VALID_STATUSES = {"broken_removed", "broken_added", "intact"}

    # Strip model-specific wrappers (thinking tags, channel tags, etc.)
    cleaned = _strip_model_wrappers(content.strip())

    # Strip markdown code block markers if present
    cleaned = re.sub(r'^```\s*json\s*\n', '', cleaned)
    cleaned = re.sub(r'\n```\s*$', '', cleaned)
    cleaned = re.sub(r'^```\s*\n', '', cleaned)
    cleaned = cleaned.strip()

    rows = None
    parse_strategy = None
    truncated = False

    # Strategy 1: Parse as JSON object with "specifications" key
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "specifications" in parsed:
            rows = parsed["specifications"]
            parse_strategy = "json_object"
        elif isinstance(parsed, list):
            rows = parsed
            parse_strategy = "json_array"
        else:
            raise ValueError("JSON is neither a list nor an object with 'specifications' key")
    except json.JSONDecodeError as e:
        logger.warning(f"Direct JSON parse failed: {e}")

    # Strategy 2: Try to extract JSON from surrounding prose
    if rows is None:
        json_obj_match = re.search(r'\{[^{}]*"specifications"\s*:\s*\[.*?\]\s*\}', cleaned, re.DOTALL)
        if json_obj_match:
            try:
                parsed = json.loads(json_obj_match.group(0))
                rows = parsed["specifications"]
                parse_strategy = "json_object_extracted"
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Extracted JSON object parse failed: {e}")

    if rows is None:
        json_arr_match = re.search(r'\[([^\[\]]*(?:"[^"]*"[^\[\]]*)*)\]', cleaned, re.DOTALL)
        if json_arr_match:
            try:
                rows = json.loads(json_arr_match.group(0))
                parse_strategy = "json_array_extracted"
            except json.JSONDecodeError as e:
                logger.warning(f"Extracted JSON array parse failed: {e}")

    # Strategy 3: Extract individual CSV entries from truncated Python list or JSON
    if rows is None:
        status_pattern = "broken_removed|broken_added|intact"
        csv_matches = re.findall(
            rf"""['"]([^'"]*(?:{status_pattern})[^'"]*)['"]""",
            cleaned,
        )
        if csv_matches:
            rows = csv_matches
            truncated = not cleaned.rstrip().endswith("]")
            parse_strategy = "truncated_list_extracted" if truncated else "python_list_extracted"
            logger.info(
                f"Extracted {len(rows)} complete CSV entries from "
                f"{'truncated' if truncated else 'complete'} list literal"
            )

    # Strategy 4: Extract dict entries from truncated JSON array of objects
    if rows is None:
        dict_matches = re.findall(
            r'\{[^{}]*"type"\s*:\s*"[^"]*"[^{}]*"status"\s*:\s*"[^"]*"[^{}]*\}',
            cleaned,
        )
        if not dict_matches:
            # Try reverse key order (status before type)
            dict_matches = re.findall(
                r'\{[^{}]*"status"\s*:\s*"[^"]*"[^{}]*"type"\s*:\s*"[^"]*"[^{}]*\}',
                cleaned,
            )
        if dict_matches:
            parsed_dicts = []
            for m in dict_matches:
                try:
                    parsed_dicts.append(json.loads(m))
                except json.JSONDecodeError:
                    continue
            if parsed_dicts:
                rows = parsed_dicts
                truncated = not cleaned.rstrip().endswith("]")
                parse_strategy = "truncated_dict_list_extracted" if truncated else "dict_list_extracted"
                logger.info(
                    f"Extracted {len(rows)} complete dict entries from "
                    f"{'truncated' if truncated else 'complete'} list"
                )

    # Strategy 5: Line-by-line fallback
    if rows is None:
        logger.warning("JSON parsing failed, falling back to line-by-line extraction")
        rows = []
        for line in cleaned.splitlines():
            line = line.strip().strip('"').strip("'").rstrip(',')
            if not line:
                continue
            first_field = line.split(',')[0].strip()
            if first_field in VALID_TYPES:
                rows.append(line)
        if rows:
            parse_strategy = "line_fallback"

    if not rows:
        raise ParserError(
            "Could not extract any specifications from response",
            category=FailureCategory.YAML_STRUCTURE,
        )

    # Parse rows into spec dicts
    specs = []
    parse_errors = []

    # Handle rows that are already parsed as dicts
    if rows and isinstance(rows[0], dict):
        for i, row_dict in enumerate(rows):
            spec = {}
            for field in FIELDS:
                spec[field] = str(row_dict.get(field, "")).strip()

            if spec["type"] not in VALID_TYPES:
                parse_errors.append(f"Row {i} has invalid type '{spec['type']}': {row_dict}")
                continue
            if spec["status"] not in VALID_STATUSES:
                parse_errors.append(f"Row {i} has invalid status '{spec['status']}': {row_dict}")
                continue

            specs.append(spec)

        if not specs:
            raise ParserError(
                f"No valid specifications parsed from dict rows. Errors: {parse_errors}",
                category=FailureCategory.YAML_STRUCTURE,
            )

        csv_lines = [
            ",".join([s["type"], s["source_node"], s["destination_prefix"],
                      s["waypoint_node"], s["num_routes"], s["status"]])
            for s in specs
        ]

        metadata = {
            "parse_strategy": parse_strategy + "_dict_rows",
            "total_rows": len(rows),
            "valid_specs": len(csv_lines),
            "parse_errors": parse_errors,
            "truncated": truncated,
        }

        return csv_lines, metadata

    # Handle rows that are CSV-like strings
    for i, row_str in enumerate(rows):
        if not isinstance(row_str, str):
            parse_errors.append(f"Row {i} is not a string: {row_str}")
            continue

        row_str = row_str.strip()
        if not row_str:
            continue

        try:
            fields = next(csv.reader(io.StringIO(row_str)))
        except StopIteration:
            parse_errors.append(f"Row {i} could not be parsed as CSV: {row_str}")
            continue

        # Pad to expected length
        fields += [''] * (len(FIELDS) - len(fields))
        spec = dict(zip(FIELDS, [f.strip() for f in fields[:len(FIELDS)]]))

        # Validate type
        if spec["type"] not in VALID_TYPES:
            parse_errors.append(f"Row {i} has invalid type '{spec['type']}': {row_str}")
            continue

        # Validate status
        if spec["status"] not in VALID_STATUSES:
            parse_errors.append(f"Row {i} has invalid status '{spec['status']}': {row_str}")
            continue

        specs.append(spec)

    if not specs:
        raise ParserError(
            f"No valid specifications parsed. Errors: {parse_errors}",
            category=FailureCategory.YAML_STRUCTURE,
        )

    # Create the CSV-like spec format for evaluation
    csv_lines = [
        ",".join([s["type"], s["source_node"], s["destination_prefix"],
                  s["waypoint_node"], s["num_routes"], s["status"]])
        for s in specs
    ]

    metadata = {
        "parse_strategy": parse_strategy,
        "total_rows": len(rows),
        "valid_specs": len(csv_lines),
        "parse_errors": parse_errors,
        "truncated": truncated,
    }

    return csv_lines, metadata


# =========================================================================== #
#                            Agent Response Parser                            #
# =========================================================================== #
def agent_response_parser(
    response_text: str,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Parse a ReAct-style agent response into thought, action, and parameters.

    Args:
        response_text (str): Raw text output from the LLM.

    Returns:
        Tuple of (thought, action_name, action_params).
        If parsing fails, action_name will be an empty string.
    """
    # Strip model-specific wrappers (thinking tags, channel tags, etc.)
    response_text = _strip_model_wrappers(response_text)

    thought = ""
    action = ""
    params: Dict[str, Any] = {}

    # Extract Thought
    thought_match = re.search(
        r"Thought:\s*(.*?)(?=\nAction:|\Z)", response_text, re.DOTALL
    )
    if thought_match:
        thought = thought_match.group(1).strip()

    # Extract Action
    action_match = re.search(r"Action:\s*(\w+)", response_text)
    if action_match:
        action = action_match.group(1).strip()

    # Extract Action Input
    input_match = re.search(
        r"Action Input:\s*(.*?)(?=\nThought:|\nAction:|\Z)",
        response_text,
        re.DOTALL,
    )
    if input_match:
        action_input_str = input_match.group(1).strip()
        params = _parse_agent_json_params(action_input_str)

    return thought, action, params


def _parse_agent_json_params(raw: str) -> Dict[str, Any]:
    """
    Best-effort JSON extraction from an action input string.

    Args:
        raw (str): Raw action input text (ideally valid JSON).

    Returns:
        Parsed parameters dictionary, or empty dict on failure.
    """
    if not raw:
        return {}

    # Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fallback: extract first JSON object from surrounding text
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse action input as JSON: %s", raw[:200])
    return {}