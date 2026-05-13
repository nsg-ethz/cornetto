""" 
This script contains method to approximate the number of tokens in a text instance. 
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import tiktoken
import  logging

# Set up logging
logger = logging.getLogger(__name__)


# =========================================================================== #
#                          Token Counting Function                            #
# =========================================================================== #
def token_counter(
    text: str, 
    model: str = "gpt-5-mini",
    log_count = True
) -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text (str): Text to count tokens for.
        model (str): Model name for tokenizer. 
            Defaults to 'gpt-5-mini'.
        
    Returns:
        Number of tokens.
    """
    try:
        # Try for the selected model
        encoding = tiktoken.encoding_for_model(model)
        if log_count:
            logger.info(f"Encoded using tokenizer for the model: {model}")
    except KeyError:
        # Fallback to o200k_base encoding if model not found
        encoding = tiktoken.get_encoding("o200k_base")
        if log_count:
            logger.info("Used the base encoding")
    
    # Print the reached context length in the console
    context_length = len(encoding.encode(text))
    if log_count:
        logger.info(f"Context length of {context_length} is reached!")

    return context_length