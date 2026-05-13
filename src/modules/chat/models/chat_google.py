"""
This block establishes an implementation for Google models via API-based variants.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import os
import time
import random
import logging
from typing import Any, List, Optional, Dict, Literal
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun

# Import the Google Gen AI SDK
from google import genai
from google.genai import types

from src.modules.chat.base import BaseChat

# Load environment variables from .env file
load_dotenv()

# Set logger
logger = logging.getLogger(__name__)

# Retry configuration for rate limiting
MAX_RETRIES = 20
INITIAL_BACKOFF = 5  # seconds
MAX_BACKOFF = 120  # seconds
BACKOFF_MULTIPLIER = 2
JITTER_RANGE = 0.5  # ±50% jitter


# =========================================================================== #
#                             Google Chat Pipeline                            #
# =========================================================================== #
class _ChatGoogle(BaseChat):
    """
    Chat model implementation for Google-family models using the Google Gen AI SDK.
    """
    # Google specific config
    api_key: Optional[str] = None
    client: Optional[Any] = None
    
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Google chat model configs.

        Args:
            model_name (Optional[str]): Name of Google model.
                Defaults to None.
            max_tokens (int): Maximum number of tokens to generate.
                Defaults to 256.
            temperature (float): Temperature for generation.
                Defaults to 0.0.
            top_p (float): Top-p sampling parameter.
                Defaults to 1.0
            api_key (Optional[str]): API Token for Google.
                Defaults to None.
            client (Optional[Any]): Google API connection client.
                Defaults to None.
            last_k_messages (int): Number of most recent messages to include in context.
                Defaults to 2.
        """
        super().__init__(**kwargs)

        # Parse API token from environment variables
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            logger.error("Google API key not found in environment variables")
            raise ValueError("Missing Google API key. Please make sure it is set in your .env file.")

        # Initialize the Google client
        self._init_model()
        
    @property
    def _llm_type(self) -> str:
        """
        Return identifier for model of use.
        """
        return f"google-{self.model_name}"

    def _init_model(self) -> None:
        """
        Initialize Google client.
        """
        try:                        
            # Initialize Google client with the Google Gen AI SDK
            self.client = genai.Client(api_key=self.api_key)
            
            logger.info(f"Successfully initialized Google client for model: {self._llm_type}")
            
        except Exception as e:
            logger.error(f"Error initializing Google client: {str(e)}")
            raise

    def _clean_response(self, text: str) -> str:
        """
        Clean the response text if needed.
        
        Args:
            text (str): Raw response text from the model.
            
        Returns:
            Cleaned response text.
        """
        return text.strip()
    
    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """
        Format LangChain messages into Google's expected format with proper roles.
        
        Args:
            messages (List[BaseMessage]): List of LangChain messages.
            
        Returns:
            List of formatted messages with roles for Google's API.
        """
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted_messages.append({
                    "role": "user",
                    "parts": [{"text": f"system: {msg.content}"}]
                })
            elif isinstance(msg, HumanMessage):
                formatted_messages.append({
                    "role": "user",
                    "parts": [{"text": msg.content}]
                })
            elif isinstance(msg, AIMessage):
                formatted_messages.append({
                    "role": "model",
                    "parts": [{"text": msg.content}]
                })
            else:
                # Handle any other message types as user messages
                formatted_messages.append({
                    "role": "user",
                    "parts": [{"text": msg.content}]
                })

        return formatted_messages
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Check if an error is retryable (rate limiting, quota exceeded, server errors).
        
        Args:
            error (Exception): The exception to check.
            
        Returns:
            True if the error is retryable, False otherwise.
        """
        error_str = str(error).lower()
        retryable_codes = ['429', '503', '500', '502', '504']
        retryable_messages = [
            'resource_exhausted',
            'quota',
            'rate limit',
            'too many requests',
            'overloaded',
            'temporarily unavailable',
            'internal error',
            'server error',
            'returned empty response',  # Gemini sometimes returns None transiently
            'content is none'
        ]
        
        # Check for retryable HTTP codes
        for code in retryable_codes:
            if code in error_str:
                return True
        
        # Check for retryable error messages
        for msg in retryable_messages:
            if msg in error_str:
                return True
        
        return False
    
    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate backoff time with exponential increase and jitter.
        
        Args:
            attempt (int): Current attempt number (0-indexed).
            
        Returns:
            Backoff time in seconds.
        """
        # Exponential backoff
        backoff = min(INITIAL_BACKOFF * (BACKOFF_MULTIPLIER ** attempt), MAX_BACKOFF)
        # Add jitter (±50%)
        jitter = backoff * JITTER_RANGE * (2 * random.random() - 1)
        return backoff + jitter
        
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate a response from the model given a list of messages.
        Implements retry logic with exponential backoff for rate limiting errors.
        
        Args:
            messages (List[BaseMessage]): List of messages.
            stop (Optional[List[str]]): List of strings to stop generation when encountered.
                Defaults to None.
            run_manager (Optional[CallbackManagerForLLMRun]): Callback manager for the run.
                Defaults to None.
            
        Returns:
            ChatResult with the model response.
        """
        # Use ConversationLatestMemory to only include last-k messages
        prompt_messages = self.memory._prepend_buffer_memory(
            messages, 
            as_prompt=False, 
        )

        # Convert to Google's expected format with proper roles
        formatted_messages = self._format_messages(prompt_messages)
        
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                # Generate content using the properly formatted messages with roles
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=formatted_messages,
                    config=types.GenerateContentConfig(
                            max_output_tokens=self.max_tokens,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            candidate_count=1,
                            stop_sequences=stop if stop else [],
                            # thinking_config=types.ThinkingConfig(thinking_level="low")
                        )            
                )
                
                # Handle empty or blocked responses
                response_text = response.text
                if response_text is None:
                    # Check for blocked content or empty response
                    block_reason = None
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                        block_reason = getattr(response.prompt_feedback, 'block_reason', None)
                    if hasattr(response, 'candidates') and response.candidates:
                        for candidate in response.candidates:
                            if hasattr(candidate, 'finish_reason'):
                                finish_reason = candidate.finish_reason
                                if finish_reason and str(finish_reason).upper() in ['SAFETY', 'RECITATION', 'OTHER']:
                                    block_reason = f"finish_reason={finish_reason}"
                                    break
                    
                    if block_reason:
                        raise ValueError(f"Response blocked by Gemini: {block_reason}")
                    else:
                        raise ValueError("Gemini returned empty response (content is None)")
                
                # Create an AIMessage with the response text
                message = AIMessage(content=response_text)
                generation = ChatGeneration(message=message)
                
                # Return a ChatResult with the generated message
                return ChatResult(generations=[generation])
                
            except Exception as e:
                last_error = e
                
                if self._is_retryable_error(e) and attempt < MAX_RETRIES - 1:
                    backoff_time = self._calculate_backoff(attempt)
                    logger.warning(
                        f"Retryable error on attempt {attempt + 1}/{MAX_RETRIES}: {str(e)}. "
                        f"Retrying in {backoff_time:.1f}s..."
                    )
                    time.sleep(backoff_time)
                else:
                    # Non-retryable error or max retries reached
                    if attempt >= MAX_RETRIES - 1:
                        logger.error(f"Max retries ({MAX_RETRIES}) exceeded. Last error: {str(e)}")
                    else:
                        logger.error(f"Non-retryable error during generation: {str(e)}")
                    break
        
        # Return error message after all retries failed
        error_msg = f"Error generating response after {MAX_RETRIES} attempts: {str(last_error)}"
        message = AIMessage(content=error_msg)
        return ChatResult(generations=[ChatGeneration(message=message)])
    
    def invoke(
        self, 
        messages: List[BaseMessage], 
        **kwargs: Any
    ) -> AIMessage:
        """
        Invoke the model with the given messages.
        
        Args:
            messages (List[BaseMessage]): List of messages.
            
        Returns:
            AI message with the model response.
        """
        result = self._generate(messages, **kwargs)
        return result.generations[0].message