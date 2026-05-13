"""
This block establishes an implementation for Ollama models direct API call.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import re
import json
import logging
import requests
import socket
import time
from typing import Any, List, Optional

import torch

from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun

from src.modules.chat.base import BaseChat

# Set logger
logger = logging.getLogger(__name__)


# =========================================================================== #
#                             Ollama Chat Pipeline                            #
# =========================================================================== #
class _ChatOllama(BaseChat):
    """
    Chat model implementation for Ollama-family models.
    """
    # Ollama specific config
    num_ctx: int = 32_768
    num_gpu: int = 0
    api_base: str = "http://localhost:11434"
    
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Ollama chat model configs.

        Args:
            model_name (Optional[str]): Name of Ollama model.
                Defaults to None.
            max_tokens (int): Maximum number of tokens to generate.
                Defaults to 256.
            temperature (float): Temperature for generation.
                Defaults to 0.0.
            top_p (float): Top-p sampling parameter.
                Defaults to 1.0
            num_ctx (int): Number of tokens for context window.
                Defaults to 32_768.
            num_gpu (int): Number of GPUs to use.
                Defaults to 0.
            api_base (str): Standard API address for Ollama.
                Defaults to 'http://localhost:11434'.
        """
        super().__init__(**kwargs)

        # Set num_gpu based on available CUDA devices
        if torch.cuda.is_available():
            self.num_gpu = torch.cuda.device_count()
            logger.info(f"Setting num_gpu={self.num_gpu} based on available CUDA devices")
        else:
            logger.info("CUDA not available, using CPU for inference")
            
        # Sanity check for API format
        if not self.api_base.startswith("http"):
            self.api_base = f"http://{self.api_base}"
            
        # Initialize the Ollama API connection
        self._init_model()
        logger.info(f"Initialized Ollama model: {self._llm_type}")

    @property
    def _llm_type(self) -> str:
        """
        Return identifier for model of use.
        """
        return f"ollama-{self.model_name}"
        
    def _init_model(self) -> None:
        """
        Initialize Ollama connection by testing the API.
        """
        max_retries = 3
        count_retries = 0
        delay_retries = 2

        while count_retries < max_retries:
            try:
                start = time.time()
                response = requests.get(f"{self.api_base}/api/version", timeout=10)
                elapsed = time.time() - start
                
                if response.status_code == 200:
                    version = response.json().get("version", "unknown")
                    logger.info(f"Connected to Ollama API (version: {version}) in {elapsed:.2f}s")
                    return
                else:
                    logger.warning(f"Connected to Ollama API but received status code {response.status_code}")
                    
            except (requests.exceptions.RequestException, socket.error) as e:
                logger.warning(f"Attempt {count_retries + 1}/{max_retries} failed to connect to Ollama API: {str(e)}")
                count_retries += 1

                if count_retries < max_retries:
                    logger.info(f"Retrying in {delay_retries} seconds...")
                    time.sleep(delay_retries)
                    delay_retries *= 2
                else:
                    logger.error("Max retries reached. Unable to connect to Ollama API.")
                    raise ConnectionError(f"Cannot connect to Ollama API: {str(e)}")

        logger.info("Ollama API connection established successfully.")
            
    def _clean_response(self, text: str) -> str:
        """
        Clean the response text if needed. Ollama answers
        typically have <think> tags that need to be removed.

        Args:
            text (str): Raw response text from the model.
            
        Returns:
            Cleaned response text.
        """
        # Remove any <think> tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)            
        return text.strip()

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate a response from the Ollama model given a list of messages.
        
        Args:
            messages (List[BaseMessage]): List of messages.
            stop (Optional[List[str]]): List of strings to stop generation.
            run_manager (Optional[CallbackManagerForLLMRun]): Callback manager.
            
        Returns:
            ChatResult with the model response.
        """
        try:
            # Use ConversationLatestMemory to only include last-k messages
            prompt_messages = self.memory._prepend_buffer_memory(
                messages, 
                as_prompt=True, 
            )

            logger.info(f"Sending {len(messages)} messages to Ollama API")
            logger.info(f"Starting generate request to Ollama model...")
                        
            # Send an API request to the 'generate' endpoint
            response = requests.post(
                f"{self.api_base}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt_messages,
                    "num_predict": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_ctx": self.num_ctx,
                    "num_gpu": self.num_gpu,
                    "stream": False,
                },
                timeout=100
            )
                        
            # Process the response
            if response.status_code == 200:
                json_obj = response.json()
                full_response = json_obj.get("response", "")
                
                # Clean up the response
                full_response = self._clean_response(full_response)
                logger.info(f"Received response of length {len(full_response)}")
                
                message = AIMessage(content=full_response)
                return ChatResult(generations=[ChatGeneration(message=message)])
            else:
                error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                message = AIMessage(content=f"Error generating response: {error_msg}")
                return ChatResult(generations=[ChatGeneration(message=message)])
                
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            message = AIMessage(content=f"Error generating response: {str(e)}")
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