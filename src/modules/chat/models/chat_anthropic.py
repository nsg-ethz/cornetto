"""
This block establishes an implementation for Anthropic models via API-based variants.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import os
import logging
from typing import Any, List, Optional, Dict
from dotenv import load_dotenv

from anthropic import Anthropic

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun

from src.modules.chat.base import BaseChat

# Load environment variables from .env file
load_dotenv()

# Set logger
logger = logging.getLogger(__name__)


# =========================================================================== #
#                            Anthropic Chat Pipeline                            #
# =========================================================================== #
class _ChatAnthropic(BaseChat):
    """
    Base class for building and running a model via Anthropic client.
    """
    # Anthropic wrapper specific config
    api_key: str = None
    client: Optional[Any] = None
    
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Anthropic chat model configs.

        Args:
            model_name (str): Name of Anthropic model.
                Defaults to None.
            max_tokens (int): Maximum number of tokens to generate.
                Defaults to 256.
            temperature (float): Temperature for generation.
                Defaults to 0.0.
            top_p (float): Top-p sampling parameter.
                Defaults to 1.0
            api_key (str): API Token for Anthropic API.
                Defaults to None.
            client (Optional[Any]): Anthropic API connection client.
                Defaults to None.
            last_k_messages (int): Number of most recent messages to include in context.
                Defaults to 2.
        """
        super().__init__(**kwargs)

        # Parse API token - note the correct env var name
        self.api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            logger.error("Anthropic API key not found in environment variables")
            raise ValueError("Missing Anthropic API key. Please make sure ANTHROPIC_API_KEY is set in your .env file.")

        # Initialize client via Anthropic wrapper
        self._init_model()
        
    @property
    def _llm_type(self) -> str:
        """
        Return identifier for model of use.
        """
        return f"anthropic-{self.model_name}"

    def _init_model(self) -> None:
        """
        Initialize Anthropic models' client.
        """
        try:            
            # Initialize the client
            self.client = Anthropic(
                api_key=self.api_key,
            )

            logger.info(f"Successfully initialized Anthropic client for model: {self._llm_type}")
            
        except Exception as e:
            logger.error(f"Error initializing Anthropic client: {str(e)}")
            raise

    def _format_messages(
        self,
        messages: List[BaseMessage]
    ) -> tuple[Optional[str], List[Dict[str, str]]]:
        """
        Refine messages according to Anthropic's format.
        Anthropic requires system messages to be separate from the messages list.

        Args:
            messages (List[BaseMessage]): List of input messages.

        Returns:
            List of role-based formatted messages.
        """
        system_message = None
        formatted = []
        previous_role = None
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                # System messages go in a separate parameter
                if system_message is None:
                    system_message = msg.content
                else:
                    # Multiple system messages - concatenate them
                    system_message += "\n\n" + msg.content
                continue
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, HumanMessage):
                role = "user"
            else:
                # Default to user for unknown message types
                role = "user"
            
            # Merge consecutive same roles
            if role == previous_role and formatted:
                formatted[-1]["content"] += "\n\n" + msg.content
            else:
                formatted.append({"role": role, "content": msg.content})
                previous_role = role
                
        return system_message, formatted

    def _clean_response(self, text: str) -> str:
        """
        Clean the response text if needed. Anthropic responses are typically
        clean but we maintain this for consistency with the base class.
        
        Args:
            text (str): Raw response text from the model.
            
        Returns:
            Cleaned response text.
        """
        return text.strip()
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate a response from the model given a list of messages.
        
        Args:
            messages (List[BaseMessage]): List of messages.
            stop (Optional[List[str]]): List of strings to stop generation when encountered.
                Defaults to None.
            run_manager (Optional[CallbackManagerForLLMRun]): Callback manager for the run.
                Defaults to None.
            
        Returns:
            ChatResult with the model response.
        """
        try:            
            # Use ConversationLatestMemory to only include last-k messages
            # Set as_prompt=False, since Anthropic models want to see list of messages
            prompt_messages = self.memory._prepend_buffer_memory(
                messages, 
                as_prompt=False, 
            )
            
            # Format messages and extract system prompt
            system_message, formatted_messages = self._format_messages(prompt_messages)

            # Build API call parameters
            api_params = {
                "model": self.model_name,
                "messages": formatted_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                # "top_p": self.top_p, # anthropic allows to set either temperature or top_p only
            }
            
            # Add system message if present
            if system_message:
                api_params["system"] = system_message
            
            # Add stop sequences if provided
            if stop:
                api_params["stop_sequences"] = stop
            
            # Use stream and get the final message
            content = ""
            with self.client.messages.stream(**api_params) as stream:
                for text in stream.text_stream:
                    content += text
            
            # Write content from the response
            message = AIMessage(content=content)
            
            # Return a ChatResult with the generated message
            return ChatResult(generations=[ChatGeneration(message=message)])
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            return ChatResult(generations=[
                ChatGeneration(message=AIMessage(content=f"Error generating response: {str(e)}"))
            ])
        
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