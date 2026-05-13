"""
This block establishes an implementation for Mistral models via API-based variants.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import os
import logging
from typing import Any, List, Optional, Dict, ClassVar
from dotenv import load_dotenv

from mistralai import Mistral

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun

from src.modules.chat.base import BaseChat

# Load environment variables from .env file
load_dotenv()

# Set logger
logger = logging.getLogger(__name__)


# =========================================================================== #
#                            Mistral Chat Pipeline                            #
# =========================================================================== #
class _ChatMistral(BaseChat):
    """
    Base class for building and running a model via Mistral client.
    """
    # Mistral wrapper specific config
    api_key: str = None
    client: Optional[Any] = None
    
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Mistral chat model configs.

        Args:
            model_name (str): Name of Mistral model.
                Defaults to None.
            max_tokens (int): Maximum number of tokens to generate.
                Defaults to 256.
            temperature (float): Temperature for generation.
                Defaults to 0.0.
            top_p (float): Top-p sampling parameter.
                Defaults to 1.0
            api_key (str): API Token for Mistral API.
                Defaults to None.
            client (Optional[Any]): Mistral API connection client.
                Defaults to None.
        """
        super().__init__(**kwargs)

        # Parse API token and base URL from model kwargs
        self.api_key = kwargs.get("api_key") or os.getenv("MISTRAL_API_KEY")
        
        if not self.api_key:
            logger.error("Mistral API key not found in environment variables")
            raise ValueError("Missing Mistral API key. Please make sure it is set in your .env file.")

        # Initialize client via Mistral wrapper
        self._init_model()
        
    @property
    def _llm_type(self) -> str:
        """
        Return identifier for model of use.
        """
        return f"mistral-{self.model_name}"

    def _init_model(self) -> None:
        """
        Initialize Mistral models' client.
        """
        try:            
            # Initialize the client
            self.client = Mistral(
                api_key=self.api_key,
            )

            logger.info(f"Successfully initialized Mistral client for model: {self._llm_type}")
            
        except Exception as e:
            logger.error(f"Error initializing Mistral client: {str(e)}")
            raise

    def _format_messages(
        self,
        messages: List[BaseMessage]
    ) -> List[Dict[str, str]]:
        """
        Refine messages according to roles.

        Args:
            messages (List[BaseMessage]): List of input messages.

        Returns:
            List of role-based formatted messages. 
        """
        # Search for roles and store messages accordingly
        formatted = []
        previous_role = None
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, HumanMessage):
                role = "user"
            else:
                continue
            
            # Ensure system message is followed by user
            if formatted and formatted[-1]["role"] == "system" and role != "user":
                continue
            
            # Merge consecutive same roles (excluding system)
            if role == previous_role and role != "system":
                formatted[-1]["content"] += "\n" + msg.content
            else:
                formatted.append({"role": role, "content": msg.content})
                previous_role = role
        return formatted

    def _clean_response(self, text: str) -> str:
        """
        Clean the response text if needed. Mistral responses are typically
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
            # Set as_prompt=False, since Mistral models want to see list of messages
            prompt_messages = self.memory._prepend_buffer_memory(
                messages, 
                as_prompt=False, 
            )
            # Format messages
            formatted_messages = self._format_messages(prompt_messages)

            response = self.client.chat.complete(
                model=self.model_name,
                messages=formatted_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                stop=stop
            )
            content = response.choices[0].message.content
            
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