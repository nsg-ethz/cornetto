"""
Abstract class for building the chat skeleton. Model specifications are 
added on top of this class (e.g., OpenAI-/or HuggingFace-family models).
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from abc import ABC, abstractmethod
from typing import Any, Union, List, Optional
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun

from src.utils.chat_utils.memory import ConversationLatestMemory


# =========================================================================== #
#                              Chat Model Base                                #
# =========================================================================== #
class BaseChat(BaseChatModel, ABC):
    """
    Abstract base class for all chat LLM implementations.
    """
    # Model config
    model_name: str = None
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    last_k_messages: int = 2
    memory: ConversationLatestMemory = None

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the chat model with the given configurations.

        Args:
            model_name (str): Name of the model used.
                Defaults to None.
            max_tokens (int): Maximum number of tokens to generate.
                Defaults to 256.
            temperature (float): Temperature for generation.
                Defaults to 0.0.
            top_p (float): Top-p sampling parameter.
                Defaults to 1.0.
            last_k_messages (int): Number of last messages preserved.
                Defaults to 2.
            memory (ConversationLatestMemory): Conversation memory object.
                Defaults to None.
        """
        super().__init__(**kwargs)
        self.memory = ConversationLatestMemory(
            memory_key="chat_history", 
            last_k_messages=self.last_k_messages,
            return_messages=True,
        )

    @property
    @abstractmethod
    def _llm_type(self) -> str:
        """
        Return identifier for model type.
        """
        pass

    @abstractmethod
    def _init_model(self) -> None:
        """
        Initialize models' client from preferred family
        (e.g. HuggingFace Hub, OpenAI, Ollama).
        """
        pass

    @abstractmethod
    def _clean_response(self, text: str) -> str:
        """
        Clean the response text to remove artifacts and formatting.
        
        Args:
            text (str): Raw response text from the model.
            
        Returns:
            Cleaned response text.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate a response from the model given a list of messages.

        Args:
            messages (List[BaseMessage]): List of messages.
            stop (Optional[List[str]]): List of strings to stop generation when encountered. 
                Defaults to None.
            run_manager (Optional[Any]): Callback manager for LLM run. 
                Defaults to None.

        Returns:
            Model response.
        """
        pass

    @property
    def is_local(self) -> bool:
        """
        Check if the model is running locally.
        
        Returns:
            Flag indicating if the model is local.
        """
        return (hasattr(self, "use_api") and self.use_api is False) or \
               (hasattr(self, "api_base") and self.api_base is None)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Async version of _generate.
        Currently just a wrapper around the sync version.

        Args:
            messages (List[BaseMessage]): List of messages.
            stop (Optional[List[str]]): List of strings to stop generation when encountered. 
                Defaults to None.
            run_manager (Optional[Any]): Callback manager for LLM run. 
                Defaults to None.
        """
        #  Call the sync version
        return self._generate(messages, stop, run_manager, **kwargs)
