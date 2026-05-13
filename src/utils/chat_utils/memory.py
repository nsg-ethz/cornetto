# References Disclaimer:
# The author acknowledges that the following code block is adapted from an existing repository. Repo can be found at
# https://github.com/RedHatResearch/conext24-NetConfEval/blob/main/netconfeval/foundation/langchain/memory/conversation_latest_memory.py

"""
Helper function to customize ConversationBufferMemory class.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from abc import ABC, abstractmethod
from typing import Any, Union, List, Optional

from langchain_classic.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, \
    SystemMessage, get_buffer_string


# =========================================================================== #
#                         Conversation Latest Memory                          #
# =========================================================================== #
class ConversationLatestMemory(ConversationBufferMemory, ABC):
    """
    Memory class that only retains system message and the last-k messages.
    """
    last_k_messages: int = 2

    def __init__(self, last_k_messages: int = 2, **kwargs: Any) -> None:
        """
        Initialize the memory with a buffer of last-k messages.
        Args:
            last_k_messages (int): Number of last messages preserved.
                Defaults to 2.
        """
        super().__init__(**kwargs)
        self.last_k_messages = last_k_messages

    @property
    def buffer_as_str(self) -> str:
        """
        Exposes the buffer as a string in case return_messages is True
        """
        return get_buffer_string(
            self._get_latest_messages(),
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )
    
    @property
    def buffer_as_messages(self) -> list[BaseMessage]:
        """
        Exposes the buffer as a list of messages in case return_messages is False.
        """
        return self._get_latest_messages()
    
    def _get_latest_messages(self) -> list[BaseMessage]:
        """
        Get the system message (if present) and the last-k user and assistant messages.
        This reduces repetition in the prompt.
        """
        messages = self.chat_memory.messages

        if not messages:
            return []

        # Extract system and all other messages 
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        content_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]

        # Separate user-model messages
        user_msgs = [msg for msg in content_messages if isinstance(msg, HumanMessage)]
        assistant_msgs = [msg for msg in content_messages if isinstance(msg, AIMessage)]

        # If we always alternate, take 2k latest content messages
        k = self.last_k_messages
        n_content = len(content_messages)

        # If last message is HumanMessage and not yet responded to
        last_is_unanswered_user = isinstance(content_messages[-1], HumanMessage) and \
                                (n_content < 2 or not isinstance(content_messages[-2], AIMessage))

        # Base slice size (k full turns = 2k messages)
        base_slice = 2 * k

        # Add 1 more if there is an unanswered user message at the end
        slice_size = base_slice + 1 if last_is_unanswered_user else base_slice

        return system_messages + content_messages[-slice_size:]

    def _prepend_buffer_memory(
        self, 
        messages: Union[BaseMessage, List[BaseMessage]],
        as_prompt: bool = False,
        **kwargs: Any
    ) -> Union[str, List[BaseMessage]]:
        """
        Prepend the buffer memory to the messages.
        
        Args:
            messages (List[BaseMessage]): List of messages.
            as_prompt (bool): Flag to return the memory as a prompt string.
                Defaults to False.
            last_k_messages (Optional[int]): Temporarily override the number of messages to include.
                Defaults to None.

        Returns:
            List of messages with buffer memory prepended.
        """
        if not isinstance(messages, list):
            messages = [messages]

        for message in messages:
            if isinstance(message, HumanMessage):
                self.chat_memory.add_user_message(message.content)
            elif isinstance(message, AIMessage):
                self.chat_memory.add_ai_message(message.content)
            elif isinstance(message, SystemMessage):
                self.chat_memory.add_message(message)  
                    
        result = self.buffer_as_str + "\nAssistant" \
            if as_prompt else self.buffer_as_messages
            
        return result