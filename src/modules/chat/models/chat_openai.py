"""
This block establishes an implementation for OpenAI models via API-based variants.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import os
import time
import logging
import json
from typing import Any, List, Optional, Dict, ClassVar, Union
from dotenv import load_dotenv

from openai import OpenAI, BadRequestError

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun

from src.modules.chat.base import BaseChat

# Load environment variables from .env file
load_dotenv()

# Set logger
logger = logging.getLogger(__name__)


# =========================================================================== #
#                               OpenAI Base Class                             #
# =========================================================================== #
class BaseOpenAI(BaseChat):
    """
    Base class for building and running a model via OpenAI client.
    """
    # OpenAI wrapper specific config
    model_provider: Optional[str] = None
    api_key: str = None
    base_url: str = None
    batch_api: Optional[bool] = False
    organization: Optional[str] = None
    client: Optional[Any] = None

    # Used reasoning models inventory
    REASONING_MODELS: ClassVar[set[str]] = {
        "gpt-5.2-2025-12-11", "gpt-5.1-2025-11-13", "gpt-5-2025-08-07", 
        "gpt-5-mini-2025-08-07", "o4-mini-2025-04-16", "o3-2025-04-16"
    }
    
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the OpenAI chat model configs.

        Args:
            model_provider (Optional[str]): Original model provider.
                Defaults to None.
            model_name (str): Name of OpenAI model.
                Defaults to None.
            max_tokens (int): Maximum number of tokens to generate.
                Defaults to 256.
            temperature (float): Temperature for generation.
                Defaults to 0.0.
            top_p (float): Top-p sampling parameter.
                Defaults to 1.0
            api_key (str): API Token for OpenAI API.
                Defaults to None.
            base_url (str): Base URL for OpenAI API.
                Defaults to None.
            batch_api (Optional[bool]): OpenAI batch processing service.
                Defaults to False.
            organization (Optional[str]): Organization for OpenAI API.
                Defaults to None. 
            client (Optional[Any]): OpenAI API connection client.
                Defaults to None.
        """
        super().__init__(**kwargs)

        # Parse API token and base URL from model kwargs
        self.api_key = kwargs.get("api_key", "") or os.getenv("OPENAI_API_KEY")

        # Extract base OpenAI url
        self.base_url = kwargs.get("base_url") or "https://api.openai.com/v1"

        # Extract single or batched processing flag
        self.batch_api = kwargs.get("batch_api", False)
        
        if not self.api_key:
            logger.error("OpenAI API key not found in environment variables")
            raise ValueError("Missing OpenAI API key. Please make sure it is set in your .env file.")

        # Extract original provider of the model
        self.model_provider = kwargs.get("provider", "")

        # Initialize client via OpenAI wrapper
        self._init_model()
        
    @property
    def _llm_type(self) -> str:
        """
        Return identifier for model of use.
        """
        return f"{self.model_provider}-{self.model_name}"

    def _init_model(self) -> None:
        """
        Initialize OpenAI models' client.
        """
        try:            
            # Initialize the client
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                max_retries=5,
            )

            logger.info(f"Successfully initialized OpenAI client for model: {self._llm_type}")
            
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            raise

    def _format_messages(
        self,
        messages: List[BaseMessage]
    ) -> List[Dict[str, str]]:
        """
        Refine one-time messages according to roles.

        Args:
            messages (List[BaseMessage]): List of one-time input messages.

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
    
    def _format_batch_messages(
        self,
        batch_messages: Dict[str, Dict[str, Any]],
    ): 
        """
        Refine batch messages according to expected format.

        Args:
            batch_messages (Dict[str, Dict[str, Any]]): Dictionary of batch input messages.

        Returns:
            Batch-wise formatted messages. 
        """
        # Store tasks
        tasks = []

        # Iterate over all examples and create expected format
        for idx, sample in batch_messages.items():
            # Get system and user prompts
            system_prompt = sample["system_prompt"]
            user_prompt = sample["user_prompt"]

            # Create batch tasks one at a time
            task = {
                "custom_id": str(idx),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": f"{self.model_name}",
                    "top_p": self.top_p,
                    "temperature": self.temperature,
                    "max_completion_tokens": self.max_tokens,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                }
            }
            tasks.append(task)

        return tasks
    
    def _clean_response(
        self, 
        text: str
    ) -> Dict[str, str]:
        """
        Clean the response text if needed. OpenAI responses are typically
        clean, but OpenAI batch API response needs relevant cleaning.

        Args:
            text (str): Raw text content of .jsonl file.

        Returns:
            Mapping from custom_id to model-generated content.
        """
        # Collector
        response = {}

        # Iterate over each .jsonl example and extract
        for line in text.strip().splitlines():
            try:
                entry = json.loads(line)
                custom_id = entry.get("custom_id")
                
                # Check if the request was successful
                if entry.get("response", {}).get("status_code") != 200:
                    logger.warning(f"Request failed for task ID: {custom_id}")
                    response[custom_id] = f"Error: Request failed with status {entry.get('response', {}).get('status_code')}"
                    continue
                    
                body = entry["response"]["body"]
                content = body["choices"][0]["message"]["content"]
                response[custom_id] = content
                
            except Exception as e:
                logger.warning(f"Failed to parse line with task ID: {custom_id}... Error: {e}")
                # Still add an entry not to lose track of failed samples
                response[custom_id] = f"Error: Failed to parse response - {str(e)}"
                continue
                
        return response
        
    def _generate(
        self,
        messages: Union[List[BaseMessage], Dict[str, Dict]],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Union[ChatResult, Dict[str, str]]:
        """
        Generate a response from the model given singleton or batch messages.
        
        Args:
            messages (Union[List[BaseMessage], Dict[str, Dict]]): Input messages.
            stop (Optional[List[str]]): List of strings to stop generation when encountered.
                Defaults to None.
            run_manager (Optional[CallbackManagerForLLMRun]): Callback manager for the run.
                Defaults to None.
            
        Returns:
            Either singleton or batch model response.
        """
        # Check if batch processing is enabled
        if self.batch_api:
            try:
                # Fit messages into batch format
                batch_data = self._format_batch_messages(messages)

                # Save batch as .jsonl file
                file_name = f"batch_data_{int(time.time())}.jsonl"
                with open(file_name, "w") as f:
                    for task in batch_data:
                        f.write(json.dumps(task) + "\n")

                # Upload input data
                with open(file_name, "rb") as f:
                    batch_file = self.client.files.create(
                        file=f,
                        purpose="batch"
                    )
                logger.info("Successfully created batch file")

                # Clean up local file
                os.remove(file_name)

                # Create batch file
                batch_file_id = batch_file.id
                logger.info(f"Batch created with ID: {batch_file_id}")

                # Start batch job
                batch_job = self.client.batches.create(
                    input_file_id=batch_file_id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "description": f"Benchmark experiments from {time.time()}"
                    }
                )
                logger.info("Batch job started execution...")

                # Periodically check batch status
                error_detected = False
                while True:
                    # Retrieve job
                    job_id = batch_job.id
                    batch_job = self.client.batches.retrieve(job_id)
                    
                    # Check if failed
                    if batch_job.status == "failed":
                        logger.error(f"Batch job {job_id} failed!")
                        error_detected = True
                        break

                    # Check if in progress
                    elif batch_job.status in ["validating", "in_progress", "finalizing"]:
                        logger.info(f"Batch job {job_id} is {batch_job.status}...")
                
                    # Check if completed
                    elif batch_job.status == "completed":
                        logger.info(f"Batch job {job_id} is completed!")
                        break
                    
                    # Check any other status
                    else:
                        logger.info(f"Batch job {job_id} is in status {batch_job.status}")
                    
                    # Wait before next check
                    time.sleep(20)

                if error_detected:
                    raise Exception("Batch job failed")

                # Extract results if ready
                result_file_id = batch_job.output_file_id
                result = self.client.files.content(result_file_id).text

                # Parse raw result
                parsed_outputs = self._clean_response(result)
                return parsed_outputs
            
            except Exception as e:
                logger.error(f"OpenAI Batch API processing failed due to {e}")
                raise

        # Generate for one-round communication
        else:
            try:            
                # Use ConversationLatestMemory to only include last-k messages
                # Set as_prompt=False, since OpenAI models want to see list of messages
                prompt_messages = self.memory._prepend_buffer_memory(
                    messages, 
                    as_prompt=False, 
                )
                # Format messages
                formatted_messages = self._format_messages(prompt_messages)

                # Generate model response based on its type
                if self.model_name in self.REASONING_MODELS:
                    response = self.client.responses.create(
                        model=self.model_name,
                        reasoning={"effort": "medium"},
                        input=formatted_messages,
                        max_output_tokens=self.max_tokens
                    )
                    content = response.output_text

                else:
                    call_kwargs = {
                        "model": self.model_name,
                        "messages": formatted_messages,
                        "temperature": self.temperature,
                        "max_completion_tokens": self.max_tokens,
                        "top_p": self.top_p,
                        "stop": stop,
                    }
                    while True:
                        try:
                            response = self.client.chat.completions.create(**call_kwargs)
                            break
                        except BadRequestError as e:
                            err = e.body or {}
                            if e.status_code == 400 and err.get("code") == "unsupported_parameter":
                                param = err.get("param")
                                if param and param in call_kwargs:
                                    logger.warning(f"Dropping unsupported param '{param}' for {self.model_name}")
                                    del call_kwargs[param]
                                    continue
                            raise
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
        messages: Union[List[BaseMessage], Dict[str, Dict]], 
        **kwargs: Any
    ) -> Union[AIMessage, Dict[str, str]]:
        """
        Invoke the model with the given messages.
        
        Args:
            messages (Union[List[BaseMessage], Dict[str, Dict]]): Input messages.
            
        Returns:
            Either singleton or batch AI message with the model response.
        """
        try:
            # Generate single or multiple model responses
            result = self._generate(messages, **kwargs)

            # Check for run type and return result accordingly
            return result.generations[0].message if isinstance(result, ChatResult) else result
        except:
            raise TypeError("Unexpected return type from model generation")


# =========================================================================== #
#                             OpenAI Chat Pipeline                            #
# =========================================================================== #
class _ChatOpenAI(BaseOpenAI):
    def __init__(self, **kwargs):
        kwargs["provider"] = "openai"
        super().__init__(**kwargs)


# =========================================================================== #
#                              xAI Chat Pipeline                              #
# =========================================================================== #
class _ChatXAI(BaseOpenAI):
    def __init__(self, **kwargs):
        kwargs["provider"] = "xai"
        kwargs["api_key"] = os.getenv("XAI_API_KEY")
        kwargs["base_url"] = "https://api.x.ai/v1"
        super().__init__(**kwargs)


# =========================================================================== #
#                            DeepSeek Chat Pipeline                           #
# =========================================================================== #
class _ChatDeepSeek(BaseOpenAI):
    def __init__(self, **kwargs):
        kwargs["provider"] = "deepseek"
        kwargs["api_key"] = os.getenv("DEEPSEEK_API_KEY")
        kwargs["base_url"] = "https://api.deepseek.com"
        super().__init__(**kwargs)


# =========================================================================== #
#                              Z.AI Chat Pipeline                             #
# =========================================================================== #
class _ChatZAI(BaseOpenAI):
    def __init__(self, **kwargs):
        kwargs["provider"] = "zai"
        kwargs["api_key"] = os.getenv("ZAI_API_KEY")
        kwargs["base_url"] = "https://api.z.ai/api/paas/v4/"
        super().__init__(**kwargs)