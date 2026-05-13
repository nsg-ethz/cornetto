"""
This block establishes an implementation for HuggingFace models both local and API-based variants.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import os
import logging
from typing import Any, List, Optional, Dict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_community.llms import HuggingFaceHub

from src.modules.chat.base import BaseChat

# Load environment variables from .env file
load_dotenv()

# Set logger
logger = logging.getLogger(__name__)


# =========================================================================== #
#                             GPU Allocation Class                            #
# =========================================================================== #
class GPUAllocation:
    """
    A PyTorch GPU allocation class for loading model on vLLM or CUDA engines.
    """
    def __init__(self, num_gpus: int) -> None:
        # Simple sequential GPU selection
        selected_gpus = [str(i) for i in range(num_gpus)]
        
        # Set environment variable
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpus)
        logger.info(f"Set CUDA_VISIBLE_DEVICES to: {selected_gpus}")


# =========================================================================== #
#                          HuggingFace Chat Pipeline                          #
# =========================================================================== #
class _ChatHuggingFace(BaseChat):
    """
    Chat model implementation for HuggingFace-family models.
    """
    # HuggingFace specific config
    use_api: bool = True
    api_token: str = None
    use_4_bit: bool = False
    use_8_bit: bool = False
    use_vllm: bool = False
    num_gpus: int = 2
    max_model_len: int = 128_000

    model: Optional[Any] = None
    tokenizer: Optional[Any] = None

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the HuggingFace chat model configs.

        Args:
            model_name (str): Name of HuggingFace model.
                Defaults to None.
            max_tokens (int): Maximum number of tokens to generate.
                Defaults to 256.
            temperature (float): Temperature for generation.
                Defaults to 0.0.
            top_p (float): Top-p sampling parameter.
                Defaults to 1.0    
            use_4_bit (bool): Condition to use 4-bit quantization.
                Defaults to False.
            use_8_bit (bool): Condition to use 8-bit quantization.
                Defaults to False.
            use_api (bool): Condition on using the HuggingFace API.
                Defaults to True.
            use_vllm (bool): Condition on using vLLM for fast inference.
                Defaults to False.
            num_gpus (int): Number of GPUs to use for inference.
                Defaults to 2.
            max_model_len (int): Maximum sequence length supported by model.
                Defaults to 70_000.
            api_token (str): API Token for HuggingFace.
                Defaults to None.
            model (Optional[Any]): HuggingFace model instance.
                Defaults to None.
            tokenizer (Optional[Any]): HuggingFace tokenizer instance.
                Defaults to None.
        """
        super().__init__(**kwargs)

        # Select CUDA devices
        try:
            GPUAllocation(self.num_gpus)
        except Exception as e:
            logger.warning(f"Could not set 'CUDA_VISIBLE_DEVICES': {str(e)}")


        if self.use_api:
            # Extract API token from environment variables        
            self.api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            if not self.api_token:
                raise ValueError("Missing HuggingFace Hub API token. Set HUGGINGFACEHUB_API_TOKEN in your .env file")
        else:
            # Initialize local model if we are not on the API
            self._init_model()

    @property
    def _llm_type(self) -> str:
        """
        Return identifier for model of use.
        """
        return f"huggingface-{self.model_name}"

    def _init_model(self) -> None:
        """
        Initialize HuggingFace models' (local) client.
        """
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            if self.use_vllm:
                self._load_on_vllm()
            else:
                self._load_on_cuda()
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _load_on_vllm(self) -> None:
        """
        Initialize vLLM model for fast inference.
        """
        try:            
            from vllm import LLM
            from transformers import AutoTokenizer

            # vLLM configuration
            vllm_kwargs = {
                "dtype": "bfloat16",
                "max_model_len": self.max_model_len,
                "tensor_parallel_size": self.num_gpus,
                "gpu_memory_utilization": 0.7,
                "disable_custom_all_reduce": True,
            }
            
            # Add quantization if specified
            if self.use_4_bit or self.use_8_bit:
                vllm_kwargs["quantization"] = "bitsandbytes"
            # if self.use_4_bit:
            #     vllm_kwargs["quantization"] = "awq"
            # elif self.use_8_bit:
            #     vllm_kwargs["quantization"] = "fp8"

            # Initialize model
            logger.info("Initializing vLLM model...")
            self.model = LLM(
                self.model_name, 
                **vllm_kwargs,
                trust_remote_code=True,
            )
            
            # Load tokenizer separately for chat template
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"Successfully loaded {self.model_name} with vLLM")            
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"vLLM failed to import: {str(e)}")
            raise

    def _load_on_cuda(self) -> None:
        """
        Initialize standard transformers model.
        """
        # Import PyTorch and transformers after GPU selection
        import torch
        from transformers import (
            AutoTokenizer, 
            AutoModelForCausalLM, 
            BitsAndBytesConfig
        )

        try:
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "sdpa",
                "low_cpu_mem_usage": True,
            }
            
            # Setup quantization
            if self.use_4_bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_storage=torch.bfloat16,
                )
            elif self.use_8_bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
                trust_remote_code=True
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"Successfully loaded {self.model_name} on CUDA")
        except Exception as e:
            logger.error(f"Failed to initialize on CUDA: {str(e)}")
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

    def _api_generate(self, messages: List[BaseMessage]) -> str:
        """""
        Generate response using HuggingFace Hub model via API key.

        Args:
            messages (List[BaseMessage]): List of messages.

        Returns:
            Prompt embedded model call.
        """""        
        # Set the model via HuggingFace API instance
        llm = HuggingFaceHub(
            repo_id=self.model_name,
            huggingfacehub_api_token=self.api_token, 
            task="text-generation",
            model_kwargs={  
                "temperature": self.temperature,
                "max_new_tokens": self.max_tokens,
                "top_p": self.top_p,
                "num_return_sequences": 1,
                "return_full_text": False,
            }
        )

        # Use ConversationLatestMemory for API-based model to only include last-k messages
        prompt_messages = self.memory._prepend_buffer_memory(
            messages, 
            as_prompt=True, 
        )

        logger.info(f"Successfully initialized via HuggingFace API for model: {self._llm_type}")

        # Return final response
        return llm(prompt_messages)
    
    def _local_generate(self, messages: List[BaseMessage]) -> str:
        """
        Generate response using local HuggingFace model.

        Args:
            messages (List[BaseMessage]): List of messages.

        Returns:
            Full decoded model generation.
        """
        # Use ConversationLatestMemory for local model to only include last-k messages
        prompt_messages = self.memory._prepend_buffer_memory(
            messages, 
            as_prompt=False, 
        )
        
        # Format messages with roles
        formatted_messages = self._format_messages(prompt_messages)

        # Convert to prompt string
        prompt = self.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if self.use_vllm:
            from vllm import SamplingParams
            # Set up sampling parameters
            sampling_params = SamplingParams(
                temperature=self.temperature if self.temperature > 0 else 0.0,
                top_p=self.top_p if self.temperature > 0 else 1.0,
                max_tokens=self.max_tokens,
                stop_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else None,
            )

            # Generate
            outputs = self.model.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text.strip()
        else:
            import torch
            # Tokenize inputs
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=50_000
            )

            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]

            # Make model inference
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs, 
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature if self.temperature > 0 else None,
                    top_p=self.top_p if self.temperature > 0 else None,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    disable_compile=True,
                    use_cache=True
                )
                generation = generation[0][input_len:]

            decoded = self.tokenizer.decode(generation, skip_special_tokens=True)
            return decoded
        
    def _clean_response(self, text: str) -> str:
        """
        Clean the response text. For gpt-oss models, parse the harmony
        response format to extract only the final channel output.
        Other HuggingFace models get basic cleanup.
        
        Args:
            text (str): Raw response text from the model.
            
        Returns:
            Cleaned response text.
        """
        if "gpt-oss" in self.model_name.lower():
            try:
                from openai_harmony import (
                    load_harmony_encoding,
                    HarmonyEncodingName,
                    Role,
                )
                encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
 
                # Tokenize the raw output, then parse it
                tokens = encoding.encode(text)
                parsed_messages = encoding.parse_messages_from_completion_tokens(
                    tokens, Role.ASSISTANT
                )
 
                # Extract only the "final" channel (user-facing response)
                final_parts = []
                for msg in parsed_messages:
                    if msg.channel == "final":
                        final_parts.append(msg.content)
 
                return "\n".join(final_parts).strip() if final_parts else text.strip()
            except ImportError:
                logger.warning(
                    "openai-harmony not installed. Install with: pip install openai-harmony. "
                    "Falling back to raw response."
                )
                return text.strip()
            except Exception as e:
                logger.warning(f"Failed to parse harmony response: {str(e)}")
                return text.strip()
 
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
            # Condition on API vs local generation
            response_text = self._api_generate(messages) if self.use_api else self._local_generate(messages)
            
            # Clean up response text
            cleaned_response = self._clean_response(response_text)
            
            # Get model message from response object
            message = AIMessage(content=cleaned_response)
            
            # Return a ChatResult with the generated message
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