"""
Unit test for zero-shot learning pipeline with improved chat logging.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import os
import hydra
import logging
import warnings
import re
import torch
from omegaconf import DictConfig
from langchain_core.messages import HumanMessage, SystemMessage

from src.modules.chat.model_registry import create_chat_model

# Ignore mainly hydra based warnings
warnings.filterwarnings("ignore")


# =========================================================================== #
#                         Basic Chat CLI Main Function                        #
# =========================================================================== #
@hydra.main(config_path="../../configs", config_name="zero_shot")
def main(config: DictConfig):
    """
    Main prompting function with Hydra configuration.

    Args:
        config (DictConfig): Hydra configuration object containing all parameters.
    """
    # Save directory is automatically managed by Hydra
    save_dir = f"{os.getcwd()}/"
    print(f"Saving results to {save_dir}")
        
    # Create model from config using the factory
    model = create_chat_model(config.model.provider, **config.model)
        
    # Create system message
    system_message = SystemMessage(content=config.system_message)
    model.memory.chat_memory.add_message(system_message)
    
    # Clean header for the chat
    print("\n" + "="*60)
    print(f"  Chat Session with {model._llm_type}")
    print("="*60)
    print("\nSystem: " + config.system_message)
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("-"*60 + "\n")
        
    # Interactive chat loop
    while True:
        try:
            # Get user input
            user_input = input("You: ")
            
            # Check for exit
            if user_input.lower() in ["quit", "exit"]:
                print("\nChat session ended. Goodbye!")
                break
            
            # Create a human message
            human_message = HumanMessage(content=user_input)
            
            # Add user message to memory
            model.memory.chat_memory.add_user_message(user_input)
            
            # Paste "thinking" message for local models
            print("Model: Thinking...", end="\r")
            
            # Get model response by explicitly passing the current message
            response = model.invoke([human_message])

            # Add model response to memory
            model.memory.chat_memory.add_ai_message(response.content)
            
            # Clear "thinking" message;
            # And paste model response
            print("\r" + " " * 30 + "\r", end="")                  
            print(f"Model: {response.content}")
            print("-"*60)
            
        except KeyboardInterrupt:
            print("\n\nChat session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Let's continue the conversation.")
            print("-"*60)
    
    # Clear stale cache if any
    model.memory.chat_memory.clear()
    # Clear CUDA memory
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()