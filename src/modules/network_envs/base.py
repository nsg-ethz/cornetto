"""
Abstract class for building the network environment. Specifications of the
environment are added on top of this class (e.g., Config2Spec, Batfish).
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import subprocess
import os
import json
import logging

logger = logging.getLogger(__name__)


# =========================================================================== #
#                           Network Environment Base                          #
# =========================================================================== #
class NetEnv(ABC):
    """
    Abstract base class for all network environments implementations.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize network environment with common parameters.
        """
        pass

    @abstractmethod
    def setup(self, **kwargs) -> bool:
        """
        Prepare connection for the tool of use.
        """
        pass

    @abstractmethod
    def execute(self):
        """
        Compile the tool of use.
        """
        pass

    @abstractmethod
    def cleanup(self):
        """
        Clean up the call.
        """
        pass