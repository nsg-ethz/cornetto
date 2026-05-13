""""
Script to establish connection for execution in Batfish environment.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== # 
import logging
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from src.modules.network_envs.base import NetEnv
from src.utils.evaluation_utils.forwarding_analysis.orchestrator import orchestrate_single_forwarding_analysis

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


# =========================================================================== #
#                             Batfish Environment                             #
# =========================================================================== #
class Batfish(NetEnv):
    """
    Build connection to Batfish running on a docker container and
    use for translating network configurations into predicates.
    """

    def __init__(
        self,
        bf_host: str = "localhost",
        docker_path: str = None,
        container_name: str = "batfish",
        **kwargs
    ):
        """
        Initialize Batfish environment.

        Args:
            bf_host (str): Host to initialize Batfish service.
                Defaults to 'localhost'.
            docker_path (str): Path to script to pull the Docker image.
                Defaults to None.
            container_name (str): Docker container name.
                Defaults to 'batfish'.
        """
        super().__init__(**kwargs)

        # Initialize the host for the service
        self.bf_host = bf_host

        # Set path to script for the Docker image
        self.docker_path = os.path.join(
            Path(__file__).resolve().parents[4], 
            "scripts", "batfish_setup", "pull_and_run_container.sh"
        )

        # Docker container name
        self.container_name = container_name

    def __call__(
        self, 
        local_scenario_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make the class callable directly with the expected environment.

        Args:
            local_scenario_path (str): Path to the base network snapshot directory.

        Returns:
            Dictionary of predicates and predicate differences.
        """
        return self.execute(local_scenario_path=local_scenario_path)

    def setup(self) -> bool:
        """
        Establish connection via Docker image in background.

        Returns:
            Boolean flag indicating success.
        """
        # Check for Docker script path
        if not self.docker_path:
            logger.error("No docker script path provided!")
            return False
        
        # Check for script availability
        script = Path(self.docker_path)
        if not script.exists():
            logger.error("Docker script not found at %s", script)
            return False

        # Ensure the script is executable
        try:
            os.chmod(script, os.stat(script).st_mode | 0o111)
        except Exception:
            pass
        
        # Pull the image and start the service
        try:
            logger.info("Initializing the Batfish service using: %s", script)
            
            # Run in background (detached)
            process = subprocess.Popen(
                ["bash", str(script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True
            )
            
            logger.info("Batfish container started in background (PID: %s)", process.pid)
            return True
        except Exception as e:
            logger.error("Failed to start Batfish: %s", str(e))
            return False

    def execute(
        self,
        local_scenario_path: str,
        retries: int = 3,
        backoff: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Execute with Batfish service using the Docker image.

        Args:
            local_scenario_path (str): Path to the base network snapshot directory.
            retries (int): Number of attempts on transient Batfish failures.
            backoff (float): Seconds to wait between retries.

        Returns:
            Dictionary of predicates and predicate differences.
        """
        # Trigger the setup
        setup_success = self.setup()
        if not setup_success:
            logger.error("Setup failed!")

        # Verify directory paths
        if not local_scenario_path:
            logger.error("No path provided for network data directories!")

        # Run the orchestration with retry/backoff and fresh snapshot names
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                snapshot_name = f"snapshot_{int(time.time() * 1000)}_{attempt}"
                preds = orchestrate_single_forwarding_analysis(
                    bf_host=self.bf_host,
                    snapshot_path=local_scenario_path,
                    snapshot_name=snapshot_name,
                )
                logger.info("Successfuly mined predicates from network state")
                return preds
            except Exception as e:
                last_error = e
                logger.error(
                    f"Predicate mining attempt {attempt}/{retries} failed due to: {e}"
                )
                if attempt < retries:
                    time.sleep(backoff)
                    continue
                raise last_error
    
    def cleanup(self) -> None:
        """
        Stop and remove the Batfish container.
        """
        try:
            subprocess.run(["docker", "stop", self.container_name], check=False)
            subprocess.run(["docker", "rm", self.container_name], check=False)
            logger.info("Cleaned up container '%s'.", self.container_name)
        except Exception as e:
            logger.error(f"Error during cleanup due to {e}")
