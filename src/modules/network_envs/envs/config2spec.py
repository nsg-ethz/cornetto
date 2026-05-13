# References Disclaimer:
# We utilize Config2Spec to translate network configurations to specifications. 
# Repo can be found at https://github.com/nsg-ethz/config2spec?tab=readme-ov-file 

# Citation:
#   Birkner, R., Drachsler-Cohen, D., Vanbever, L., & Vechev, M. (2020). 
#   Config2Spec: Mining Network Specifications from Network Configurations. 
#   17th USENIX Symposium on Networked Systems Design and Implementation (NSDI 20) (pp. 969-984). 
#   USENIX Association. https://www.usenix.org/conference/nsdi20/presentation/birkner

"""
Script to establish tunnel to build Config2Spec environment from remote machine. 
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import os 
import logging
import time
import tempfile
import paramiko
from io import StringIO
import os.path

from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from src.modules.network_envs.base import NetEnv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


# =========================================================================== #
#                           Config2Spec Environment                           #
# =========================================================================== #
class Config2Spec(NetEnv):
    """
    Build a tunnel to Config2Spec running on a remote machine and 
    use for translating network configurations into specifications.
    """

    def __init__(
        self,
        use_ssh: bool = True,
        username: str = None,
        hostname: str = None,
        local_port: Optional[str] = None,
        remote_port: Optional[str] = None,
        remote_host: Optional[str] = None,
        c2s_repo_path: str = None,
        backend_path: str = None,
        batfish_path: str = None,
        max_failures: int = 0,
        dense_policies: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize Config2Spec environment.

        Args:
            use_ssh (bool): Whether to use SSH tunnel or not.
                Defaults to True.
            username (str): User name for remote machine login.
                Defaults None.
            hostname (str): Host name of remote machine.
                Defaults to None.
            local_port (Optional[str]): Port to forward locally.
                Defaults to None.
            remote_port (Optional[str]): Remote port where Config2Spec runs.
                Defaults to None.
            remote_host (Optional[str]): Remote machine IP or hostname.
                Defaults to None.
            c2s_repo_path (str): Path to the Config2Spec repository on remote machine.
                Defaults to None.
            backend_path (str): Path to the backend jar on remote machine.
                Defaults to None.
            batfish_path (str): Path to Batfish on remote machine.
                Defaults to None.
            max_failures (int): Failure model used for Config2Spec.
                Defaults to 0.
            dense_policies (bool): Flag to download condensed output.
                Defaults to True.
        """
        super().__init__(**kwargs)

        # Get credentials from interface variables
        config = self._get_vm_credentials()  

        # Set instance variables from .env variables
        self.use_ssh = use_ssh
        self.username = username or config["USER_NAME"]
        self.hostname = hostname or config["HOST_NAME"]
        self.local_port = local_port or config["LOCAL_PORT"]
        self.remote_port = remote_port or config["REMOTE_PORT"]
        self.remote_host = remote_host or config["REMOTE_HOST"]
        
        # Use constructor arguments for paths with sensible defaults
        self.c2s_repo_path = \
            c2s_repo_path or "config2spec"
        self.backend_path = \
            backend_path or os.path.join(
                "batfish_interface",
                "batfish-73946b2f1bdea5f1146e4db4f2586e071da752df",
                "projects",
                "backend",
                "target",
                "backend-bundle-0.36.0.jar"
            )
        self.batfish_path = \
            batfish_path or "~/tmp"

        # SSH key path
        self.ssh_key_path = Path.home() / ".ssh" / "id_rsa"
        
        # Initialize SSH client and SFTP client
        self.ssh_client = None
        self.sftp_client = None

        # Store Config2Spec based variables
        self.max_failures = max_failures
        self.dense_policies = dense_policies

    def __call__(self, **kwargs):
        """
        Make the class callable directly with the expected environment.
        """
        if self.use_ssh:
            self._keep_connection_alive()

        return self.execute(
            local_scenario_path=kwargs.get('local_scenario_path'),
            max_failures=self.max_failures,
            dense_policies=self.dense_policies
        )
    
    def _keep_connection_alive(self):
        """
        Keeping SSH connection alive.
        """
        if not self.ssh_client or not self.ssh_client.get_transport() or not self.ssh_client.get_transport().is_active():
            if not self.setup():
                raise ConnectionError("SSH reconnect failed. Cannot proceed.")

    @staticmethod
    def _get_vm_credentials() -> Dict[str, Any]:
        """
        Pull specifications and credentials for running remote machine.

        Returns:
            A dictionary of remote machine specifications.
        """
        return {
            "USER_NAME": os.getenv("USER_NAME"),
            "HOST_NAME": os.getenv("HOST_NAME"),
            "LOCAL_PORT": os.getenv("LOCAL_PORT"),
            "REMOTE_PORT": os.getenv("REMOTE_PORT"),
            "REMOTE_HOST": os.getenv("REMOTE_HOST"),
        }

    def setup(
            self, 
            retries=3, 
            wait=5
        ) -> bool:
        """
        Establish connection using only password authentication.

        Args:
            retries (int): Number of retries to connect.
                Defaults to 3.
            wait (int): Waiting time between retries.
                Defaults to 5.
        
        Returns:
            Boolean flag indicating success.
        """
        for attempt in range(retries):
            try:
                # Create SSH client
                self.ssh_client = paramiko.SSHClient()
                self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                # Type in password and connect
                password = os.getenv("PASSWORD")
                if not password:
                    import getpass
                    password = getpass.getpass(f"Enter password for {self.username}@{self.hostname}: ")
                
                self.ssh_client.connect(
                    hostname=self.hostname,
                    username=self.username,
                    password=password,
                    allow_agent=False,
                    look_for_keys=False 
                )
                
                # Create SFTP client
                self.sftp_client = self.ssh_client.open_sftp()
                logger.info("SSH connection successful")
                return True
            except Exception as e:
                print(f"Connection failed: {str(e)}")
                time.sleep(wait)
        
        logger.error("SSH setup failed after retries.")
        return False

    def _run_command(
        self, 
        command: str,
        timeout: int = 60,
    ) -> Tuple[int, str, str]:
        """
        Run a command on the remote machine.

        Args:
            command (str): Command to run.
            timeout (int): Timeout duration in seconds.
                Defaults to 60.

        Returns:
            Tuple of (exit_code, stdout, stderr).
        """
        # Sanity check command
        kill_command = "kill -9 $(lsof -t -i :8192)"

        if not self.ssh_client:
            raise ValueError("SSH client not initialized. Call setup() first.")
            
        try:
            logger.info(f"Running remote command: {command}")
            # Kill all stale jobs
            for _ in range(10):
                self.ssh_client.exec_command(kill_command)

            stdin, stdout, stderr = self.ssh_client.exec_command(command)

            # Wait for command to complete with timeout
            start_time = time.time()
            while not stdout.channel.exit_status_ready():
                if time.time() - start_time > timeout:
                    logger.warning(f"Command timed out after {timeout} seconds")
                    
                    # Kill processes on port 8192
                    logger.info("Killing stale processes on port 8192")
                    try:
                        for _ in range(10):
                            self.ssh_client.exec_command(kill_command)
                    except Exception as kill_err:
                        logger.error(f"Failed to run kill command: {kill_err}")
                    
                    # Close and send 124 code
                    stdout.channel.close()
                    return 124, "", f"Command timed out after {timeout} seconds"
                
                # Make a stop
                time.sleep(0.1)

            exit_code = stdout.channel.recv_exit_status()
            stdout_str = stdout.read().decode('utf-8')
            stderr_str = stderr.read().decode('utf-8')

            if exit_code != 0:
                logger.warning(f"Command exited with code {exit_code}: {stderr_str}")

            return exit_code, stdout_str, stderr_str

        except Exception as e:
            logger.error(f"Error running command: {str(e)}")
            return -1, "", str(e)

    def _upload_file(
        self, 
        local_path: str, 
        remote_path: str
    ) -> bool:
        """
        Upload a file to the remote machine.
        
        Args:
            local_path (str): Path to local file.
            remote_path (str): Path where to store file on remote machine.
            
        Returns:
            Boolean flag indicating success.
        """
        if not self.sftp_client:
            raise ValueError("SFTP client not initialized. Call setup() first.")
            
        try:
            logger.info(f"Uploading {local_path} to {remote_path}")
            self.sftp_client.put(local_path, remote_path)
            return True
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            return False
            
    def _upload_directory_contents(
        self, 
        local_dir: str, 
        remote_dir: str
    ) -> bool:
        """
        Upload all files in a directory to the remote machine.
        
        Args:
            local_dir (str): Path to local directory.
            remote_dir (str): Path where to store files on remote machine.
            
        Returns:
            Boolean flag indicating success.
        """
        if not self.sftp_client:
            raise ValueError("SFTP client not initialized. Call setup() first.")
            
        try:
            # Ensure the remote directory exists
            self._run_command(f"mkdir -p {remote_dir}")
            
            # Find all files in the local directory
            all_files = []
            for root, dirs, files in os.walk(local_dir):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    
                    # Get the relative path from the local_dir
                    rel_path = os.path.relpath(local_file_path, local_dir)
                    remote_file_path = os.path.join(remote_dir, rel_path)
                    
                    # Ensure remote subdirectories exist
                    remote_subdir = os.path.dirname(remote_file_path)
                    self._run_command(f"mkdir -p {remote_subdir}")
                    
                    all_files.append((local_file_path, remote_file_path))
            
            # Upload each file
            for local_file, remote_file in all_files:
                if not self._upload_file(local_file, remote_file):
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error uploading directory: {str(e)}")
            return False

    def _download_file(
        self, 
        remote_path: str, 
        local_path: str
    ) -> bool:
        """
        Download a file from the remote machine.
        
        Args:
            remote_path (str): Path to file on remote machine.
            local_path (str): Path where to store file locally.
            
        Returns:
            Boolean indicating success.
        """
        if not self.sftp_client:
            raise ValueError("SFTP client not initialized. Call setup() first.")
            
        try:
            logger.info(f"Downloading {remote_path} to {local_path}")
            self.sftp_client.get(remote_path, local_path)
            return True
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            return False

    def _download_directory_contents(
        self, 
        remote_dir: str, 
        local_dir: str, 
        exclude_patterns: List[str] = None
    ) -> Dict[str, str]:
        """
        Download all files from a remote directory to a local directory.
        
        Args:
            remote_dir (str): Path to directory on remote machine.
            local_dir (str): Path where to store files locally.
            exclude_patterns (List[str], optional): Patterns to exclude from download.
            
        Returns:
            Dictionary mapping filenames to their local paths.
        """
        if not self.ssh_client or not self.sftp_client:
            raise ValueError("SSH/SFTP client not initialized. Call setup() first.")
            
        # Ensure local directory exists
        os.makedirs(local_dir, exist_ok=True)
        
        # Build find command
        find_cmd = f"find {remote_dir} -type f"
        if exclude_patterns:
            for pattern in exclude_patterns:
                find_cmd += f" -not -path '*{pattern}*'"
        
        # Find all files in the remote directory
        exit_code, stdout, stderr = self._run_command(find_cmd)
        if exit_code != 0:
            logger.error(f"Error finding files: {stderr}")
            return {}
        
        file_paths = stdout.strip().split('\n')
        downloaded_files = {}
        
        # Download each file
        for remote_path in file_paths:
            if not remote_path.strip():
                continue
                
            # Get relative path from remote_dir
            rel_path = os.path.relpath(remote_path, remote_dir)
            local_path = os.path.join(local_dir, rel_path)
            
            # Create subdirectories if needed
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download the file
            if self._download_file(remote_path, local_path):
                downloaded_files[rel_path] = local_path
        
        return downloaded_files

    def execute(
        self,
        local_scenario_path: str,
        max_failures: int = None,
        dense_policies: bool = True
    ) -> Dict[str, Any]:
        """
        Execute Config2Spec on remote machine with data from local machine.
        
        Args:
            local_scenario_path (str): Path to scenario data on local machine.
            max_failures (int): Maximum failures to consider (optional).
                Defaults to None.
            dense_policies (bool): Download condensed and cleaned policies only.
                Defaults to True.
        Returns:
            Dictionary containing the extracted network specifications.
        """
        try:
            if not local_scenario_path:
                return {"error": "Missing required scenario_path in input data"}
                
            # Ensure setup is complete
            if not (self.ssh_client 
                    and self.ssh_client.get_transport() 
                    and self.ssh_client.get_transport().is_active()):
                print("SSH connection not active, setting up...")
                if not self.setup():
                    return {"error": "Failed to set up SSH connection"}
            else:
                print("Using existing SSH connection")
                
            # Create a secure temporary directory for our operation
            with tempfile.TemporaryDirectory(prefix="config2spec_") as local_temp_dir:
                # Generate unique name for remote directory
                run_id = int(time.time())
                remote_base_dir = f"scenarios_tmp/input_{run_id}"
                remote_config_dir = f"{remote_base_dir}/configs"

                # Clean any leftover
                self._run_command(f"rm -rf {self.c2s_repo_path}/scenarios_tmp")
                
                # Create remote directories
                exit_code, _, stderr = self._run_command(
                    f"cd {self.c2s_repo_path} && "
                    f"mkdir -p {remote_config_dir}"
                )
                if exit_code != 0:
                    return {"error": f"Failed to create remote directory: {stderr}"}
                
                # Make a stop
                time.sleep(0.1)
                
                # Handle the scenario path
                is_directory = os.path.isdir(local_scenario_path)
                
                if is_directory:
                    # Upload directory contents
                    print(f"Uploading directory contents from {local_scenario_path}")
                    remote_upload_dir = f"{self.c2s_repo_path}/{remote_config_dir}"
                    if not self._upload_directory_contents(local_scenario_path, remote_upload_dir):
                        return {"error": "Failed to upload scenario directory to VM"}
                else:
                    # Check if it is a file
                    if os.path.isfile(local_scenario_path):
                        filename = os.path.basename(local_scenario_path)
                        remote_file_path = f"{self.c2s_repo_path}/{remote_base_dir}/{filename}"
                        
                        # Upload the file
                        if not self._upload_file(local_scenario_path, remote_file_path):
                            return {"error": "Failed to upload scenario file to VM"}                            
                    else:
                        return {"error": f"Path does not exist or is neither a file nor a directory: {local_scenario_path}"}
                
                # Make a stop
                time.sleep(0.1)

                # List files in the remote directory to verify upload
                exit_code, stdout, _ = self._run_command(f"ls -la {self.c2s_repo_path}/{remote_config_dir}")
                print(f"Files in remote directory: {stdout}")
                
                # Run Config2Spec on the remote directory
                c2s_command = (
                    f"cd {self.c2s_repo_path} && "
                    "source c2s_env/bin/activate && "
                    f"python c2s_runner.py \
                    --input_dir {remote_base_dir} \
                    --backend_path {self.backend_path} \
                    --batfish_path {self.batfish_path} \
                    --failure_model {max_failures}"
                )
                
                logger.info(f"Running Config2Spec on VM")
                exit_code, stdout, stderr = self._run_command(c2s_command)
                
                if exit_code != 0:
                    return {"error": f"Config2Spec execution failed: {stderr}"}
                
                # Make a stop
                time.sleep(0.1)
                                
                # Output directory is same as Config2Spec input data location
                remote_output_dir = f"{self.c2s_repo_path}/{remote_base_dir}"

                # Optionally, verify it exists
                exit_code, stdout, _ = self._run_command(f"ls -la {remote_output_dir}")
                if exit_code != 0:
                    return {"error": f"Could not access output directory: {remote_output_dir}"}
                
                # Make a stop
                time.sleep(0.1)

                # Create local output directory
                local_output_dir = os.path.join(local_temp_dir, "c2s_output")
                os.makedirs(local_output_dir, exist_ok=True)

                # Download all output files (excluding the configs directory)
                downloaded_files = self._download_directory_contents(
                    remote_dir=remote_output_dir,
                    local_dir=local_output_dir,
                    exclude_patterns=["/configs/"]
                )

                if not downloaded_files:
                    return {"error": "No output files were found or could be downloaded"}
                
                # Make a stop
                time.sleep(0.1)

                # Process downloaded files
                results = {}
                for rel_path, local_path in downloaded_files.items():
                    filename = os.path.basename(rel_path)

                    # Skip bigger 'policies' file if True
                    if (dense_policies and 
                        filename == "policies.csv"):
                        logger.info(f"Skipping file: {filename}")
                        continue
                    try:
                        with open(local_path, 'r') as f:
                            file_content = f.read()
                            results[filename] = {"content": file_content}
                    except Exception as e:
                        logger.error(f"Error reading file {rel_path}: {str(e)}")

                # Clean up remote directory
                self._run_command(f"rm -rf {self.c2s_repo_path}/{remote_base_dir}")

                # Make a stop
                time.sleep(0.1)

                return results         
        except Exception as e:
            logger.error(f"Error during Config2Spec execution: {str(e)}")
            return {"error": str(e)}

    def cleanup(self) -> None:
        """
        Clean up resources used by the Config2Spec environment.
        """
        try:
            if self.sftp_client:
                self.sftp_client.close()
                self.sftp_client = None
                
            if self.ssh_client:
                self.ssh_client.close()
                self.ssh_client = None
                
            logger.info("SSH connection closed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")