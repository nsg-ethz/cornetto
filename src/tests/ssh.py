"""
Unit test for checking SSH tunnel connection.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import paramiko
import getpass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# SSH connection details
username = os.getenv("USER_NAME")
hostname = os.getenv("HOST_NAME")


# =========================================================================== #
#                             Basic SSH Connection                            #
# =========================================================================== #
# Create SSH client
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    # Try with SSH key
    print("Trying key authentication...")
    client.connect(hostname=hostname, username=username)
    print("Connection successful!")
    
    # Execute a simple command
    stdin, stdout, stderr = client.exec_command("echo 'Hello from remote'")
    print(stdout.read().decode())
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {str(e)}")
    
    # Try with password
    try:
        password = getpass.getpass("Enter password: ")
        client.connect(hostname=hostname, username=username, password=password)
        print("Password connection successful!")
        
        # Execute a simple command
        stdin, stdout, stderr = client.exec_command("echo 'Hello from remote'")
        print(stdout.read().decode())
    except Exception as e:
        print(f"Password auth error: {type(e).__name__}: {str(e)}")

finally:
    client.close()