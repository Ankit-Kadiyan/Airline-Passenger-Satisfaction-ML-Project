import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"    # Format of the Log message File name
log_path=os.path.join(os.getcwd(),"logs",LOG_FILE)                  # Create a file path by combining Current working directory,logs folder and File name
os.makedirs(log_path, exist_ok=True)                                # Creating the Log Folder

LOG_FILE_PATH=os.path.join(log_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Format of the Log message
    level=logging.INFO,
)