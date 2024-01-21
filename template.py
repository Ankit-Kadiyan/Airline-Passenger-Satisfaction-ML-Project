import os
from pathlib import Path # To represent and manipulate file system paths
import logging

logging.basicConfig(level=logging.INFO)

project_name="ML_Project_1"

list_of_files=[
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/componenets/__init__.py",
    f"src/{project_name}/componenets/data_ingestion.py",
    f"src/{project_name}/componenets/data_transformation.py",
    f"src/{project_name}/componenets/model_trainer.py",
    f"src/{project_name}/componenets/model_monitering.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True) # To create a Folder
        logging.info(f"Creating Directory:{filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating empth File: {filepath}")

    else:
        logging.info(f"{filename} already exists")