from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'

# Function to get list of requirements
def get_requirements(file_path: str) -> List[str]:
    with open(file_path) as file_obj:
        requirements = [req.strip() for req in file_obj.readlines() if req.strip() != '-e.']

    return requirements

setup(
    name='ML_Project_1',
    version='0.0.1',
    author='Ankit Kadiyan',
    author_email='kadiyanankit11@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    )