from setuptools import setup, find_packages

setup(
    name="cpsc483_final_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pytorch-lightning",
        "torch-geometric",
        "pandas",
        "click"
    ],
)