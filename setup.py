from setuptools import setup, find_packages

setup(
    name="gnn-linkpred",
    version="0.1.0",
    description="GNN-based link prediction on a collaboration network",
    author="Ivna Cristine Brasileiro Valença de Oliveira",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
        "scikit-learn>=1.1",
        "matplotlib>=3.6",
        "torch>=2.0",
        "torch-geometric>=2.3",
        "tqdm>=4.64",
    ],
    python_requires=">=3.9",
)
