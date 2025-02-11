from setuptools import setup, find_packages

setup(
    name="space-settlement-sim",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'torch',
        'networkx',
        'scipy',
        'statsmodels',
        'openai',
        'python-dotenv',
        'tqdm'
    ]
)
