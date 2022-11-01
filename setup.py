from setuptools import setup, find_packages

setup(name='fb_project',
      version='1.0',
      packages=find_packages(),
      install_requires=[
            'pandas',
            'matplotlib',
            'torch',
            'transformers',
            'sklearn',
            'torchvision',
            'tensorboard',
            'seaborn',
            'Pillow',
            'pickle5',
            'fastapi',
            'uvicorn',
            'pydantic',
            'boto3',
            'python-multipart',
            ])