from setuptools import setup, find_packages

setup(name='fb_project',
      version='1.0',
      packages=find_packages(),
      install_requires=[
            'torch',
            'transformers',
            'torchvision',
            'Pillow',
            'pickle5',
            'fastapi',
            'uvicorn',
            'pydantic',
            'boto3',
            'python-multipart',
            ])