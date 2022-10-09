from setuptools import setup, find_packages

setup(name='fb_project',
      version='1.0',
      packages=find_packages(),
      install_requires=[
            'pandas',
            'psycopg2',
            'plotly',
            'missingno',
            'sklearn',
            ])