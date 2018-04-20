from setuptools import setup

setup(name='EpanetWrapper',
      version='0.1',
      description='A python wrapper used to ease simulations and analytics',
      url='https://github.com/cccristi07/EpanetWrapper',
      author='Cristian Cazan',
      author_email='cccristi07@gmail.com',
      license='MIT',
      packages=['ENWrapper'],
      zip_safe=False,
      install_requires=[
          "numpy",
          "matplotlib",
          "epanettools",
          'pandas',
          'plotly'
      ]

      )
