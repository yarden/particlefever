##
## particlefever setup
##
from setuptools import setup

## Definition of the current version
PARTFEVER_VERSION = "0.1"

## Generate a __version__ attribute
## for the module
with open("./particlefever/__init__.py", "w") as version_out:
      version_out.write("__version__ = \"%s\"\n" %(PARTFEVER_VERSION))

long_description = open("README.md").read()

setup(name = 'particlefever',
      version = PARTFEVER_VERSION,
      description = "Particle filtering Monte Carlo inference for dynamic models",
      long_description = long_description,
      author = 'Yarden Katz',
      author_email = 'yarden@hms.harvard.edu',
      maintainer = 'Yarden Katz',
      maintainer_email = 'yarden@hms.harvard.edu',
      packages = ['particlefever'],
      platforms = 'ALL',
      keywords = ['science', 'bayesian', 'inference', 'mcmc',
                  'markov-models', 'probabilistic-modeling', 'time-series',
                  'particle-filtering', 'monte-carlo'],
      classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        ]
      )

