.. particlefever documentation master file, created by
   sphinx-quickstart on Sat Feb 27 16:01:55 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

particlefever documentation
===========================

Particle filters
--------------

Testing the code
********************************

Types of built-in particle filters
********************************

1. Hidden Markov Model
2. (Switching) Autoregressive hidden Markov Model


Making a new particle filter
********************************

To make a new particle filter for a model, classes that define the particle and its prior are needed, as well as functions that deal with the initialization, transition sampling and weighing by data of
particles. For example, for the `:class:bayes_hmm.DiscreteBayesHMM`, the following are defined:

1. Two classes, :class:`particlefever.bayes_hmm.Particle` and ``ParticlePrior`` have to be defined
   for the model.
2. The following functions need to be defined:
     - ``pf_prior``: returns a ``ParticlePrior`` object
     - ``pf_trans_sample``: returns a sampled transition to a new latent state, given a previous ``Particle`` instance an a``ParticlePrior`` object.
     - ``pf_observe``: takes a data point, a particle instance and a particle prior instance and returns a weight for the particle in light of the data point.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

