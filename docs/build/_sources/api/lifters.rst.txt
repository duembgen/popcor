Base Lifters
============

.. contents::
   :depth: 1
   :local:

Overview
--------

There are a couple of lifters that serve as a basis for new lifters. 
The most basic one is :ref:`StateLifter`, which is generally the best starting point. 

For specific problems, there are a couple of abstract classes that were developed to ease
the development. In particular, we have:  

- :ref:`RobustPoseLifter` 
- :ref:`StereoLifter` 
- :ref:`PolyLifter`

Basics
------

Below are some general notes about terminology that may be useful in understanding the code and building your own lifters.

- *theta* is the original (low-dimensional) state variable.
- *x* is the lifted (higher-dimensional) state variable.
- *A* are equality constraints
- *B* are inequality constraints
- *var_dict* refers to the dictionary of variable name - variable size pairs.
- *param_dict* is used to factor out parameters when creating templates. It also comes in name - variable size pairs.

StateLifter
-----------

.. autoclass:: popr.base_lifters.StateLifter
   :members: get_x, get_theta, sample_theta, sample_parameters, get_Q, get_A_known, get_B_known, local_solver, get_cost, get_error
   :show-inheritance:

StereoLifter
------------

.. autoclass:: popr.base_lifters.StereoLifter
   :show-inheritance:
   :undoc-members:

RobustPoseLifter
------------------

.. autoclass:: popr.base_lifters.RobustPoseLifter
   :show-inheritance:
   :undoc-members:

PolyLifter
----------

.. autoclass:: popr.base_lifters.PolyLifter
   :show-inheritance:
   :undoc-members: