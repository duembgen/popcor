Lifters
=======

.. contents::
   :depth: 1
   :local:


Some general notes on terminology: 

- *theta* is the original state variable.
- *x* is the lifted state variable.
- *A* are equality constraints
- *B* are inequality constraints
- *var_dict* refers to the dictionary of variable name - variable size pairs.


StateLifter
-----------

.. autoclass:: popr.lifters.StateLifter
   :members: get_x, get_Q, sample_theta, sample_parameters, get_A_known, get_B_known
   :show-inheritance:

StereoLifter
------------

.. autoclass:: popr.lifters.StereoLifter
   :members:
   :show-inheritance:

RobustPoseLifter
------------------

.. autoclass:: popr.lifters.RobustPoseLifter
   :members:
   :show-inheritance:

RangeOnlyLifter
---------------

.. autoclass:: popr.lifters.RangeOnlyLifter
   :members:
   :show-inheritance:
