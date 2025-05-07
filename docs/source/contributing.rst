Contributing
============

We encourage people to contribute their own Lifter classes or other functionalities to the repository. 
Please follow the following guidelines as much as possible.

General guidelines
------------------
Please try to the best of your abilities to:

- use black and isort for formatting your code
- provide at least minimal documentation
- tests for core functionalities.
- add your information to the CHANGELOG.

Adding a new lifter class
-------------------------

You can start with the :ref:`ExampleLifter` sceleton, and feel free to add more functionalities depending on the nature of the problem. You can also consider adding a new base class similar to :ref:`RobustPoseLifter` or :ref:`StereoLifter` if you want to create multiple new lifters that all share similar functionalities. 

Adding new functionalities
--------------------------

We welcome new functionalities / solvers / tools that facilitate problem formulation as well. If you add something general please make sure it is tested on all examples, as done for example in *tests/test_autotight.py*.
