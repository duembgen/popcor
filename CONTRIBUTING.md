# How to contribute to POPR

We welcome any contributions to POPR! Contributions are done through pull requests. 

## General guidelines

Please try to the best of your abilities to:

- use black and isort for formatting your code
- provide at least minimal documentation
- tests for core functionalities.
- add your information to the CHANGELOG.

## Adding a new lifter class

You can start with the [ExampleLifter](../popr/examples/ExampleLifter.py) skeleton, and feel free to add more functionalities depending on the nature of the problem. You can also consider adding a new base class similar to [RobustPoseLifter](../popr/base_lifters/RobustPoseLifter.py) or [StereoLifter](../popr/base_lifters/StereoLifter.py) if you want to create multiple new lifters that all share similar functionalities.

## Adding new functionalities

We welcome new functionalities / solvers / tools that facilitate problem formulation as well. If you add something general please make sure it is tested on all examples, as done for example in *tests/test_autotight.py*.

## Resources

Below are a couple of guidelines and useful resources. This is a living document aimed to contain a list of useful resources for developers. Feel free to extend it. 

### Testing

Added functionality should include passing unit tests. Please follow the available examples in the `tests` folder. 

It is also encouraged that added functionality is added as testable code to the documentation. To do that, create `.. doctest:` blocks and run 
```
make doctest
```
to make sure there are no errors. There is also the possibility to just use literal includes from test files inside the documentation. See [docs/source/quickstart.rst](./docs/source/quickstart.rst) for an example. 
