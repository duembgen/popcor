# How to contribute to POPR

We welcome any contributions to POPR! Contributions are done through pull requests. 

## General guidelines

Please try to the best of your abilities to:

- use black and isort for formatting your code
- provide at least minimal documentation
- tests for core functionalities.
- add your information to the CHANGELOG.

## Adding a new lifter class

You can start with the [ExampleLifter](../popcor/examples/ExampleLifter.py) skeleton, and feel free to add more functionalities depending on the nature of the problem. You can also consider adding a new base class similar to [RobustPoseLifter](../popcor/base_lifters/RobustPoseLifter.py) or [StereoLifter](../popcor/base_lifters/StereoLifter.py) if you want to create multiple new lifters that all share similar functionalities.

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

### Setting up mosek license on server

This has already been done -- just keeping track of the process here to make it is easy to redo this in the future. TAken from [here](https://docs.mosek.com/11.0/faq.pdf), page 11. 

1. Go to Settings -> Security -> Secrets and variables -> Actions ([direct link](https://github.com/duembgen/popcor/settings/secrets/actions))
2. Create secret called MSK_LICENSE with content 

```
START_LICENSE\n
FEATURE PTS ....
...
... here copy the text of the "FEATURE" sections in the license ...
...
... ... 5FE1 5DBC"
END_LICENSE\n
```

3. Add the following to the workflow file: 
```
- name: Setup MOSEK & Run Tests
    env:
      MOSEKLM_LICENSE_FILE: ${{ secrets.MSK_LICENSE }}
    run: |
      pytest -sv
```

### Testing Github actions locally

It can be super useful to run github actions locally for debugging! 
I did this by using [this](https://nektosact.com/installation/index.html) tool. Basically, it came down to: 


- Installing docker
- From this repo, running: 
``` 
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```
- Also from this repo, running: 
```
sudo ./bin/act 
```
