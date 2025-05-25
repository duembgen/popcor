# How to contribute to POPR

We welcome any contributions to POPR! Contributions are done through pull requests. 

Below are a couple of guidelines and useful resources. This is a living document aimed to contain a list of useful resources for developers. Feel free to extend it. 

## Testing

Added functionality should include passing unit tests. Please follow the available examples in the `tests` folder. 

It is also encouraged that added functionality is added as testable code to the documentation. To do that, create `.. doctest:` blocks and run 
```
make doctest
```
to make sure there are no errors. There is also the possibility to just use literal includes from test files inside the documentation. See [docs/source/quickstart.rst](docs/source/quickstart.rst) for an example. 
