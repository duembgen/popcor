import inspect

from popcor.base_lifters import StateLifter


def assert_signature_match(base_cls, derived_cls, method_name):
    base_sig = inspect.signature(getattr(base_cls, method_name))
    derived_sig = inspect.signature(getattr(derived_cls, method_name))

    base_params = [p.name for p in base_sig.parameters.values()]
    derived_params = [p.name for p in derived_sig.parameters.values()]

    # make sure that the only arguments of base_params that are not in derived_params are args, kwargs.
    try:
        base_params_not_in_derived = set(base_params).difference(derived_params)
        assert (
            len(base_params_not_in_derived.difference({"args", "kwargs"})) == 0
        ), f"{derived_cls.__name__}: {method_name}: \n  {base_params} \n  {derived_params}"
    except AssertionError as e:
        print(e)
        raise


if __name__ == "__main__":
    import popcor.examples

    for Cls in popcor.examples.__all__:
        for function in [
            "get_cost",
            "local_solver",
            "get_x",
            "get_Q",
            "get_Q_from_y",
            "get_A_known",
            "get_error",
            "sample_theta",
            "sample_parameters",
        ]:
            assert_signature_match(StateLifter, Cls, function)
