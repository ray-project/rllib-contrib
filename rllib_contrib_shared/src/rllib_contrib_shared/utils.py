import numpy as np
import tensorflow as tf
import torch

tf1 = tf.compat.v1


def check(x, y, decimals=5, atol=None, rtol=None, false=False):
    """
    Checks two structures (dict, tuple, list,
    np.array, float, int, etc..) for (almost) numeric identity.
    All numbers in the two structures have to match up to `decimal` digits
    after the floating point. Uses assertions.

    Args:
        x: The value to be compared (to the expectation: `y`). This
            may be a Tensor.
        y: The expected value to be compared to `x`. This must not
            be a tf-Tensor, but may be a tf/torch-Tensor.
        decimals: The number of digits after the floating point up to
            which all numeric values have to match.
        atol: Absolute tolerance of the difference between x and y
            (overrides `decimals` if given).
        rtol: Relative tolerance of the difference between x and y
            (overrides `decimals` if given).
        false: Whether to check that x and y are NOT the same.
    """
    # A dict type.
    if isinstance(x, dict):
        assert isinstance(y, dict), "ERROR: If x is dict, y needs to be a dict as well!"
        y_keys = set(x.keys())
        for key, value in x.items():
            assert key in y, f"ERROR: y does not have x's key='{key}'! y={y}"
            check(value, y[key], decimals=decimals, atol=atol, rtol=rtol, false=false)
            y_keys.remove(key)
        assert not y_keys, "ERROR: y contains keys ({}) that are not in x! y={}".format(
            list(y_keys), y
        )
    # A tuple type.
    elif isinstance(x, (tuple, list)):
        assert isinstance(
            y, (tuple, list)
        ), "ERROR: If x is tuple, y needs to be a tuple as well!"
        assert len(y) == len(
            x
        ), "ERROR: y does not have the same length as x ({} vs {})!".format(
            len(y), len(x)
        )
        for i, value in enumerate(x):
            check(value, y[i], decimals=decimals, atol=atol, rtol=rtol, false=false)
    # Boolean comparison.
    elif isinstance(x, (np.bool_, bool)):
        if false is True:
            assert bool(x) is not bool(y), f"ERROR: x ({x}) is y ({y})!"
        else:
            assert bool(x) is bool(y), f"ERROR: x ({x}) is not y ({y})!"
    # Nones or primitives.
    elif x is None or y is None or isinstance(x, (str, int)):
        if false is True:
            assert x != y, f"ERROR: x ({x}) is the same as y ({y})!"
        else:
            assert x == y, f"ERROR: x ({x}) is not the same as y ({y})!"
    # String/byte comparisons.
    elif (
        hasattr(x, "dtype") and (x.dtype == object or str(x.dtype).startswith("<U"))
    ) or isinstance(x, bytes):
        try:
            np.testing.assert_array_equal(x, y)
            if false is True:
                assert False, f"ERROR: x ({x}) is the same as y ({y})!"
        except AssertionError as e:
            if false is False:
                raise e
    # Everything else (assume numeric or tf/torch.Tensor).
    else:
        # y should never be a Tensor (y=expected value).
        if isinstance(y, (tf1.Tensor, tf1.Variable)):
            # In eager mode, numpyize tensors.
            if tf.executing_eagerly():
                y = y.numpy()
            else:
                raise ValueError(
                    "`y` (expected value) must not be a Tensor. "
                    "Use numpy.ndarray instead"
                )
        if isinstance(x, (tf1.Tensor, tf1.Variable)):
            # In eager mode, numpyize tensors.
            if tf1.executing_eagerly():
                x = x.numpy()
            # Otherwise, use a new tf-session.
            else:
                with tf1.Session() as sess:
                    x = sess.run(x)
                    return check(
                        x, y, decimals=decimals, atol=atol, rtol=rtol, false=false
                    )
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        # Using decimals.
        if atol is None and rtol is None:
            # Assert equality of both values.
            try:
                np.testing.assert_almost_equal(x, y, decimal=decimals)
            # Both values are not equal.
            except AssertionError as e:
                # Raise error in normal case.
                if false is False:
                    raise e
            # Both values are equal.
            else:
                # If false is set -> raise error (not expected to be equal).
                if false is True:
                    assert False, f"ERROR: x ({x}) is the same as y ({y})!"

        # Using atol/rtol.
        else:
            # Provide defaults for either one of atol/rtol.
            if atol is None:
                atol = 0
            if rtol is None:
                rtol = 1e-7
            try:
                np.testing.assert_allclose(x, y, atol=atol, rtol=rtol)
            except AssertionError as e:
                if false is False:
                    raise e
            else:
                if false is True:
                    assert False, f"ERROR: x ({x}) is the same as y ({y})!"
