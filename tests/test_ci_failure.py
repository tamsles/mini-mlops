def test_ci_intentional_failure():
    expected = 42
    actual = 0

    assert actual == expected, "Intentional CI regression"
