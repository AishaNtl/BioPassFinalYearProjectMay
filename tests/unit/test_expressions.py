def test_expression_list():
    from biopass.add_faces3 import expressions
    assert isinstance(expressions, dict), "Expressions should be a dict"
    assert "happy" in expressions, "Missing 'happy' expression"