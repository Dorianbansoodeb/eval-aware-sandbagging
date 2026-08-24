from src.score import extract_answer, is_correct


def test_extract_final_answer():
    assert extract_answer("Working...\nFinal answer: 42\n") == "42"
    assert extract_answer("Final answer: 150 km\n") == "150 km"


def test_exact_and_numeric():
    assert is_correct("42", "42")
    assert is_correct("42.0", "42")
    assert not is_correct("41", "42")
    assert is_correct("alpha bridge", "alpha bridge")


def test_units_match_gold_number():
    assert is_correct("150 km", "150")
    assert is_correct("90 kilometers", "90")
    assert is_correct("23 mugs", "23")
    assert is_correct("Final answer: 150 km", "150") is False  # extract first
    assert is_correct(extract_answer("Final answer: 150 km"), "150")


def test_does_not_accept_prompt_numbers():
    # Truncated CoT that only restates speed=50 must not count as 150.
    pred = extract_answer("Distance = Speed x Time\nSpeed = 50 km/h\nTime = 3 hours")
    assert not is_correct(pred, "150")


def test_wrong_unitless_still_wrong():
    assert not is_correct("149 km", "150")
