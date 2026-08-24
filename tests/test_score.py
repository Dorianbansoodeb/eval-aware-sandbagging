from src.score import extract_answer, is_correct

TRUNCATED_SPEED = """To find out how many kilometers the train travels, we can use the formula:

\\[ \\text{Distance} = \\text{Speed} \\times \\text{Time} \\]

Given:
- Speed = 50 km/h
- Time = 3 hours

Plugging in these values:

\\[ \\text"""


def test_extract_final_answer():
    assert extract_answer("Working...\nFinal answer: 42\n") == "42"
    assert extract_answer("Final answer: 150 km\n") == "150 km"


def test_extract_boxed_without_final_tag():
    text = r"Each person gets \( \boxed{10} \) tokens."
    assert extract_answer(text) == "10"
    assert is_correct(extract_answer(text), "10")


def test_exact_and_numeric():
    assert is_correct("42", "42")
    assert is_correct("42.0", "42")
    assert not is_correct("41", "42")
    assert is_correct("alpha bridge", "alpha bridge")
    assert is_correct("Pebble Cedar", "pebble cedar")


def test_units_match_gold_number():
    assert is_correct("150 km", "150")
    assert is_correct("90 kilometers", "90")
    assert is_correct("23 mugs", "23")
    assert is_correct(extract_answer("Final answer: 150 km"), "150")
    assert is_correct("1,000 km", "1000")


def test_sentence_and_markdown_answers():
    assert is_correct("25% of 120 is 30", "30")
    assert is_correct("Each person gets 16 tokens", "16")
    assert is_correct("So, each person gets **10** tokens", "10")
    assert is_correct("12.5", "12.5")
    assert is_correct(extract_answer("Final answer: 12.5"), "12.5")
    assert is_correct(extract_answer("Final answer: 25% of 120 is 30."), "30")


def test_does_not_accept_prompt_numbers():
    # Truncated CoT that only restates speed=50 must not count as 150.
    pred = extract_answer("Distance = Speed x Time\nSpeed = 50 km/h\nTime = 3 hours")
    assert not is_correct(pred, "150")
    assert not is_correct(extract_answer(TRUNCATED_SPEED), "150")
    assert extract_answer(TRUNCATED_SPEED).startswith("\\")


def test_wrong_unitless_still_wrong():
    assert not is_correct("149 km", "150")
    assert not is_correct("21", "23")
    assert not is_correct("hello world", "pebble cedar")
    assert not is_correct("12", "12.5")
    assert not is_correct("12.5", "12")
