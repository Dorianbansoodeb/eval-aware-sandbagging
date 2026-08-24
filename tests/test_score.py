from src.score import extract_answer, is_correct


def test_extract_final_answer():
    assert extract_answer("Working...\nFinal answer: 42\n") == "42"
    assert is_correct("42", "42")
    assert is_correct("42.0", "42")
    assert not is_correct("41", "42")
    assert is_correct("alpha bridge", "alpha bridge")
