import pytest

from rewards import rewards


def test_format_and_correct_answer():
    text = "<think>2+2=4</think><answer>4</answer>"
    # default weights give full reward when both format and correctness are
    # satisfied with a matching ground truth
    assert rewards.calculate_reward(text, ground_truth="4") == pytest.approx(1.0)


def test_wrong_answer_penalty():
    text = "<think>2+2=4</think><answer>5</answer>"
    # format counts for 0.3, wrong answer adds 0.1*0.7 = 0.07
    expected = 0.3 + 0.07
    assert rewards.calculate_reward(text, ground_truth="4") == pytest.approx(expected)


def test_no_tags_returns_zero():
    text = "just some text without tags"
    assert rewards.calculate_reward(text) == 0.0


def test_partial_format_answer_only():
    text = "some reasoning<answer>4</answer>"
    # half of format weight (0.3 * 0.5)
    assert rewards.calculate_reward(text) == pytest.approx(0.15)


def test_partial_format_think_only():
    text = "<think>I am thinking</think>"
    assert rewards.calculate_reward(text) == pytest.approx(0.15)


def test_no_ground_truth_consumes_only_format():
    text = "<think>2+2=4</think><answer>4</answer>"
    # without ground truth only the format component is counted
    assert rewards.calculate_reward(text, ground_truth=None) == pytest.approx(0.3)


def test_compute_reward_wrapper():
    text = "<think>foo</think><answer>bar</answer>"
    # compute_reward should mirror calculate_reward exactly
    assert rewards.compute_reward(text, ground_truth="bar") == rewards.calculate_reward(
        text, ground_truth="bar"
    )


def test_weight_override():
    text = "<think>x</think><answer>y</answer>"
    # override the weights to give equal emphasis and ensure we still cap at 1
    r = rewards.compute_reward(text, ground_truth="y", format_weight=0.5, correctness_weight=0.5)
    assert r == pytest.approx(1.0)
