import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import checks
from utils import extracts
import rewards

class TestFormattingChecks(unittest.TestCase):
    
    def test_single_thinking_block(self):
        # Perfect case
        self.assertTrue(checks.check_single_thinking_block("<think>Thinking...</think>"))
        # Double tags (should fail)
        self.assertFalse(checks.check_single_thinking_block("<think>One</think><think>Two</think>"))
        # Missing tags
        self.assertFalse(checks.check_single_thinking_block("Just some text"))

    def test_single_answer_block(self):
        # Perfect case
        self.assertTrue(checks.check_single_answer_block("<answer>42</answer>"))
        # Double tags
        self.assertFalse(checks.check_single_answer_block("<answer>42</answer> Wait, <answer>43</answer>"))

    def test_no_preamble(self):
        # Perfect case
        self.assertTrue(checks.check_no_text_before_think("<think>Thinking...</think>"))
        # Allowed whitespace/newlines
        self.assertTrue(checks.check_no_text_before_think("   \n\n <think>Thinking...</think>"))
        # Forbidden text before tag
        self.assertFalse(checks.check_no_text_before_think("Sure, here is the answer: <think>Thinking...</think>"))


class TestCalculateReward(unittest.TestCase):

    def setUp(self):
        # Default weights from our function to calculate expected scores
        self.format_weight = 0.3
        self.correctness_weight = 0.7
        self.ground_truth = "136"

    def test_perfect_output(self):
        text = "<think>Calculating...</think><answer>136</answer>"
        expected_reward = self.format_weight + self.correctness_weight # 1.0
        
        score = rewards.calculate_reward(text, self.ground_truth)
        self.assertAlmostEqual(score, expected_reward)

    def test_format_perfect_wrong_answer(self):
        text = "<think>Calculating...</think><answer>999</answer>"
        # Should get full format points, plus the 0.1 penalty multiplier for a wrong answer
        expected_reward = self.format_weight + (self.correctness_weight * 0.1)
        
        score = rewards.calculate_reward(text, self.ground_truth)
        self.assertAlmostEqual(score, expected_reward)

    def test_preamble_penalty(self):
        text = "Let's solve this. <think>Calculating...</think><answer>136</answer>"
        # Fails the no_preamble check, so it misses the full format reward.
        # It should still trigger the partial answer reward (0.5 * format_weight)
        # Plus full correctness points.
        expected_reward = (self.format_weight * 0.5) + self.correctness_weight
        
        score = rewards.calculate_reward(text, self.ground_truth)
        self.assertAlmostEqual(score, expected_reward)

    def test_multiple_thinking_blocks_penalty(self):
        text = "<think>Step 1</think><think>Step 2</think><answer>136</answer>"
        # Fails single thinking block. Only gets partial answer format points + correctness.
        expected_reward = (self.format_weight * 0.5) + self.correctness_weight
        
        score = rewards.calculate_reward(text, self.ground_truth)
        self.assertAlmostEqual(score, expected_reward)

if __name__ == "__main__":
    unittest.main()