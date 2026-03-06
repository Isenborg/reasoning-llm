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
        # Updated to match the defaults of your new function
        self.format_weight = 0.25
        self.correctness_weight = 0.75
        self.ground_truth = "136"

    def test_perfect_output(self):
        text = "<think>Calculating...</think><answer>136</answer>"
        # Format: 0.08 (think) + 0.08 (answer) + 0.04 (no preamble) = 0.20
        # Correctness: +0.75
        expected_reward = 0.20 + 0.75 # 0.95
        
        score = rewards.calculate_reward(text, self.ground_truth)
        self.assertAlmostEqual(score, expected_reward)

    def test_format_perfect_wrong_answer(self):
        text = "<think>Calculating...</think><answer>999</answer>"
        # Format: 0.20
        # Correctness penalty: - (0.75 * 0.05) = -0.0375
        expected_reward = 0.20 - (self.correctness_weight * 0.05) # 0.1625
        
        score = rewards.calculate_reward(text, self.ground_truth)
        self.assertAlmostEqual(score, expected_reward)

    def test_preamble_penalty(self):
        text = "Let's solve this. <think>Calculating...</think><answer>136</answer>"
        # Format: 0.08 (think) + 0.08 (answer) + 0 (preamble failed) = 0.16
        # Correctness: +0.75
        expected_reward = 0.16 + 0.75 # 0.91
        
        score = rewards.calculate_reward(text, self.ground_truth)
        self.assertAlmostEqual(score, expected_reward)

    def test_multiple_thinking_blocks_penalty(self):
        text = "<think>Step 1</think><think>Step 2</think><answer>136</answer>"
        # Format: 0 (think failed) + 0.08 (answer) + 0.04 (no preamble) = 0.12
        # Correctness: +0.75
        expected_reward = 0.12 + 0.75 # 0.87
        
        score = rewards.calculate_reward(text, self.ground_truth)
        self.assertAlmostEqual(score, expected_reward)

    def test_answer_leakage_penalty(self):
        # The model puts the answer "136" directly inside the think block
        text = "<think>The answer is 136.</think><answer>136</answer>"
        # Format: 0.20
        # Correctness: 0.75
        # Leakage penalty: -0.10
        expected_reward = 0.20 + 0.75 - 0.10 # 0.85
        
        score = rewards.calculate_reward(text, self.ground_truth)
        self.assertAlmostEqual(score, expected_reward)

if __name__ == "__main__":
    unittest.main()