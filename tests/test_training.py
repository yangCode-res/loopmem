import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.controller_sft import fit_prompt, target_only_labels
from training.build_controller_rl import reviewed_reward, split_qids
from training.retrospective_annotation import annotate


class TrainingCoreTests(unittest.TestCase):
    def test_sft_masks_only_prompt(self):
        self.assertEqual(target_only_labels(3, [8, 9]), [-100, -100, -100, 8, 9])

    def test_sft_preserves_prefix_and_tail(self):
        prompt, target = fit_prompt(list(range(10)), [20, 21], max_length=7, prefix_tokens=2)
        self.assertEqual(prompt, [0, 1, 7, 8, 9])
        self.assertEqual(target, [20, 21])

    def test_controller_reward_table(self):
        self.assertEqual(reviewed_reward("CONTINUE", "UNKNOWN"), ([1.0, -1.5], 0.0))
        self.assertEqual(reviewed_reward("SWITCH_LOOP", "INCORRECT"), ([-2.0, 2.0], 0.0))
        self.assertEqual(reviewed_reward("SWITCH_LOOP", "CORRECT"), ([-2.0, 2.25], 0.25))

    def test_controller_qid_split_is_disjoint_and_keeps_both_actions(self):
        rows = [
            {"qid": str(i), "target_action": action}
            for i, action in [(1, "CONTINUE"), (2, "SWITCH_LOOP"), (3, "CONTINUE"), (4, "SWITCH_LOOP"), (5, "CONTINUE"), (6, "SWITCH_LOOP")]
        ]
        train, dev = split_qids(rows, seed=13)
        self.assertFalse({r["qid"] for r in train} & {r["qid"] for r in dev})
        self.assertEqual({r["target_action"] for r in train}, {"CONTINUE", "SWITCH_LOOP"})
        self.assertEqual({r["target_action"] for r in dev}, {"CONTINUE", "SWITCH_LOOP"})

    def test_annotation_uses_two_immutable_stages(self):
        calls = []
        def complete(system, payload):
            calls.append((system, json.loads(payload)))
            if len(calls) == 1:
                return {"workstreams": [{"workstream_id": "W001", "loop_description": "Find entity", "completion_test": "Entity identified"}]}
            return {"action_workstream_ids": ["W001"]}
        result = annotate("Identify the entity.", ["search for entity"], complete)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["action_workstream_ids"], ["W001"])
        self.assertEqual(calls[1][1]["fixed_workstreams"][0]["workstream_id"], "W001")
        self.assertEqual(calls[0][1]["question"], "Identify the entity.")
        self.assertEqual(calls[1][1]["question"], "Identify the entity.")

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "PyTorch is optional and unavailable")
    def test_controller_rl_loss_is_finite(self):
        import torch
        from training.train_controller_rl import ppo_loss
        loss, _ = ppo_loss(torch.tensor([[0.2, -0.2]]), torch.log_softmax(torch.tensor([[0.1, -0.1]]), -1),
                           torch.log_softmax(torch.tensor([[0.1, -0.1]]), -1),
                           torch.tensor([[1.0, -1.5]]), torch.ones(1))
        self.assertTrue(torch.isfinite(loss).item())

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "PyTorch is optional and unavailable")
    def test_reranker_loss_includes_policy_kl_and_soft_ce(self):
        import torch
        from training.train_reranker import listwise_ppo_loss
        scores = torch.tensor([0.1, -0.1], requires_grad=True)
        old = torch.log_softmax(torch.tensor([0.0, 0.0]), -1)
        loss, metrics = listwise_ppo_loss(scores, old, torch.tensor([2, 0]), event_weight=1.0,
            policy_temperature=1.0, target_temperature=0.35, clip_range=0.2, kl_coef=0.03, anchor_coef=0.2)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIn("anchor", metrics)
        self.assertIsNotNone(scores.grad)


if __name__ == "__main__":
    unittest.main()
