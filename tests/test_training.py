import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.controller_sft import fit_prompt, target_only_labels, target_token_sum_batchmean_loss
from training.build_controller_rl import build, reviewed_reward, split_qids
from training.retrospective_annotation import INVENTORY_SYSTEM_PROMPT, annotate


class TrainingCoreTests(unittest.TestCase):
    def test_sft_masks_only_prompt(self):
        self.assertEqual(target_only_labels(3, [8, 9]), [-100, -100, -100, 8, 9])

    def test_sft_preserves_prefix_and_tail(self):
        prompt, target = fit_prompt(list(range(10)), [20, 21], max_length=7, prefix_tokens=2)
        self.assertEqual(prompt, [0, 1, 7, 8, 9])
        self.assertEqual(target, [20, 21])

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "PyTorch is optional and unavailable")
    def test_sft_loss_sums_target_tokens_per_example_then_batchmeans(self):
        import torch
        import torch.nn.functional as F

        logits = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            requires_grad=True,
        )
        # Example 1 has one target token; example 2 has two. The final -100
        # on example 2 represents padding and must not add to its loss.
        labels = torch.tensor([[-100, 1, -100], [-100, 0, 2]])
        loss = target_token_sum_batchmean_loss(logits, labels)

        one_token_ce = F.cross_entropy(logits[0, 0].view(1, -1), labels[0, 1].view(1))
        two_token_ce = (
            F.cross_entropy(logits[1, 0].view(1, -1), labels[1, 1].view(1))
            + F.cross_entropy(logits[1, 1].view(1, -1), labels[1, 2].view(1))
        )
        torch.testing.assert_close(loss, (one_token_ce + two_token_ce) / 2)
        self.assertFalse(torch.isclose(loss, F.cross_entropy(
            logits[:, :-1].reshape(-1, 3), labels[:, 1:].reshape(-1), ignore_index=-100
        )))

        loss.backward()
        self.assertIsNotNone(logits.grad)
        # Gradients occur only at positions predicting target tokens: prompt
        # predictions and positions whose next label is masked stay untouched.
        self.assertGreater(logits.grad[0, 0].abs().sum().item(), 0.0)
        self.assertEqual(logits.grad[0, 1:].abs().sum().item(), 0.0)
        self.assertGreater(logits.grad[1, :2].abs().sum().item(), 0.0)
        self.assertEqual(logits.grad[1, 2].abs().sum().item(), 0.0)

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

    def test_reviewed_samples_build_rewards_and_question_disjoint_files(self):
        from training.build_controller_rl import sha

        rows = [
            {"sample_id": f"sample-{qid}-{action}", "qid": str(qid),
             "messages": [{"role": "system", "content": "controller"},
                          {"role": "user", "content": "observed decision-time input"}],
             "target_action": action, "terminal_label": "CORRECT"}
            for qid in range(6) for action in ("CONTINUE", "SWITCH_LOOP")
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "reviewed.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            original = source.read_bytes()
            output = root / "prepared"
            manifest = build(source, output)
            train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
            dev = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()]
            self.assertFalse({row["qid"] for row in train} & {row["qid"] for row in dev})
            self.assertAlmostEqual(sum(row["weight"] for row in train) / len(train), 1.0)
            for row in train + dev:
                self.assertEqual(row["reward"], [1.0, -1.5] if row["target_action"] == "CONTINUE" else [-2.0, 2.25])
            self.assertEqual(manifest["files"]["train.jsonl"], sha(output / "train.jsonl"))
            self.assertEqual(source.read_bytes(), original)
            if __import__("importlib").util.find_spec("torch"):
                from training.train_controller_rl import read_data
                actual_train, actual_dev, _ = read_data(output)
                self.assertEqual((actual_train, actual_dev), (train, dev))
                with (output / "train.jsonl").open("a") as handle:
                    handle.write("\n")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    read_data(output)

    def test_annotation_candidate_agreement_skips_adjudication(self):
        calls = []
        def complete(system, payload):
            data = json.loads(payload)
            calls.append((system, data))
            if system == INVENTORY_SYSTEM_PROMPT:
                return {"workstreams": [{"workstream_id": "W001", "loop_description": "Find entity", "completion_test": "Entity identified"}]}
            return {"action_workstream_ids": ["W001"]}
        result = annotate("Identify the entity.", ["search for entity"], complete, candidate_count=2)
        self.assertEqual(len(calls), 4)
        self.assertEqual(result["action_workstream_ids"], ["W001"])
        self.assertEqual(calls[2][1]["fixed_workstreams"][0]["workstream_id"], "W001")
        self.assertEqual(calls[0][1]["question"], "Identify the entity.")
        self.assertEqual(calls[2][1]["question"], "Identify the entity.")
        self.assertTrue(all("candidate_inventories" not in payload for _, payload in calls))
        self.assertTrue(all("assignment_candidates" not in payload for _, payload in calls))

    def test_annotation_adjudicates_disagreement_and_renumbers_by_first_use(self):
        from training.retrospective_annotation import INVENTORY_SYSTEM_PROMPT

        inventory_calls = 0
        assignment_calls = 0
        assignment_payloads = []

        def complete(system, payload):
            nonlocal inventory_calls, assignment_calls
            data = json.loads(payload)
            if system == INVENTORY_SYSTEM_PROMPT:
                inventory_calls += 1
                if "candidate_inventories" in data:
                    self.assertEqual(len(data["candidate_inventories"]), 2)
                    return {"workstreams": [
                        {"workstream_id": "W001", "loop_description": "Alpha", "completion_test": "Alpha resolved"},
                        {"workstream_id": "W002", "loop_description": "Beta", "completion_test": "Beta resolved"},
                    ]}
                if inventory_calls == 1:
                    items = [
                        {"workstream_id": "W001", "loop_description": "Alpha", "completion_test": "Alpha resolved"},
                        {"workstream_id": "W002", "loop_description": "Beta", "completion_test": "Beta resolved"},
                    ]
                else:
                    items = [
                        {"workstream_id": "W001", "loop_description": "Beta", "completion_test": "Beta resolved"},
                        {"workstream_id": "W002", "loop_description": "Alpha", "completion_test": "Alpha resolved"},
                    ]
                return {"workstreams": items}

            assignment_calls += 1
            assignment_payloads.append(data)
            if "assignment_candidates" in data:
                self.assertEqual(data["disputed_action_indices"], [0, 1, 2])
                self.assertEqual(data["fixed_workstreams"], [
                    {"workstream_id": "W001", "loop_description": "Alpha", "completion_test": "Alpha resolved"},
                    {"workstream_id": "W002", "loop_description": "Beta", "completion_test": "Beta resolved"},
                ])
                return {"action_workstream_ids": ["W002", "W001", "W002"]}
            if assignment_calls == 1:
                return {"action_workstream_ids": ["W001", "W002", "W001"]}
            return {"action_workstream_ids": ["W002", "W001", "W002"]}

        result = annotate("Research Alpha and Beta.", ["check beta", "check alpha", "verify beta"], complete, candidate_count=2)
        self.assertEqual(inventory_calls, 3)
        self.assertEqual(assignment_calls, 3)
        self.assertEqual(result["action_workstream_ids"], ["W001", "W002", "W001"])
        self.assertEqual([w["loop_description"] for w in result["workstreams"]], ["Beta", "Alpha"])
        self.assertTrue(all(payload["fixed_workstreams"] == assignment_payloads[0]["fixed_workstreams"]
                            for payload in assignment_payloads))

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
