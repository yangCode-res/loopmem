from __future__ import annotations
import json
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loopmem import (
    ContextOnlyLoopBoundaryJudge,
    OnlineMemorySession,
    RetrievalBudgetGate,
    TransitionPlanner,
)


class ScriptedClient:
    model = "test-fixture"
    def __init__(self): self.decisions=[]; self.boundary_inputs=[]; self.planner_inputs=[]
    def complete_json(self, system_prompt, user_prompt):
        payload=json.loads(user_prompt)
        if "You manage context-only memory boundaries" in system_prompt:
            self.boundary_inputs.append(payload)
            return self.decisions.pop(0)
        if "candidate_action" in payload and "active_loop" in payload and "task_summary" in payload:
            self.boundary_inputs.append(payload)
            return self.decisions.pop(0)
        self.planner_inputs.append(payload)
        is_new=not payload.get("closed_loop_handoff")
        desc="source discovery" if is_new else "publication date verification"
        return {"next_loop":{"loop_description":desc,"target_relation":desc,"completion_test":"tool evidence establishes or rules out the relation"},"retrieval_request":{"subject_scope":"Alpha project","target_relation":desc,"required_evidence":"source-backed date or source record","key_entities":["Alpha"]}}


def call(call_id, query):
    return {"id":call_id,"type":"function","function":{"name":"search","arguments":json.dumps({"query":query})}}


def batch(prefix, queries, texts):
    action={"role":"assistant","tool_calls":[call(f"{prefix}{i}",q) for i,q in enumerate(queries)]}
    results=[{"role":"tool","tool_call_id":f"{prefix}{i}","name":"search","content":text} for i,text in enumerate(texts)]
    return action,results


class RuntimeTests(unittest.TestCase):
    def make_session(self, client):
        return OnlineMemorySession(question="Find Alpha's source and publication date.",controller=ContextOnlyLoopBoundaryJudge(client),planner=TransitionPlanner(client),top_k=3,record_top_k=5)

    def test_classifies_full_multi_tool_batch_after_paired_results(self):
        client=ScriptedClient(); session=self.make_session(client)
        first=batch("a",["Alpha source","Alpha project report"],["Alpha source describes project funding.","The Alpha report lists funding year 2020."])
        session.submit_batch(*first)
        client.decisions=[{"decision":"SWITCH_LOOP","closed_loop_key":"L001"}]
        incoming=batch("b",["Alpha publication date","report publication date"],["Alpha report published 2024-01-01.","Publication metadata: 2024-01-01."])
        decision=session.submit_batch(*incoming)
        self.assertTrue(decision.split)
        self.assertEqual(len(decision.controller_input["candidate_results"]),2)
        self.assertIn("Alpha publication date",decision.controller_input["candidate_action"]["arguments"])
        self.assertIn("report publication date",decision.controller_input["candidate_action"]["arguments"])
        self.assertEqual(len(session.completed_loops),1)

    def test_switch_archives_old_loop_and_seeds_new_loop_with_whole_batch(self):
        client=ScriptedClient(); session=self.make_session(client)
        first=batch("a",["Alpha source","Alpha funding"],["Alpha source describes project funding.","Funding year was 2020."])
        session.submit_batch(*first)
        old_sequences={event.sequence for event in session.current_events}
        client.decisions=[{"decision":"SWITCH_LOOP","closed_loop_key":"L001"}]
        incoming=batch("b",["publication date","report date"],["The publication date is 2024-01-01.","The report date is 2024-01-01."])
        session.submit_batch(*incoming)
        archived=session.completed_loops[0]
        self.assertEqual(archived.subgoal, "source discovery")
        self.assertTrue(old_sequences.isdisjoint({event.sequence for event in session.current_events}))
        self.assertEqual([e.role for e in session.current_events], ["assistant","tool","tool"])
        self.assertEqual(len(archived.actions),1)
        self.assertEqual(len(archived.observations),2)
        self.assertEqual(session.current_events[0].message, incoming[0])
        self.assertEqual(session.controller.active_contract["loop_key"],"L002")

    def test_continue_retains_objective_without_replanning(self):
        client=ScriptedClient(); session=self.make_session(client)
        first=batch("a",["Alpha source"],["Alpha source describes project funding."])
        session.submit_batch(*first)
        planner_calls=len(client.planner_inputs)
        client.decisions=[{"decision":"CONTINUE","closed_loop_key":""}]
        same=batch("b",["Alpha source"],["The same source gives another line of evidence."])
        decision=session.submit_batch(*same)
        self.assertFalse(decision.split)
        self.assertEqual(len(client.planner_inputs),planner_calls)
        self.assertEqual(session.controller.active_contract["loop_description"],"source discovery")
        self.assertEqual(len(session.current_events),4)

    def test_each_pack_retrieves_evidence_with_parent_loop_context(self):
        client=ScriptedClient(); session=self.make_session(client)
        first=batch("a",["Alpha source"],["Alpha source established the project funding year 2020."])
        session.submit_batch(*first)
        client.decisions=[{"decision":"SWITCH_LOOP","closed_loop_key":"L001"}]
        second=batch("b",["publication date"],["The date is 2024-01-01."])
        session.submit_batch(*second)
        system={"role":"system","content":"research"}; user={"role":"user","content":session.question}
        packed=session.build_prompt(system,user)
        self.assertEqual(len(session.traces),1)
        self.assertIn("<cross_loop_memory>",packed[0]["content"])
        self.assertTrue(session.traces[0].selected_loop_ids)
        memory_block=packed[0]["content"].split("<cross_loop_memory>",1)[1]
        self.assertIn("source discovery",memory_block)
        self.assertIn("records",memory_block)
        session.build_prompt(system,user)
        self.assertEqual(len(session.traces),2)

    def test_budget_expands_with_ambiguous_rank_gaps(self):
        gate=RetrievalBudgetGate()
        decision=gate.begin(query="alpha evidence",loop_scores=[0.90,0.89,0.88,0.10],candidate_loop_count=4)
        self.assertEqual((decision.loop_k,decision.record_k),(3,0))
        result_scores=[0.95,0.90,0.85,0.80,0.75,0.74,0.3,0.1]
        finished=gate.finish(decision,query="alpha evidence",record_scores=result_scores,compact_record_ids=[f"r{i}" for i in range(5)],candidate_record_count=8)
        self.assertEqual(finished.record_k,8)
        self.assertEqual(finished.mode,"expanded")

    def test_single_result_preserves_controller_and_planner_wire_format(self):
        client=ScriptedClient(); session=self.make_session(client)
        session.submit_batch(*batch("a",["Alpha source"],["Alpha source was located."]))
        self.assertIsInstance(client.planner_inputs[0]["observed_result"],str)
        client.decisions=[{"decision":"CONTINUE","closed_loop_key":""}]
        decision=session.submit_batch(*batch("b",["Alpha source"],["Another Alpha line."]))
        self.assertIn("candidate_result",decision.controller_input)
        self.assertNotIn("candidate_results",decision.controller_input)

    def test_handoff_keeps_outgoing_loop_even_without_lexical_match(self):
        client=ScriptedClient(); session=self.make_session(client)
        session.submit_batch(*batch("a",["Alpha source"],["Alpha source was located."]))
        client.decisions=[{"decision":"SWITCH_LOOP","closed_loop_key":"L001"}]
        session.submit_batch(*batch("b",["publication date"],["A date was verified."]))
        outgoing=session.completed_loops[-1].loop_id
        session.retrieval_request={"target_relation":"zzzzunrelated"}
        session._latest_observation="zzzzunrelated"
        session.build_prompt({"role":"system","content":"research"},{"role":"user","content":session.question})
        self.assertEqual(session.traces[-1].selected_loop_ids[0],outgoing)

    def test_finalization_does_not_archive_same_loop_twice(self):
        client=ScriptedClient(); session=self.make_session(client)
        session.submit_batch(*batch("a",["Alpha source"],["Alpha source was located."]))
        session.finalize()
        self.assertIsNone(session.finalize())
        self.assertEqual(len(session.completed_loops),1)


if __name__ == "__main__": unittest.main()
