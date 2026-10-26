"""Materialize only explicitly reviewed labels, causal inputs and qid-disjoint split."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

ACTIONS = ['CONTINUE', 'SWITCH_LOOP']


def reviewed_reward(label, terminal_outcome):
    """Return rewards for CONTINUE, SWITCH_LOOP in that action order."""
    if label not in ACTIONS or terminal_outcome not in {'CORRECT', 'INCORRECT', 'UNKNOWN'}:
        raise ValueError('invalid reviewed action or terminal outcome')
    bonus = 0.25 if label == 'SWITCH_LOOP' and terminal_outcome == 'CORRECT' else 0.0
    return ([1.0, -1.5] if label == 'CONTINUE' else [-2.0, 2.0 + bonus]), bonus


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_lines(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def expand(groups, key):
    result = {}
    for g in groups:
        if not g.get('reason'):
            raise ValueError('missing manual reason')
        for identity in g[key]:
            if identity in result:
                raise ValueError('duplicate manual identity '+identity)
            result[identity] = g
    return result


def split_qids(rows, seed=20260906):
    groups = defaultdict(list)
    for r in rows:
        groups[r['qid']].append(r)
    strata = defaultdict(list)
    for q,items in groups.items():
        strata[any(r['target_action']=='SWITCH_LOOP' for r in items)].append(q)
    rng = random.Random(seed)
    heldout = set()
    for key in sorted(strata):
        values = sorted(strata[key], key=int)
        rng.shuffle(values)
        n = max(1, round(len(values)*0.2))
        heldout.update(values[:n])
    train = [r for r in rows if r['qid'] not in heldout]
    dev = [r for r in rows if r['qid'] in heldout]
    for part in [train,dev]:
        if set(r['target_action'] for r in part)!=set(ACTIONS):
            raise ValueError('each qid-disjoint partition must contain both actions')
    return train,dev


def build(review, output):
    audit = json.loads((review/'source-audit.json').read_text())
    if not audit.get('training_safe') or not audit.get('source_verified'):
        raise ValueError('source isolation not verified')
    for name,key in [('events.jsonl','events_sha256'),('outcomes-private.jsonl','outcomes_sha256')]:
        if sha(review/name)!=audit[key]:
            raise ValueError('source changed '+name)
    annotation = json.loads((review/'manual-boundaries.json').read_text())
    if annotation['selection_sha256']!=sha(review/'review-selection.jsonl'):
        raise ValueError('review selection changed')
    labels = expand(annotation['groups'],'ids')
    outcome_labels = expand(json.loads((review/'manual-outcomes.json').read_text())['groups'],'qids')
    outcomes = {r['qid']:r for r in load_lines(review/'outcomes-private.jsonl')}
    if set(outcomes)!=set(outcome_labels):
        raise ValueError('outcomes not completely reviewed')
    selected = load_lines(review/'review-selection.jsonl')
    if set(labels)!={r['event_id'] for r in selected}:
        raise ValueError('incomplete review selection')
    original = {r['event_id']:r for r in load_lines(review/'events.jsonl')}
    rows=[]
    for r in selected:
        if r!=original[r['event_id']]:
            raise ValueError('not the exact runtime event')
        label=labels[r['event_id']]['label']
        if label=='EXCLUDE':
            continue
        if label not in ACTIONS:
            raise ValueError('invalid manual action')
        if r['trace_sha256']!=outcomes[r['qid']]['trace_sha256']:
            raise ValueError('trace mismatch')
        if [m['role'] for m in r['messages']]!=['system','user']:
            raise ValueError('noncausal messages')
        outcome=outcome_labels[r['qid']]['label']
        # Terminal correctness is a small association bonus, never claimed causal.
        reward,bonus=reviewed_reward(label,outcome)
        rows.append({'sample_id':r['event_id'],'qid':r['qid'],'round':r['round'],
                     'messages':r['messages'],'target_action':label,'reward':reward,
                     'runtime_action':r['runtime_action']['decision'],
                     'manual_reason':labels[r['event_id']]['reason'],
                     'terminal_label':outcome,'terminal_bonus':bonus,
                     'trace_sha256':r['trace_sha256']})
    train,dev=split_qids(rows)
    counts=Counter(r['target_action'] for r in train)
    switch_multiplier=min(3.0,counts['CONTINUE']/counts['SWITCH_LOOP'])
    qcounts=Counter(r['qid'] for r in train)
    weights=[(switch_multiplier if r['target_action']=='SWITCH_LOOP' else 1.0)/qcounts[r['qid']] for r in train]
    mean=sum(weights)/len(weights)
    for r,w in zip(train,weights):
        r['weight']=w/mean
    for r in dev:
        r['weight']=1.0
    output.mkdir(parents=True,exist_ok=False)
    for name,items in [('train.jsonl',train),('validation.jsonl',dev)]:
        with (output/name).open('x') as f:
            for r in items:
                f.write(json.dumps(r,ensure_ascii=False)+'\n')
    def stats(part):
        return {'rows':len(part),'qids':sorted({r['qid'] for r in part},key=int),
                'actions':dict(Counter(r['target_action'] for r in part))}
    manifest={'schema_version':'closedloop-manual-rl-v1','training_ready':True,
              'safety':{'train_source_verified':True,'test_question_overlap':0,'qid_split_overlap':0,
                        'actual_causal_inputs_unchanged':True,'no_external_annotation_api':True},
              'train':stats(train),'validation':stats(dev),
              'reviewed_calls':len(selected),'excluded_ambiguous_calls':len(selected)-len(rows),
              'manual_outcome_counts':dict(Counter(g['label'] for g in outcome_labels.values())),
              'switch_weight_multiplier':switch_multiplier,'weight_mean':1.0,
              'reward_definition':{'proper_continue':1.0,'false_switch':-1.5,'missed_switch':-2.0,
                                   'proper_switch':2.0,'correct_terminal_bonus_on_proper_switch':0.25,
                                   'counterfactual_or_causal_claim':False},
              'files':{n:sha(output/n) for n in ['train.jsonl','validation.jsonl']},
              'provenance':{n:sha(review/n) for n in ['source-audit.json','events.jsonl','outcomes-private.jsonl',
                                                    'manual-boundaries.json','manual-outcomes.json','review-selection.jsonl']}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    build(a.review,a.output)
