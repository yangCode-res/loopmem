"""Prepare question-disjoint boundary-policy data from reviewed JSONL samples."""
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_qids(rows, seed=20260906):
    groups = defaultdict(list)
    for row in rows:
        groups[row['qid']].append(row)
    strata = defaultdict(list)
    for qid, items in groups.items():
        strata[any(row['target_action'] == 'SWITCH_LOOP' for row in items)].append(qid)
    rng = random.Random(seed)
    heldout = set()
    for key in sorted(strata):
        values = sorted(strata[key])
        rng.shuffle(values)
        heldout.update(values[:max(1, round(len(values) * 0.2))])
    train = [row for row in rows if row['qid'] not in heldout]
    validation = [row for row in rows if row['qid'] in heldout]
    for part in (train, validation):
        if {row['target_action'] for row in part} != set(ACTIONS):
            raise ValueError('each question-disjoint partition must contain both actions')
    return train, validation


def build(review, output):
    rows = [json.loads(line) for line in review.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('reviewed sample file is empty')
    identities = set()
    for row in rows:
        identity = row.get('sample_id')
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError('missing or duplicate sample_id')
        identities.add(identity)
        if not isinstance(row.get('qid'), str) or not row['qid']:
            raise ValueError('qid must be a nonempty string')
        messages = row.get('messages')
        if not isinstance(messages, list) or [m.get('role') for m in messages] != ['system', 'user']:
            raise ValueError('controller input must contain system and user messages only')
        if any(not isinstance(m.get('content'), str) or not m['content'].strip() for m in messages):
            raise ValueError('message content must be nonempty text')
        row['reward'], row['terminal_bonus'] = reviewed_reward(
            row['target_action'], row.get('terminal_label', 'UNKNOWN'))
    train, validation = split_qids(rows)
    counts = Counter(row['target_action'] for row in train)
    switch_multiplier = min(3.0, counts['CONTINUE'] / counts['SWITCH_LOOP'])
    qcounts = Counter(row['qid'] for row in train)
    weights = [(switch_multiplier if row['target_action'] == 'SWITCH_LOOP' else 1.0)
               / qcounts[row['qid']] for row in train]
    mean = sum(weights) / len(weights)
    for row, weight in zip(train, weights):
        row['weight'] = weight / mean
    for row in validation:
        row['weight'] = 1.0
    output.mkdir(parents=True, exist_ok=False)
    for name, items in [('train.jsonl', train), ('validation.jsonl', validation)]:
        (output / name).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in items))
    manifest = {
        'schema_version': 'loopmem-boundary-policy-data-v1',
        'source_sha256': sha(review),
        'train_questions': len({row['qid'] for row in train}),
        'validation_questions': len({row['qid'] for row in validation}),
        'switch_weight_multiplier': switch_multiplier,
        'files': {name: sha(output / name) for name in ['train.jsonl', 'validation.jsonl']},
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True, help='reviewed decision samples in JSONL format')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.review, args.output), indent=2))
