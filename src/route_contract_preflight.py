"""Check local constrained decoding for the route contract before Habitat starts."""

import json
import os

from openai import OpenAI

from navigation_contracts import completion_conditions, verification_schema
from navigation_policy import normalize_verification


def main():
    conditions = completion_conditions('Walk into the corridor. Stop beside the cabinet.', 'P1')
    schema = verification_schema(conditions, [2])['schema']
    client = OpenAI(base_url=os.environ['OPENAI_BASE_URL'], api_key=os.environ.get('OPENAI_API_KEY', 'EMPTY'))
    prompt = ('This checks JSON compatibility only. No scene or executed actions are available. '
              'Return outcome=false, failure_type=not_reached, confidence=low, '
              'relation_satisfied=false, scene_consistent=true, next_start_consistent=null, '
              'negative_landmarks_detected=[], contradiction_evidence="", target_entity_id=null, '
              'target_identity=unknown, target_proximity=unknown. For EVERY supplied condition '
              'return status=unknown, steps=[], and brief evidence stating missing observation. '
              'Condition IDs/text: ' + json.dumps(conditions))
    response = client.chat.completions.create(
        model=os.environ['OPENAI_MODEL'], messages=[{'role': 'user', 'content': prompt}],
        max_tokens=1024,
        extra_body={'enable_thinking': False,
                    'chat_template_kwargs': {'enable_thinking': False},
                    'structured_outputs': {'json': schema, 'disable_any_whitespace': True}},
    )
    parsed = json.loads(response.choices[0].message.content)
    result = normalize_verification(parsed)
    if (not result['available'] or result['outcome'] or result['rollback_required']
            or {c['id'] for c in parsed.get('condition_checks', [])} != {c['id'] for c in conditions}
            or any(c['status'] != 'unknown' or c['steps'] for c in parsed['condition_checks'])):
        raise SystemExit('Route contract preflight failed; no navigation episodes were started.')
    print('Route contract preflight passed: constrained conditions, nullable target ID, unknown state, thinking disabled.')


if __name__ == '__main__':
    main()

