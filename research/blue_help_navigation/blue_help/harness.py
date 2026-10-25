"""Bounded paired evaluation orchestrator; no credential/network fallback.

One simulator and one policy are active at a time. The caller must explicitly
inject an approved transport and evaluator geodesic. Baseline and max3 use the
same scene layout, episode order and seeds; no outcome-based episode selection.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
from .data import FROZEN_IDS
from .runner import run_episode,RunConfig


def run_paired(episodes, dummies_for_episode, *, env_factory, policy_factory,
               output_dir, config=RunConfig(), batch_size=2, batches_to_run=1,
               metric_factory=None, before_model_request=None):
    if type(batch_size) is not int or not 2<=batch_size<=4:
        raise ValueError('batch_size must be 2..4')
    if type(batches_to_run) is not int or not 1<=batches_to_run<=10:
        raise ValueError('Explicit batch count 1..10 required')
    if config.backend_label=='real_habitat' and metric_factory is None:
        raise ValueError('Real paired evaluation requires an explicit evaluator geodesic; a proxy is not sufficient')
    episodes=list(episodes)
    if [int(ep.episode_id) for ep in episodes]!=FROZEN_IDS:
        raise ValueError('Use the frozen 20-episode order without outcome selection')
    destination=Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Use a fresh output directory; never overwrite or silently resume results')
    destination.mkdir(parents=True,exist_ok=True)
    manifest={'status':'running','api_concurrency':1,'batch_size_episodes':batch_size,
              'requested_batches':batches_to_run,'episode_ids':FROZEN_IDS,'conditions':['no_ask','max3_ask'],
              'seed':config.seed,'automatic_retry':False,'provider_fallback':False,'runs':[]}
    def save():
        (destination/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    save()
    limit=min(20,batch_size*batches_to_run)
    try:
        for episode in episodes[:limit]:
            dummies=list(dummies_for_episode(episode))
            for arm in ('no_ask','max3_ask'):
                env=env_factory()
                try:
                    # Rate limits/cost controls belong to the explicitly approved
                    # transport. This hook is only a lifecycle preparation hook;
                    # it cannot pretend to gate each call inside an opaque client.
                    if before_model_request is not None:
                        raise ValueError('Per-request enforcement must wrap the transport, not a per-run hook')
                    metric=metric_factory(env) if metric_factory else None
                    if config.backend_label=="real_habitat" and not callable(metric):
                        raise ValueError("Real evaluator factory must return a callable; do not construct a policy")
                    policy=policy_factory(episode.episode_id,arm)
                    result=run_episode(env,policy,episode,dummies,arm=arm,config=config,metric_distance=metric)
                    name=f'{episode.episode_id}_{arm}.json'
                    (destination/name).write_text(json.dumps(result,indent=2,allow_nan=False))
                    manifest['runs'].append({'episode_id':episode.episode_id,'arm':arm,'file':name,
                        'valid':result['metrics']['valid'],'error_type':result['error']['type'] if result['error'] else None})
                    save()
                    if result['error'] is not None:
                        manifest['status']='paused_on_runtime_or_provider_error';save();return manifest
                    if not result['metrics']['valid']:
                        manifest['status']='paused_invalid_measurement';save();return manifest
                finally:env.close()
        manifest['status']='requested_batches_complete';save();return manifest
    except Exception as exc:
        manifest['status']='paused_on_setup_error';manifest['error_type']=type(exc).__name__;save()
        raise
