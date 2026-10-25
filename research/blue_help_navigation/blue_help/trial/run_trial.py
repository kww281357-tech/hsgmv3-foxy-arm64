"""One staged episode40 closed-loop entry. No scene goal/path goes to the policy.
--fixture-preflight uses FAKE responses and proves wiring only, NOT model ability.
Live mode requires a successful actual probe + verified Extra Usage OFF, and reads
one explicitly approved key without echo. It never changes account settings.
"""
from pathlib import Path
from dataclasses import replace
import sys,argparse,json,math,getpass,hashlib
from blue_help.data import *
from blue_help.habitat_adapter import AdapterConfig,create_environment
from blue_help.habitat_metrics import make_geodesic_evaluator
from blue_help.helper_assets import prepare_helper,replace_dummy_for_helper,orient_game_helper_to_anchor
from blue_help.runner import run_episode,RunConfig
from blue_help.policy import PolicyContext
from blue_help.trial.policy import FullRouteVisualPolicy
from blue_help.providers.go_transport import GoTransport,Limits
from blue_help.manual_route.bridge import load_frozen_card,make_card_oracle_factory
from PIL import Image,ImageDraw
class FixtureTransport:
    """Canned accept then stop; no inferred vision/route understanding."""
    def __init__(self):self.requests=0;self.records=[]
    def __call__(self,request):
        p=json.loads(json.loads(request.body)['messages'][-1]['content'][0]['text']);self.requests+=1
        base={'intent':'STOP','offer_choice':None,'offer_id':None,'candidate_id':None,'uncertainty':'high','evidence_ids':[],'short_reason':'Fixture preflight ends; this is not a model judgment.','question':None}
        if self.requests==1:
            assert p['active_offer'] and p['last_delivered_answer'] is None
            base.update(intent='REQUEST_HELP',offer_choice='accept',offer_id=p['active_offer']['offer_id'],question='From the safe spot beside you, could you give me the whole route to the bed corner, with turns, approximate distances and recognizable objects?',short_reason='Synthetic acceptance to test the new full-route interface, not uncertainty inference.')
        else:
            assert p['last_delivered_answer'] and '7.4 meters' in p['last_delivered_answer']
            assert 'to the right of the bed' in p['last_delivered_answer']
        return json.dumps({'model':'deepseek-v4.1-flash','choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':json.dumps(base)}}],'usage':{'cost':0,'opencode':{'billing_source':'go','allowance':0}}}).encode()
    def close(self):pass

def mark_staged_not_benchmark(result):
    """Invalidate BOTH score views so generic aggregators cannot pool staged runs."""
    result['is_benchmark_result'] = False
    result['measurement_valid'] = result['metrics'].get('valid', False)
    result['metrics'].update(valid=False, success=None, spl=None,
                             invalid_reason='not_applicable_staged_start')
    result['evaluation'].update(status='invalid', success=None, spl=None,
        invalid_reason='not_applicable_staged_start', benchmark_status='not_applicable_staged_start')
    return result


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--fixture-preflight',action='store_true');ap.add_argument('--confirmed-extra-usage-off',action='store_true');ap.add_argument('--successful-probe',type=Path);ap.add_argument('--confirmed-probe-image-match',action='store_true');ap.add_argument('--max-steps',type=int,default=160);ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--review-root',type=Path,required=True);ap.add_argument('--staging-report',type=Path,required=True);ap.add_argument('--helper-profile',choices=['game-v7','blue-reference'],default='game-v7');ap.add_argument('--helper-asset',type=Path);args=ap.parse_args(argv)
    if not 2<=args.max_steps<=160:raise SystemExit('Use a bounded2..160step single trial')
    output_abs=args.output.resolve()
    if any(output_abs==root.resolve() or root.resolve() in output_abs.parents for root in (args.data_root,args.review_root)):
        raise SystemExit('Private outputs must not modify frozen data or review inputs')
    if args.output.exists():raise SystemExit('Fresh output path required')
    if not args.fixture_preflight:
        if not args.confirmed_extra_usage_off:raise SystemExit('BLOCKED: verify Extra Usage OFF before any request')
        if not args.successful_probe:raise SystemExit('BLOCKED: first complete the separate one-image probe')
        if not args.confirmed_probe_image_match:raise SystemExit('BLOCKED: compare the probe description with its actual source image first')
        probe=json.loads(args.successful_probe.read_text())
        if probe.get('status')!='response_received_and_schema_valid' or probe.get('model_returned')!='deepseek-v4.1-flash' or probe.get('usage',{}).get('cost')!=0:raise SystemExit('Probe is not a successful zero-balance-cost same-model response')
    args.output.mkdir(parents=True)
    data=args.data_root;project=Path(__file__).resolve().parents[1]
    verified_data=validate_frozen_data(data)
    (args.output/'data_integrity.private.json').write_text(json.dumps(verified_data,indent=2))
    card=load_frozen_card(args.review_root)
    stage_bytes=args.staging_report.read_bytes()
    stage=json.loads(stage_bytes)['staged_view']
    (args.output/'input_references.private.json').write_text(json.dumps({'staging_report_sha256':hashlib.sha256(stage_bytes).hexdigest(),'review_card_id':card.route_card_id},indent=2))
    original=load_episodes(data/EPISODES_REL)[0];assert original.episode_id=='40'
    dummy=next(d for d in load_dummies(data/DUMMIES_REL,original) if d.dummy_id==stage['dummy_id'])
    dummy=replace(dummy,ground_position=tuple(stage['dummy_ground_position']))
    helper=prepare_helper(args.helper_profile,args.helper_asset,args.output/'helper_runtime')
    dummy=replace_dummy_for_helper(dummy,helper)
    if helper.profile=="game-v7":dummy=orient_game_helper_to_anchor(dummy,card.anchor.pose.position)
    (args.output/'helper_asset.private.json').write_text(json.dumps(helper.provenance,indent=2))
    staged=replace(original,start_position=tuple(stage['camera_ground_position']),start_rotation_xyzw=tuple(stage['camera_rotation_xyzw']))
    config=AdapterConfig(agent_radius_m=.1,turn_degrees=15,allow_sliding=True,width=640,height=480,helper_visibility_mode=helper.visibility_mode)
    env=create_environment(data/SCENE_REL,data/NAVMESH_REL,helper.config_path,config=config)
    # No key is read until local data/card/environment construction succeeds.
    transport=None;frames={};evidence=[];result=None
    def sink(frame,event):
        if frame.step not in frames:
            name=f'raw_{frame.step:03d}.png';Image.fromarray(frame.rgb).save(args.output/name)
            frames[frame.step]=name
            evidence.append({'step':frame.step,'raw_file':name,'raw_pixels_sha256':hashlib.sha256(frame.rgb.tobytes()).hexdigest(),'private_pose':{'position':frame.pose.position,'yaw_rad':frame.pose.yaw_rad},'collision':frame.collision})
    try:
        factory=make_card_oracle_factory(env,card,[dummy])
        if args.fixture_preflight:transport=FixtureTransport()
        else:
            if not sys.stdin.isatty():raise SystemExit('Use an interactive no-echo terminal; do not pass a key file or command argument')
            credential=getpass.getpass('Approved transient Go credential (input hidden; not saved): ')
            transport=GoTransport(credential,extra_usage_disabled_confirmed=True,limits=Limits(max_requests=args.max_steps));credential=None
        policy=FullRouteVisualPolicy(transport)
        result=run_episode(env,policy,staged,[dummy],config=RunConfig(max_steps=4 if args.fixture_preflight else args.max_steps,experiment_id='ep40-fullroute-organized-v0.4',backend_label='real_habitat',policy_label='fake_response_preflight' if args.fixture_preflight else 'live_vlm_staged_helper_route'),metric_distance=make_geodesic_evaluator(env),oracle_factory=factory,frame_sink=sink)
        result.update(start_overridden_for_staged_help_trial=True,is_benchmark_result=False,is_autonomous_vln_result=False,original_episode_start=original.start_position,real_model_called=not args.fixture_preflight,fixture_response_calls=transport.requests if args.fixture_preflight else 0,actual_api_calls=0 if args.fixture_preflight else transport.requests,full_route_card_id=card.route_card_id,detector_integrated_into_policy=False)
        result['staged_trial_reached_goal_region']=bool(result['metrics']['stopped'] and result['metrics']['final_distance_m'] is not None and result['metrics']['final_distance_m']<=3.0) if not args.fixture_preflight and result['error'] is None else None
        mark_staged_not_benchmark(result)
        if args.fixture_preflight:result['api_attempts']=0;result['provider_usage']=[]
        result['helper_profile']=helper.profile
        result['helper_placement_audit']=env.placement_audit
        result['helper_recognition']='simulator_rendered_instance_pixels_and_true_raycast_not_detector'
        result['transport_audit']=transport.records
        result['public_input_audit']=policy.public_audits
        (args.output/'public_model_inputs.json').write_text(json.dumps(policy.wire_public_inputs,indent=2))
        # Check numeric private coordinate sequences are not in the TEXT payload.
        text=json.dumps(policy.wire_public_inputs,sort_keys=True)
        for forbidden in (list(card.goal.position),list(card.anchor.pose.position),list(card.dummy.foot)):
            assert json.dumps(forbidden) not in text
        assert 'private_full_path' not in text and 'human_guide_master_bedroom' not in text
        result['public_privilege_checks_passed']=True
    finally:
        if transport is not None:transport.close()
        env.close()
        (args.output/'frame_evidence.private.json').write_text(json.dumps(evidence,indent=2))
        if result is not None:(args.output/'trial_result.private.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'fixture':args.fixture_preflight,'actual_api_calls':result['actual_api_calls'],'questions_delivered':result['metrics']['question_count'],'movement_m':result['metrics']['path_length_m'],'error':result['error'],'public_privilege_checks_passed':result['public_privilege_checks_passed'],'output':str(args.output)},indent=2))
    return 0 if result['error'] is None else 2
if __name__=='__main__':raise SystemExit(main())
