"""Actual Habitat interaction smoke with a clearly labelled scripted text follower.

No model API. Uses a staged near-human pose, NOT a frozen episode starting pose.
The scripted policy reads only the counted English answer and public actions; it
never receives the scene goal or path. It follows at most the first local metre
then stops for inspection, NOT as an assertion of navigation task success.
"""
from __future__ import annotations
from dataclasses import replace
import argparse,json,math,re,hashlib
from pathlib import Path
from .data import load_episodes,load_dummies,EPISODES_REL,DUMMIES_REL,SCENE_REL
from .policy import Decision
from .runner import RunConfig,run_episode

class ScriptedTextFollower:
    def reset(self,*args):
        self.accepted=False;self.queue=None;self.target_distance=0.0;self.movement_start=None
    def decide(self,context):
        p=context.public
        if not self.accepted and p['active_offer']:
            self.accepted=True
            return Decision('wait','accept',p['active_offer']['offer_id'],
                'Which way should I go from my current heading to continue toward the destination?',
                'high','Diagnostic acceptance: this scripted smoke does not infer genuine model uncertainty.')
        answer=p['last_delivered_answer']
        if answer and self.queue is None:
            # Parse only our baseline oracle's public English first-leg syntax.
            self.queue=[]
            prefix=answer.split('; then ')[0]
            turn=re.search(r'turn (left|right) about (\d+) degrees',prefix)
            angle=p.get('action_specs',{}).get('turn_left',{}).get('turn_degrees')
            if (turn or 'turn around' in prefix) and (type(angle) not in (int,float) or angle<=0):
                return Decision('stop',short_reason='No verified turn increment; do not guess the rotation.')
            if turn:self.queue=[f'turn_{turn[1]}']*int(round(float(turn[2])/angle))
            elif 'turn around' in prefix:self.queue=['turn_left']*int(round(180/angle))
            distance=re.search(r'go forward about ([0-9.]+) meters',prefix)
            self.target_distance=min(1.0,float(distance[1])) if distance else 0
        if self.queue:
            return Decision(self.queue.pop(0),short_reason='Follow the delivered answer with a quantized safe turn.')
        if answer and self.target_distance>0:
            if self.movement_start is None:self.movement_start=p['observation_step']
            travelled=sum(row.get('displacement_m',0) for row in p['executed_history'] if row['step']>self.movement_start)
            step_length=p.get('action_specs',{}).get('move_forward',{}).get('requested_distance_m')
            # Never step beyond the end of the first described leg/turn corner.
            if (type(step_length) in (int,float) and step_length>0
                    and self.target_distance-travelled>=step_length-1e-6
                    and 'move_forward' in p['legal_candidate_ids']):
                return Decision('move_forward',short_reason='Follow up to one metre of the counted verbal answer; safety remains enforced.')
        return Decision('stop',short_reason='End the staged interaction check, not a claim of reaching the instruction goal.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True,type=Path)
    parser.add_argument('--navmesh',required=True,type=Path)
    parser.add_argument('--staging-report',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--allow-sliding',action='store_true',help='Explicit alternative motion profile; do not mix with strict runs')
    parser.add_argument('--agent-radius',type=float,default=.10)
    parser.add_argument('--turn-degrees',type=float,default=15.0)
    args=parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a fresh output directory; existing evidence is immutable')
    args.output.mkdir(parents=True,exist_ok=True)
    from .habitat_adapter import create_environment,AdapterConfig
    from .debug_view import draw_interaction_circle, draw_topdown_inset
    from PIL import Image,ImageDraw
    import numpy as np
    project=Path(__file__).resolve().parent
    original=load_episodes(args.data_root/EPISODES_REL)[0]
    staging=json.loads(args.staging_report.read_text())['staged_view']
    dummy=next(d for d in load_dummies(args.data_root/DUMMIES_REL,original) if d.dummy_id==staging['dummy_id'])
    original_dummy_position=dummy.ground_position
    if tuple(staging['dummy_ground_position'])!=dummy.ground_position:
        if staging.get('placement_kind')!='explicit_goal_independent_demo_relocation':
            raise ValueError('Changed dummy pose requires explicit audited demo relocation metadata')
        dummy=replace(dummy,ground_position=tuple(staging['dummy_ground_position']))
    pos=tuple(staging['camera_ground_position']);dx=dummy.ground_position[0]-pos[0];dz=dummy.ground_position[2]-pos[2]
    yaw=math.atan2(-dx,-dz)
    staged=replace(original,start_position=pos,start_rotation_xyzw=(0,math.sin(yaw/2),0,math.cos(yaw/2)))
    config=AdapterConfig(agent_radius_m=args.agent_radius,turn_degrees=args.turn_degrees,allow_sliding=args.allow_sliding)
    env=create_environment(args.data_root/SCENE_REL,args.navmesh,project/'assets/blue_dummy/light_blue_human.object_config.json',config=config)
    frames=[];raw_hashes=[];evidence=[]
    def sink(frame,event):
        idx=len(frames)
        raw=Image.fromarray(frame.rgb)
        raw.save(args.output/f'raw_{idx:03d}.png')
        before=hashlib.sha256(frame.rgb.tobytes()).hexdigest()
        overlay=draw_interaction_circle(frame.rgb,robot_ground_position=frame.pose.position,**env.debug_camera())
        overlay=draw_topdown_inset(overlay,robot_pose=frame.pose,dummies=[dummy],
            interaction_radius_m=config.interaction_radius_m,robot_radius_m=config.agent_radius_m)
        assert hashlib.sha256(frame.rgb.tobytes()).hexdigest()==before
        picture=Image.fromarray(overlay)
        draw=ImageDraw.Draw(picture)
        draw.rectangle((0,18,picture.width,60),fill=(0,0,0))
        draw.text((3,20),f"STAGED CPU SMOKE | step {frame.step} | {event['phase']}",fill='white')
        draw.text((3,34),f"Answers: {event['interaction']['delivered_answers']}/3 | action: {event.get('action','offer check')}",fill='white')
        draw.text((3,47),'Scripted text follower, NOT a VLN score',fill='yellow')
        picture.save(args.output/f'debug_{idx:03d}.png')
        frames.append(picture);raw_hashes.append(before)
        evidence.append({'index':idx,'step':frame.step,'event':event,'internal_pose':{'position':frame.pose.position,'yaw_rad':frame.pose.yaw_rad},
                         'internal_candidates':frame.dummy_candidates,'raw_sha256':before})
    try:
        result=run_episode(env,ScriptedTextFollower(),staged,[dummy],config=RunConfig(max_steps=24,
            backend_label='real_habitat',policy_label='scripted_interaction_smoke'),frame_sink=sink)
        result.update(is_benchmark_result=False,original_episode_id=original.episode_id,
            start_pose_overridden_for_staged_demo=True,agent_radius_m=config.agent_radius_m,turn_degrees=config.turn_degrees,allow_sliding=config.allow_sliding,only_one_dummy=True,dummy_original_position=original_dummy_position,
            dummy_position_overridden_for_staged_demo=dummy.ground_position!=original_dummy_position,
            official_episode_metrics_not_reported=True,placement_audit=env.placement_audit)
        # These task metrics are not applicable to an edited-start smoke. Preserve
        # physical movement/questions, but do not display edited-start success/SPL.
        result['metrics']['success']=None;result['metrics']['spl']=None
        result['metrics']['benchmark_status']='not_applicable_staged_interaction'
        (args.output/'interaction_result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        (args.output/'frame_evidence.private.json').write_text(json.dumps(evidence,indent=2,allow_nan=False))
        if frames:frames[0].save(args.output/'interaction_debug.gif',save_all=True,append_images=frames[1:],duration=500,loop=0)
        print(json.dumps({'question_count':result['metrics']['question_count'],'movement_m':result['metrics']['path_length_m'],
            'error':result['error'],'frames':len(frames),'output':str(args.output),'real_habitat':True,'benchmark_result':False}))
        return 0 if result['error'] is None and result['metrics']['question_count']>0 and result['metrics']['path_length_m']>0 else 2
    finally:env.close()

if __name__=='__main__':raise SystemExit(main())
