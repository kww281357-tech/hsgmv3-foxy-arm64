"""New frozen English prompt; original allowlisted public context/action parser."""
from pathlib import Path
import copy,math
import sys,json
from blue_help.policy import VisualLanguagePolicy,parse_decision,PolicyContext
from blue_help.providers.go_contract import prepare_request,parse_response
PROMPT_PATH=Path(__file__).resolve().parents[1]/'prompt_data/01_policy_en.txt'
FORBIDDEN_KEYS={'goal_position','annotated_goal','goal','reference_path','private_full_path','query_corners','world_pose','route_origin','internal_pose','anchor','dummy_ground','true_goal_distance','final_distance_m','shortest_start_distance_m'}
def audit_public(public):
    def walk(value):
        if isinstance(value,dict):
            if FORBIDDEN_KEYS.intersection(value):raise ValueError('Privileged field in policy input')
            for v in value.values():walk(v)
        elif isinstance(value,(tuple,list)):
            for v in value:walk(v)
    walk(public)
    # Coordinate/pose fields are absent by the original context allowlist; prose
    # from an explicitly delivered answer is permitted, never an encoded map.
    return True
class FullRouteVisualPolicy(VisualLanguagePolicy):
    def __init__(self,transport=None):
        super().__init__(transport);self.public_audits=[];self.wire_public_inputs=[];self._observed_history={}
    def reset(self,*args):
        super().reset(*args);self.public_audits=[];self.wire_public_inputs=[];self._observed_history={}
    def enriched_context(self,context):
        public=copy.deepcopy(context.public)
        for row in public.get("executed_history",[]):
            clean={k:row[k] for k in ("step","action","collision","displacement_m","yaw_delta_rad") if k in row}
            self._observed_history[clean["step"]]=clean
        if len(self._observed_history)>500:raise ValueError("Observed history bound exceeded")
        public["executed_history"]=list(self._observed_history.values())
        public["evidence_ids"]=["OBS_"+str(public["observation_step"])]+[f"ACTION_{x}" for x in self._observed_history]
        meta=public.get("last_answer_metadata")
        if meta:
            rows=[r for r in self._observed_history.values() if r["step"]>meta["question_step"]]
            public["observed_odometry_since_answer"]={"travelled_m":sum(r.get("displacement_m",0) for r in rows),"accumulated_turn_degrees":math.degrees(sum(r.get("yaw_delta_rad",0) for r in rows)),"meaning":"Measured executed motion only; not distance to goal or route completion."}
        audit_public(public);return PolicyContext(public,context.image_data_uri)
    def decide(self,context):
        if self.transport is None:raise RuntimeError('No authorized transport; no request')
        if self._session is None:raise RuntimeError('Reset policy first')
        context=self.enriched_context(context)
        self.wire_public_inputs.append(copy.deepcopy(context.public))
        request=prepare_request([{'role':'system','content':PROMPT_PATH.read_text()},
          {'role':'user','content':[{'type':'text','text':json.dumps(context.public,ensure_ascii=False,allow_nan=False)},
          {'type':'image_url','image_url':{'url':context.image_data_uri}}]}],session=self._session,max_tokens=1024)
        self.public_audits.append({'step':context.public.get('observation_step'),'passed':True,'request_bytes':len(request.body),'image_count':request.image_count})
        self.api_attempts+=1
        response=parse_response(self.transport(request));self.usage.append(response.usage)
        return parse_decision(response.content,context)
