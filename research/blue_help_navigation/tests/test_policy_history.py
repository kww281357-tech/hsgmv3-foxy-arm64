import unittest
from blue_help.trial.policy import FullRouteVisualPolicy
from blue_help.policy import PolicyContext
class HistoryTests(unittest.TestCase):
    def test_measured_history_accumulates_without_private_fields(self):
        policy=FullRouteVisualPolicy();policy.reset('fixture','40','max3_ask')
        for step in range(1,20):
            public={'observation_step':step,'executed_history':[{'step':step,'action':'move_forward','displacement_m':.25,'yaw_delta_rad':0.,'collision':False,'private_pose':[9,8,7]}],'evidence_ids':[],'last_answer_metadata':{'question_step':0}}
            context=policy.enriched_context(PolicyContext(public,'synthetic-image'))
        self.assertEqual(len(context.public['executed_history']),19)
        self.assertAlmostEqual(context.public['observed_odometry_since_answer']['travelled_m'],4.75)
        self.assertNotIn('private_pose',str(context.public))
        self.assertIn('ACTION_1',context.public['evidence_ids'])
        policy.reset('fixture','40','max3_ask');self.assertEqual(policy._observed_history,{})
    def test_original_context_not_modified(self):
        p=FullRouteVisualPolicy();ctx=PolicyContext({'observation_step':0,'executed_history':[],'evidence_ids':[],'last_answer_metadata':None},'image')
        out=p.enriched_context(ctx);self.assertEqual(ctx.public['evidence_ids'],[]);self.assertEqual(out.public['evidence_ids'],['OBS_0'])
if __name__=='__main__':unittest.main()
