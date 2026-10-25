import json
from pathlib import Path
import sys
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import target_identity
import visual_candidate_binding as vcb


def _make_candidate(obs_id=7, view='front', idx=0, label='door', conf=0.85,
                    bbox=None, image_size=None, entity_id=11, distance=2.4):
    cid = f'{obs_id}:{view}:{idx}'
    return cid, {
        'candidate_id': cid,
        'canonical_label': label,
        'detector_class_name': label,
        'observation_id': obs_id,
        'step': 3,
        'view': view,
        'confidence': conf,
        'bbox': bbox if bbox is not None else [0.2, 0.1, 0.6, 0.9],
        'image_size': image_size if image_size is not None else [100, 80],
        'mask_key': cid,
        'entity_id': entity_id,
        'geometry_status': 'valid' if entity_id is not None else 'no_valid_depth',
        'geometry_reason': 'ok' if entity_id is not None else 'depth_missing',
        'valid_depth_fraction': 0.8 if entity_id is not None else 0.0,
        'horizontal_distance_m': distance if entity_id is not None else None,
    }


class VisualCandidateBindingTests(unittest.TestCase):
    def test_crop_color_from_raw_original_and_coordinate_frame_metadata(self):
        raw = np.zeros((80, 100, 4), dtype=np.uint8)
        raw[:, :, 3] = 255
        raw[:, :, :3] = [10, 20, 30]
        raw[8:72, 20:60, :3] = [210, 45, 90]
        raw_copy = raw.copy()
        cid, rec = _make_candidate(obs_id=5, view='front', idx=0, label='sofa', bbox=[0.2, 0.1, 0.6, 0.9])

        crops = vcb.select_candidate_crops(
            {cid: rec}, {'front': raw}, observation_id=5, roles={cid: ['primary', 'primary']}, padding=0.0
        )
        self.assertEqual(len(crops), 1)
        row = crops[0]
        self.assertEqual(row['crop_bounds_px'], [20, 8, 60, 72])
        self.assertEqual(row['image'].mode, 'RGB')
        self.assertEqual(row['image'].size, (40, 64))
        self.assertEqual(row['image'].getpixel((20, 30)), (210, 45, 90))
        self.assertEqual(row['roles'], ['primary'])
        self.assertEqual(row['bbox_space'], 'full_view')
        self.assertIn(f'candidate_id={cid}', row['caption'])
        self.assertIn('observation_id=5', row['caption'])
        self.assertIn('view=front', row['caption'])
        self.assertIn('FULL_VIEW', row['caption'])
        np.testing.assert_array_equal(raw, raw_copy)

        meta = {k: v for k, v in row.items() if k != 'image'}
        encoded = json.dumps(meta)
        self.assertNotIn('mask', encoded)
        self.assertNotIn('distance', encoded)

    def test_crop_padding_clipping_aspect_max_side_limit_stale_and_dimensions(self):
        img = np.full((200, 400, 3), 128, dtype=np.uint8)
        c1, r1 = _make_candidate(obs_id=3, view='front', idx=0, label='chair', conf=0.60,
                                 bbox=[0.02, 0.10, 0.52, 0.60], image_size=[400, 200])
        c2, r2 = _make_candidate(obs_id=3, view='front', idx=1, label='table', conf=0.95,
                                 bbox=[0.20, 0.20, 0.70, 0.70], image_size=[400, 200])
        c_stale, r_stale = _make_candidate(obs_id=2, view='front', idx=2, label='chair', conf=0.99,
                                           bbox=[0.1, 0.1, 0.4, 0.4], image_size=[400, 200])
        c_nan, r_nan = _make_candidate(obs_id=3, view='front', idx=3, label='chair', conf=float('nan'),
                                       bbox=[0.1, 0.1, 0.4, 0.4], image_size=[400, 200])
        c_dim, r_dim = _make_candidate(obs_id=3, view='front', idx=4, label='chair', conf=0.90,
                                       bbox=[0.1, 0.1, 0.4, 0.4], image_size=[320, 200])

        candidates = [r2, r1, r1, r_stale, r_nan, r_dim]
        crops = vcb.select_candidate_crops(
            candidates, {'front': img}, observation_id=3, roles={c1: ['reference']},
            limit=10, max_side=80, padding=0.15
        )
        self.assertEqual([c['candidate_id'] for c in crops], [c1, c2])
        self.assertEqual(crops[0]['crop_bounds_px'], [0, 5, 238, 135])
        self.assertEqual(crops[0]['image'].size, (80, int(round(130 * (80 / 238.0)))))

        extra = {}
        for i in range(8):
            cid, rec = _make_candidate(obs_id=3, view='front', idx=i, label='chair',
                                       conf=0.9 - i * 0.05, bbox=[0.1, 0.1, 0.2, 0.2], image_size=[400, 200])
            extra[cid] = rec
        capped = vcb.select_candidate_crops(extra, {'front': img}, observation_id=3, limit=9, max_side=640)
        self.assertEqual(len(capped), 6)
        self.assertEqual(capped[0]['image'].size, (52, 26))

    def test_resolve_candidate_region_and_bbox_masked_refinement(self):
        cid, rec = _make_candidate(obs_id=4, view='left', idx=0, label='doorframe',
                                   bbox=[0.20, 0.10, 0.60, 0.90], image_size=[100, 80], entity_id=9)
        mask = np.zeros((80, 100), dtype=bool)
        mask[8:72, 20:60] = True
        masks = {rec['mask_key']: mask}

        full = vcb.resolve_candidate_region(
            {'candidate_id': cid}, candidates={cid: rec}, masks=masks,
            observation_id=4, endpoint_text='stop at the door'
        )
        self.assertTrue(full['valid'])
        self.assertEqual(full['bbox'], [0.20, 0.10, 0.60, 0.90])
        self.assertEqual(int(np.count_nonzero(full['mask'])), 64 * 40)

        jamb_roi = [0.20, 0.10, 0.30, 0.90]
        refined = vcb.resolve_candidate_region(
            {'candidate_id': cid, 'bbox': jamb_roi, 'bbox_space': 'full_view'},
            candidates={cid: rec}, masks=masks, observation_id=4, endpoint_text='stop at the door'
        )
        self.assertTrue(refined['valid'])
        self.assertEqual(refined['bbox'], jamb_roi)
        self.assertEqual(int(np.count_nonzero(refined['mask'])), 64 * 10)
        self.assertFalse(np.any(refined['mask'][:, 30:]))

        bad_space = vcb.resolve_candidate_region(
            {'candidate_id': cid, 'bbox': jamb_roi, 'bbox_space': 'crop'},
            candidates={cid: rec}, masks=masks, observation_id=4, endpoint_text='stop at the door'
        )
        self.assertFalse(bad_space['valid'])

    def test_resolve_rejects_stale_wrong_view_class_mask_id_and_entity_conflicts(self):
        cid, rec = _make_candidate(obs_id=8, view='right', idx=1, label='door',
                                   bbox=[0.2, 0.2, 0.5, 0.8], image_size=[100, 80], entity_id=3)
        mask = np.zeros((80, 100), dtype=bool)
        mask[16:64, 20:50] = True
        masks = {rec['mask_key']: mask}
        cands = {cid: rec}

        self.assertFalse(vcb.resolve_candidate_region(
            {'candidate_id': '8:right:99'}, candidates=cands, masks=masks, observation_id=8, endpoint_text='the door'
        )['valid'])
        self.assertFalse(vcb.resolve_candidate_region(
            {'candidate_id': cid}, candidates=cands, masks=masks, observation_id=9, endpoint_text='the door'
        )['valid'])
        self.assertFalse(vcb.resolve_candidate_region(
            {'candidate_id': cid}, candidates=cands, masks=masks, observation_id=True, endpoint_text='the door'
        )['valid'])
        self.assertFalse(vcb.resolve_candidate_region(
            {'candidate_id': cid, 'view': 'front'}, candidates=cands, masks=masks, observation_id=8, endpoint_text='the door'
        )['valid'])
        self.assertFalse(vcb.resolve_candidate_region(
            {'candidate_id': cid, 'entity_id': 4}, candidates=cands, masks=masks, observation_id=8, endpoint_text='the door'
        )['valid'])

        cab_cid, cab_rec = _make_candidate(obs_id=8, view='right', idx=2, label='cabinet door',
                                           bbox=[0.2, 0.2, 0.5, 0.8], image_size=[100, 80], entity_id=5)
        self.assertFalse(vcb.resolve_candidate_region(
            {'candidate_id': cab_cid}, candidates={cab_cid: cab_rec}, masks={cab_rec['mask_key']: mask},
            observation_id=8, endpoint_text='go to the door'
        )['valid'])

        wrong_mask = np.zeros((80, 100), dtype=bool)
        wrong_mask[0:10, 0:10] = True
        self.assertFalse(vcb.resolve_candidate_region(
            {'candidate_id': cid}, candidates=cands, masks={rec['mask_key']: wrong_mask},
            observation_id=8, endpoint_text='the door'
        )['valid'])

        other_cid, _ = _make_candidate(obs_id=8, view='right', idx=2, label='door',
                                       bbox=[0.2, 0.2, 0.5, 0.8], image_size=[100, 80], entity_id=6)
        wrong_key_rec = dict(rec, mask_key=other_cid)
        self.assertFalse(vcb.resolve_candidate_region(
            {'candidate_id': cid}, candidates={cid: wrong_key_rec},
            masks={cid: mask, other_cid: mask}, observation_id=8, endpoint_text='the door'
        )['valid'])

        missing_cid_rec = dict(rec)
        del missing_cid_rec['candidate_id']
        self.assertFalse(vcb.resolve_candidate_region(
            {'candidate_id': cid}, candidates={cid: missing_cid_rec},
            masks=masks, observation_id=8, endpoint_text='the door'
        )['valid'])

        neg_entity_rec = dict(rec, entity_id=-1)
        self.assertFalse(vcb.resolve_candidate_region(
            {'candidate_id': cid}, candidates={cid: neg_entity_rec},
            masks=masks, observation_id=8, endpoint_text='the door'
        )['valid'])

    def test_no_3d_candidate_and_endpoint_role_semantics(self):
        cid, rec = _make_candidate(obs_id=6, view='back', idx=0, label='sofa',
                                   bbox=[0.3, 0.3, 0.7, 0.7], image_size=[100, 80], entity_id=None, distance=None)
        mask = np.zeros((80, 100), dtype=bool)
        mask[24:56, 30:70] = True
        masks = {rec['mask_key']: mask}

        res = vcb.resolve_candidate_region(
            {'candidate_id': cid}, candidates={cid: rec}, masks=masks,
            observation_id=6, endpoint_text='stop at the couch'
        )
        self.assertTrue(res['valid'])
        self.assertIsNone(res['entity_id'])
        self.assertNotIn('horizontal_distance_m', res)
        self.assertNotIn('world_center', res)

        claimed_entity = vcb.resolve_candidate_region(
            {'candidate_id': cid, 'entity_id': 1}, candidates={cid: rec}, masks=masks,
            observation_id=6, endpoint_text='stop at the couch'
        )
        self.assertFalse(claimed_entity['valid'])

        wall_beside_sofa = vcb.resolve_candidate_region(
            {'candidate_id': cid}, candidates={cid: rec}, masks=masks,
            observation_id=6, endpoint_text='stop at the wall beside the sofa', evidence_role='primary'
        )
        self.assertFalse(wall_beside_sofa['valid'])

        chair_cid, chair_rec = _make_candidate(obs_id=6, view='back', idx=1, label='armchair',
                                               bbox=[0.3, 0.3, 0.7, 0.7], image_size=[100, 80], entity_id=None)
        between_text = 'stand between the couch and the chair'
        primary_res = vcb.resolve_candidate_region(
            {'candidate_id': cid}, candidates={cid: rec}, masks=masks,
            observation_id=6, endpoint_text=between_text, evidence_role='primary'
        )
        ref_res = vcb.resolve_candidate_region(
            {'candidate_id': chair_cid}, candidates={chair_cid: chair_rec}, masks={chair_rec['mask_key']: mask},
            observation_id=6, endpoint_text=between_text, evidence_role='reference'
        )
        self.assertEqual(primary_res['valid'], target_identity.detector_matches_endpoint('sofa', between_text, 'primary'))
        self.assertEqual(ref_res['valid'], target_identity.detector_matches_endpoint('armchair', between_text, 'reference'))
        self.assertTrue(primary_res['valid'])
        self.assertTrue(ref_res['valid'])


    def test_huge_numeric_inputs_fail_closed_without_overflow(self):
        huge = 10 ** 1000
        cid, rec = _make_candidate(obs_id=2, view='front', idx=0, label='door',
                                   bbox=[0.2, 0.2, 0.6, 0.8], image_size=[100, 80], entity_id=1)
        mask = np.zeros((80, 100), dtype=bool)
        mask[16:64, 20:60] = True
        masks = {cid: mask}
        raw = np.zeros((80, 100, 3), dtype=np.uint8)

        huge_bbox_rec = dict(rec, bbox=[0.2, 0.2, huge, 0.8])
        self.assertFalse(vcb.resolve_candidate_region(
            {'candidate_id': cid}, candidates={cid: huge_bbox_rec}, masks=masks,
            observation_id=2, endpoint_text='the door'
        )['valid'])
        self.assertFalse(vcb.resolve_candidate_region(
            {'candidate_id': cid, 'bbox': [0.2, 0.2, huge, 0.8], 'bbox_space': 'full_view'},
            candidates={cid: rec}, masks=masks, observation_id=2, endpoint_text='the door'
        )['valid'])
        self.assertEqual(vcb.select_candidate_crops({cid: huge_bbox_rec}, {'front': raw}, observation_id=2), [])

        huge_conf_rec = dict(rec, confidence=huge)
        self.assertFalse(vcb.resolve_candidate_region(
            {'candidate_id': cid}, candidates={cid: huge_conf_rec}, masks=masks,
            observation_id=2, endpoint_text='the door'
        )['valid'])
        self.assertEqual(vcb.select_candidate_crops({cid: huge_conf_rec}, {'front': raw}, observation_id=2), [])

        self.assertEqual(vcb.select_candidate_crops({cid: rec}, {'front': raw}, observation_id=2, padding=huge), [])
        self.assertEqual(vcb.select_candidate_crops({cid: rec}, {'front': raw}, observation_id=2, padding=1e308), [])


if __name__ == '__main__':
    unittest.main()

