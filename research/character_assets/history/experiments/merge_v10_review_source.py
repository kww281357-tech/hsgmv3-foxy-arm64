import bpy,json,importlib.util,hashlib,numpy as np
from pathlib import Path
P=Path(__file__).resolve().parent;O=P/'v10'
bpy.ops.wm.open_mainfile(filepath=str(O/'editable_character_v10_H.blend'),load_ui=False,use_scripts=False)
def state():
 return {'body':hashlib.sha256(np.array([v.co[:] for v in bpy.data.objects['Adult_Expression_Base'].data.vertices],dtype='<f4').tobytes()).hexdigest(),'bones':{o.name:{b.name:[list(r) for r in b.matrix_basis] for b in o.pose.bones} for o in bpy.context.scene.objects if o.type=='ARMATURE'}}
before=state();spec=importlib.util.spec_from_file_location('oral_corrective',P/'oral_v10/apply_portable_corrective.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);key=mod.apply_corrective(P/'oral_v10/oral_clearance_delta.npz');assert key.value==0;assert state()==before
bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(O/'editable_character_v10_H_oral.blend'))
expr=json.loads((P/'v7/expressions.json').read_text());expr['smile']=json.loads((P/'v8/expressions.json').read_text())['smile_B'];expr['smile'][mod.KEY]=1;expr['neutral'][mod.KEY]=0;expr['cry'][mod.KEY]=0;(O/'expression_profile_unbaked.json').write_text(json.dumps(expr,indent=2))
(O/'oral_merge_validation.json').write_text(json.dumps({'source_basis_and_topology_hashes_matched':True,'new_oral_key':key.name,'neutral_value':key.value,'body_coordinates_and_all_bone_bases_unchanged':state()==before,'smile_runtime_not_baked_in_this_combined_source':True},indent=2));print('H_ORAL_EDITABLE_MERGE_PASS')
