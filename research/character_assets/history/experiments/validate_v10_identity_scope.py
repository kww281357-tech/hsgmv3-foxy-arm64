import bpy,json,hashlib,numpy as np
from pathlib import Path
P=Path(__file__).resolve().parent;O=P/'v10';r={}
for label,path in [('before',P/'v9/editable_character_v9_D.blend'),('after',O/'editable_character_v10_H_oral.blend')]:
 bpy.ops.wm.open_mainfile(filepath=str(path),load_ui=False,use_scripts=False);meshes={}
 for ob in bpy.context.scene.objects:
  if ob.type!='MESH':continue
  keys=ob.data.shape_keys.key_blocks if ob.data.shape_keys else []
  meshes[ob.name]={'vertices':len(ob.data.vertices),'coordinates_sha256':hashlib.sha256(np.array([v.co[:] for v in ob.data.vertices],dtype='<f4').tobytes()).hexdigest(),'polygons_sha256':hashlib.sha256(str([tuple(f.vertices) for f in ob.data.polygons]).encode()).hexdigest(),'shape_keys':[(k.name,k.value,hashlib.sha256(np.array([v.co[:] for v in k.data],dtype='<f4').tobytes()).hexdigest()) for k in keys]}
 rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE');r[label]={'meshes':meshes,'bones':{b.name:[list(row) for row in b.matrix_basis] for b in rig.pose.bones}}
a=r['before']['meshes'];b=r['after']['meshes'];removed=set(a)-set(b);added=set(b)-set(a);assert removed=={'Adult_Expression_Base.eyelashes02'};assert added=={'v10_Mindfront_CC0_lashes_03'}
for name in set(a)&set(b):
 for k in ['vertices','coordinates_sha256','polygons_sha256']:assert a[name][k]==b[name][k]
 assert a[name]['shape_keys']==b[name]['shape_keys'][:len(a[name]['shape_keys'])]
 if len(a[name]['shape_keys'])!=len(b[name]['shape_keys']):assert name.endswith('.teeth_base') and b[name]['shape_keys'][-1][0]=='v10_lower_anterior_oral_clearance' and b[name]['shape_keys'][-1][1]==0
assert r['before']['bones']==r['after']['bones']
(O/'identity_scope_validation.json').write_text(json.dumps({'all_common_original_mesh_coordinates_and_topology_unchanged':True,'all_original_shape_keys_and_neutral_values_unchanged':True,'all_bone_basis_unchanged':True,'only_replaced_meshes':{'removed':sorted(removed),'added':sorted(added)},'only_added_corrective':'teeth lower anterior oral clearance, neutral 0','new_lash_shape_keys':len(b['v10_Mindfront_CC0_lashes_03']['shape_keys'])},indent=2));print('V10_IDENTITY_HAND_SCOPE_PASS')
