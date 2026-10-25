import bpy,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;report={}
for version in ['v6','v7']:
 bpy.ops.wm.open_mainfile(filepath=str(P/version/f'editable_character_{version}.blend'),load_ui=False,use_scripts=False)
 meshes={}
 for ob in bpy.context.scene.objects:
  if ob.type!='MESH':continue
  keys=ob.data.shape_keys.key_blocks if ob.data.shape_keys else []
  meshes[ob.name]={'base_vertices':len(ob.data.vertices),'coordinates_sha256':hashlib.sha256(str([tuple(v.co) for v in ob.data.vertices]).encode()).hexdigest(),'shape_keys':[(k.name,k.value,len(k.data),hashlib.sha256(str([tuple(v.co) for v in k.data]).encode()).hexdigest()) for k in keys],'materials':[m.name if m else None for m in ob.data.materials]}
 rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
 report[version]={'meshes':meshes,'bone_basis':{b.name:[list(row) for row in b.matrix_basis] for b in rig.pose.bones}}
assert report['v6']['meshes']==report['v7']['meshes'],'Unexpected mesh/morph/material assignment change'
changed=[n for n,m in report['v6']['bone_basis'].items() if m!=report['v7']['bone_basis'][n]]
assert all(n.startswith(('hand_','index_','middle_','ring_','pinky_','thumb_')) for n in changed),changed
(P/'v7/scope_validation.json').write_text(json.dumps({'base_mesh_shape_keys_and_material_assignments_unchanged':True,'changed_bone_basis_only':changed,'original_meshes':report['v6']['meshes']},indent=2));print('V7_HAND_BONES_ONLY_PASS',changed)
