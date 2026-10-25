import bpy,json,math,bmesh
from pathlib import Path
from mathutils import Vector,Quaternion
P=Path(__file__).resolve().parent;O=P/'v7';O.mkdir(exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(P/'v6/editable_character_v6.blend'),load_ui=False,use_scripts=False);scene=bpy.context.scene
expr={'neutral':{'mouthClose':.10,'eyeBlinkLeft':.12,'eyeBlinkRight':.105,'eyeSquintLeft':.06,'eyeSquintRight':.06},'smile':{'mouthSmileLeft':1.0,'mouthSmileRight':.98,'cheekSquintLeft':.95,'cheekSquintRight':.92,'eyeSquintLeft':.50,'eyeSquintRight':.48,'eyeBlinkLeft':.13,'eyeBlinkRight':.12,'jawOpen':.36,'mouthUpperUpLeft':.46,'mouthUpperUpRight':.44,'mouthLowerDownLeft':.10,'mouthLowerDownRight':.10},'cry':{'browInnerUp':.72,'browDownLeft':.24,'browDownRight':.24,'mouthFrownLeft':.72,'mouthFrownRight':.72,'mouthShrugLower':.45,'eyeBlinkLeft':.25,'eyeBlinkRight':.25,'eyeSquintLeft':.22,'eyeSquintRight':.22,'jawOpen':.12}}
def setkeys(weights):
 for o in scene.objects:
  if o.type=='MESH' and o.data.shape_keys:
   for k in o.data.shape_keys.key_blocks:
    if k.name!='Basis' and not k.name.startswith('$'):k.value=weights.get(k.name,0)
 bpy.context.view_layer.update()
# v7 changes restricted to articulated hand bones; v6 face/hair/clothes retained.
rig=next(o for o in scene.objects if o.type=='ARMATURE')
for side,sign in [('l',1),('r',-1)]:
 wrist=rig.pose.bones['hand_'+side];mat=wrist.matrix.copy();rot=Quaternion(mat.to_3x3().col[1],sign*.18).to_matrix().to_4x4();out=rot@mat;out.translation=mat.translation;wrist.matrix=out
 bpy.context.view_layer.update()
 for finger,angles in {'index':(.06,.12,.04),'middle':(.08,.15,.05),'ring':(.10,.18,.06),'pinky':(.12,.20,.07),'thumb':(.03,.07,.03)}.items():
  for k,angle in enumerate(angles,1):
   bone=rig.pose.bones[f'{finger}_{k:02d}_{side}'];bone.rotation_mode='XYZ';bone.rotation_euler.x=angle
 bpy.context.view_layer.update()
 # Adduct first thumb segment toward the longitudinal hand axis, avoiding a wide C-shaped gap.
 thumb=rig.pose.bones['thumb_01_'+side];mat=thumb.matrix.copy();direction=(thumb.tail-thumb.head).normalized();down=(wrist.tail-wrist.head).normalized();target=(direction*.55+down*.45).normalized();rot=direction.rotation_difference(target).to_matrix().to_4x4();out=rot@mat;out.translation=mat.translation;thumb.matrix=out
 bpy.context.view_layer.update()
 # Small index MCP adduction closes the residual finger fan without joining geometry.
 index=rig.pose.bones['index_01_'+side];middle=rig.pose.bones['middle_01_'+side];mat=index.matrix.copy();direction=index.tail-index.head;target=(direction+(middle.head-index.head)*.14).normalized();rot=direction.normalized().rotation_difference(target).to_matrix().to_4x4();out=rot@mat;out.translation=mat.translation;index.matrix=out
 bpy.context.view_layer.update()
(O/'hand_pose_after.json').write_text(json.dumps({b.name:{'head':list(rig.matrix_world@b.head),'tail':list(rig.matrix_world@b.tail)} for b in rig.pose.bones if b.name.startswith(('hand_','index_','middle_','ring_','pinky_','thumb_'))},indent=2))
report={};topology=[]
for name,weights in expr.items():
 setkeys(weights);deps=bpy.context.evaluated_depsgraph_get();originals=[o for o in scene.objects if o.type=='MESH'];bpy.ops.object.select_all(action='DESELECT');copies=[]
 for o in originals:
  data=bpy.data.meshes.new_from_object(o.evaluated_get(deps),preserve_all_data_layers=True,depsgraph=deps);ob=bpy.data.objects.new(o.name+'_'+name,data);scene.collection.objects.link(ob);ob.matrix_world=o.matrix_world.copy();copies.append(ob);ob.select_set(True)
 # Fixed quad diagonals prevent expression-dependent glTF triangulation.
 for ob in copies:
  bm=bmesh.new();bm.from_mesh(ob.data);bmesh.ops.triangulate(bm,faces=list(bm.faces),quad_method='FIXED',ngon_method='EAR_CLIP');bm.to_mesh(ob.data);bm.free();ob.data.update()
 floor=min((ob.matrix_world@v.co).z for ob in copies for v in ob.data.vertices)
 for ob in copies:ob.location.z-=floor
 bpy.ops.export_scene.gltf(filepath=str(O/('adult_'+name+'.glb')),export_format='GLB',use_selection=True,export_animations=False)
 topo=[(len(o.data.vertices),tuple(tuple(f.vertices) for f in o.data.polygons)) for o in copies];topology.append(topo)
 report[name]={'vertices':sum(len(o.data.vertices) for o in copies),'triangles':sum(len(f.vertices)-2 for o in copies for f in o.data.polygons),'meshes':len(copies)}
 for ob in copies:bpy.data.objects.remove(ob,do_unlink=True)
 conf=json.loads((P/'v6/adult_neutral.object_config.json').read_text());conf['front']=[0,0,-1];conf['render_asset']='adult_'+name+'.glb';conf['user_defined']['expression']=name;conf['user_defined']['integration_status']='not yet Habitat runtime tested';(O/('adult_'+name+'.object_config.json')).write_text(json.dumps(conf,indent=2))
assert topology[0]==topology[1]==topology[2]
setkeys(expr['neutral']);bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(O/'editable_character_v7.blend'))
(O/'expressions.json').write_text(json.dumps(expr,indent=2));(O/'build_metrics.json').write_text(json.dumps({'expressions':report,'same_evaluated_topology_all_expressions':True,'modifications':'hand bones only'},indent=2));print('V7_ALL_BUILT',report)
