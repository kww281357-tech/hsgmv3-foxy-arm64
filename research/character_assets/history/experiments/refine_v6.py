"""CC0 geometry iteration on the same editable v5 identity; offline Blender only."""
import bpy,json,math
import numpy as np
from pathlib import Path
from mathutils import Vector, Quaternion
from mathutils.bvhtree import BVHTree
P=Path(__file__).resolve().parent;O=P/'v6';O.mkdir(exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(P/'output/editable_character_v5.blend'),load_ui=False,use_scripts=False);scene=bpy.context.scene
expr={'neutral':{'mouthClose':.10,'eyeBlinkLeft':.12,'eyeBlinkRight':.105,'eyeSquintLeft':.06,'eyeSquintRight':.06},'smile':{'mouthSmileLeft':.72,'mouthSmileRight':.72,'cheekSquintLeft':.50,'cheekSquintRight':.50,'eyeSquintLeft':.20,'eyeSquintRight':.20,'jawOpen':.20,'mouthUpperUpLeft':.24,'mouthUpperUpRight':.24,'mouthLowerDownLeft':.04,'mouthLowerDownRight':.04},'cry':{'browInnerUp':.72,'browDownLeft':.24,'browDownRight':.24,'mouthFrownLeft':.72,'mouthFrownRight':.72,'mouthShrugLower':.45,'eyeBlinkLeft':.25,'eyeBlinkRight':.25,'eyeSquintLeft':.22,'eyeSquintRight':.22,'jawOpen':.12}}
def setkeys(weights):
 for o in scene.objects:
  if o.type=='MESH' and o.data.shape_keys:
   for k in o.data.shape_keys.key_blocks:
    if k.name!='Basis' and not k.name.startswith('$'):k.value=weights.get(k.name,0)
 bpy.context.view_layer.update()
setkeys(expr['neutral']);deps=bpy.context.evaluated_depsgraph_get();hair=next(o for o in scene.objects if o.name.endswith('.ponytail01'));bvh=BVHTree.FromObject(hair,deps);center=Vector((0,-.005,1.41));rng=np.random.default_rng(621);created=[]
mats=[]
for i,col in enumerate([(.028,.014,.008),(.037,.019,.010),(.047,.024,.012),(.021,.010,.005)]):
 m=bpy.data.materials.new('v6 individual groom '+str(i));m.use_nodes=True;n=next(n for n in m.node_tree.nodes if n.type=='BSDF_PRINCIPLED');n.inputs['Base Color'].default_value=(*col,1);n.inputs['Roughness'].default_value=.61;n.inputs['Specular IOR Level'].default_value=.24;mats.append(m)
def strand(points,width,index):
 cu=bpy.data.curves.new('v6 tapered groom','CURVE');cu.dimensions='3D';cu.bevel_depth=width;cu.bevel_resolution=0;s=cu.splines.new('POLY');s.points.add(len(points)-1)
 for j,(pt,co) in enumerate(zip(s.points,points)):pt.co=(*co,1);pt.radius=max(.08,math.sin(math.pi*(j+.5)/len(points))**.3)
 ob=bpy.data.objects.new('v6_strand',cu);scene.collection.objects.link(ob);ob.data.materials.append(mats[index%4]);bpy.ops.object.select_all(action='DESELECT');ob.select_set(True);bpy.context.view_layer.objects.active=ob;bpy.ops.object.convert(target='MESH');created.append(bpy.context.object)
# Dense continuous root-to-crown bundles span the original wide cap strips.
for i in range(360):
 x=-.93+1.86*i/359;raw=[];normals=[]
 for j in range(30):
  t=.16+1.83*j/29;d=Vector((x*math.cos(t*.25),-math.sqrt(1-x*x)*math.cos(t),math.sqrt(1-x*x)*math.sin(t))).normalized();hit,norm,_,_=bvh.ray_cast(center+d*.35,-d,.5)
  if hit is not None and hit.z>1.428:raw.append(np.array(hit));normals.append(np.array(norm))
 if len(raw)<10:continue
 arr=np.array(raw);smooth=arr.copy()
 for _ in range(3):smooth[1:-1]=(smooth[:-2]+2*smooth[1:-1]+smooth[2:])/4
 points=[]
 for j,co in enumerate(smooth):
  # Millimetre-scale lift bridges piece boundaries; width remains submillimetre.
  lift=.0012+.0009*math.sin(math.pi*j/max(1,len(smooth)-1))
  points.append(Vector(co+np.array(normals[j])*lift))
 strand(points,.00012+float(rng.uniform(0,.000065)),i)
# A few small temple wisps soften the geometric silhouette without changing face shape.
for side in [-1,1]:
 for i in range(14):
  x=side*(.057+i*.00035);points=[]
  for j in range(20):
   t=j/19;points.append(Vector((x+side*.003*math.sin(t*math.pi),-.108+.014*t+.001*math.sin(t*5+i),1.442-.058*t-.002*i/14)))
  strand(points,.000075,i)
if created:
 bpy.ops.object.select_all(action='DESELECT')
 for ob in created:ob.select_set(True)
 bpy.context.view_layer.objects.active=created[0];bpy.ops.object.join();ob=bpy.context.object;ob.name='v6_continuous_hair_groom';rig=next(o for o in scene.objects if o.type=='ARMATURE');vg=ob.vertex_groups.new(name='head');vg.add(list(range(len(ob.data.vertices))),1.,'REPLACE');arm=ob.modifiers.new('Follow head','ARMATURE');arm.object=rig
# True articulated hand refinement: gentle wrist alignment and nonuniform finger cascade.
rig=next(o for o in scene.objects if o.type=='ARMATURE')
hand_report={}
for side in ['l','r']:
 forearm=rig.pose.bones['lowerarm_'+side];fm=forearm.matrix.copy();fr=Quaternion((1,0,0),.45).to_matrix().to_4x4()@fm;fr.translation=fm.translation;forearm.matrix=fr
 bpy.context.view_layer.update()
 wrist=rig.pose.bones['hand_'+side];before=wrist.matrix.copy();adjust=Quaternion((1,0,0),.20).to_matrix().to_4x4();posed=adjust@before;posed.translation=before.translation;wrist.matrix=posed
 for finger,angles in {'index':(.10,.15,.07),'middle':(.17,.23,.10),'ring':(.23,.29,.13),'pinky':(.27,.32,.15),'thumb':(.10,.16,.08)}.items():
  for k,angle in enumerate(angles,1):
   b=rig.pose.bones[f'{finger}_{k:02d}_{side}'];b.rotation_mode='XYZ';b.rotation_euler.x=angle
bpy.context.view_layer.update()
for b in rig.pose.bones:
 if b.name.startswith(('hand_','index_','middle_','ring_','pinky_','thumb_')):
  hand_report[b.name]={'head':list(rig.matrix_world@b.head),'tail':list(rig.matrix_world@b.tail)}
(O/'hand_pose_after.json').write_text(json.dumps(hand_report,indent=2))
report={};topology=[]
for name,weights in expr.items():
 setkeys(weights);deps=bpy.context.evaluated_depsgraph_get();originals=[o for o in scene.objects if o.type=='MESH'];bpy.ops.object.select_all(action='DESELECT');copies=[]
 for o in originals:
  data=bpy.data.meshes.new_from_object(o.evaluated_get(deps),preserve_all_data_layers=True,depsgraph=deps);ob=bpy.data.objects.new(o.name+'_'+name,data);scene.collection.objects.link(ob);ob.matrix_world=o.matrix_world.copy();copies.append(ob);ob.select_set(True)
 floor=min((ob.matrix_world@v.co).z for ob in copies for v in ob.data.vertices)
 for ob in copies:ob.location.z-=floor
 bpy.ops.export_scene.gltf(filepath=str(O/('adult_'+name+'.glb')),export_format='GLB',use_selection=True,export_animations=False)
 topo=[(len(o.data.vertices),tuple(tuple(f.vertices) for f in o.data.polygons)) for o in copies];topology.append(topo)
 report[name]={'vertices':sum(len(o.data.vertices) for o in copies),'triangles':sum(len(f.vertices)-2 for o in copies for f in o.data.polygons),'meshes':len(copies)}
 for ob in copies:bpy.data.objects.remove(ob,do_unlink=True)
 conf=json.loads((P/'output/adult_neutral_refined.object_config.json').read_text());conf['front']=[0,0,-1];conf['render_asset']='adult_'+name+'.glb';conf['user_defined']['expression']=name;conf['user_defined']['integration_status']='not yet Habitat runtime tested';(O/('adult_'+name+'.object_config.json')).write_text(json.dumps(conf,indent=2))
assert topology[0]==topology[1]==topology[2]
setkeys(expr['neutral']);bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(O/'editable_character_v6.blend'))
(O/'expressions.json').write_text(json.dumps(expr,indent=2));(O/'build_metrics.json').write_text(json.dumps({'expressions':report,'same_evaluated_topology_all_expressions':True,'groom_strands':len(created)},indent=2));print('V6_ALL_BUILT',report)
