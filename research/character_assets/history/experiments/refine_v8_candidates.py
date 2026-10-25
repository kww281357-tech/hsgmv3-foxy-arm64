import bpy,json,math,bmesh,numpy as np
from pathlib import Path
P=Path(__file__).resolve().parent;O=P/'v8';O.mkdir(exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(P/'v7/editable_character_v7.blend'),load_ui=False,use_scripts=False);scene=bpy.context.scene
neutral=json.loads((P/'v7/expressions.json').read_text())['neutral']
a={'mouthSmileLeft':1.0,'mouthSmileRight':.98,'cheekSquintLeft':1.0,'cheekSquintRight':.98,'eyeSquintLeft':.78,'eyeSquintRight':.74,'eyeBlinkLeft':.07,'eyeBlinkRight':.065,'jawOpen':.23,'mouthUpperUpLeft':.28,'mouthUpperUpRight':.27,'mouthLowerDownLeft':.035,'mouthLowerDownRight':.035}
expr={'smile_A':a,'smile_B':dict(a,v8_smile_cheek_corrective=1.0)}
def setkeys(weights):
 for o in scene.objects:
  if o.type=='MESH' and o.data.shape_keys:
   for k in o.data.shape_keys.key_blocks:
    if k.name!='Basis' and not k.name.startswith('$'):k.value=weights.get(k.name,0)
 bpy.context.view_layer.update()
# Additive smile-only corrective, neutral original data remains untouched.
o=bpy.data.objects['Adult_Expression_Base'];keys=o.data.shape_keys.key_blocks;base=np.array([v.co[:] for v in keys[0].data]);mix=base.copy()
for k in list(keys)[1:]:mix+=(np.array([v.co[:] for v in k.data])-base)*k.value
x,y,z=mix.T;ax=np.abs(x);side=np.sign(x);front=np.clip((-y-.075)/.025,0,1);delta=np.zeros_like(base)
def g(cx,cz,sx,sz):return np.exp(-.5*((ax-cx)/sx)**2-.5*((z-cz)/sz)**2)*front
cheek=g(.039,1.359,.014,.012);corner=g(.027,1.330,.010,.007)
delta[:,2]+=.0024*cheek+.0018*corner
delta[:,1]-=.0013*cheek
delta[:,0]+=side*(.0010*corner+.00035*cheek)
# Submillimetre paired groove/ridge to articulate nasolabial skin compression.
line=.024+(1.351-z)*.58;length=np.exp(-.5*((z-1.338)/.012)**2)*front
fold=np.exp(-.5*((ax-line)/.0016)**2)*length;ridge=np.exp(-.5*((ax-line-.0030)/.0025)**2)*length
delta[:,1]+=.00060*fold-.00055*ridge
# Strict face-only support, excluding body and posterior/inner face geometry.
mask=(z>1.305)&(z<1.385)&(ax<.068)&(y<-.080);delta[~mask]=0
key=o.shape_key_add(name='v8_smile_cheek_corrective',from_mix=False)
for v,co in zip(key.data,base+delta):v.co=co
key.value=0
(O/'corrective_metrics.json').write_text(json.dumps({'affected_vertices':int(np.count_nonzero(np.linalg.norm(delta,axis=1)>1e-8)),'max_delta_m':float(np.linalg.norm(delta,axis=1).max()),'neutral_value':0,'mesh_topology_unchanged':True},indent=2))
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
 conf=json.loads((P/'v7/adult_neutral.object_config.json').read_text());conf['front']=[0,0,-1];conf['render_asset']='adult_'+name+'.glb';conf['user_defined']['expression']=name;conf['user_defined']['integration_status']='not yet Habitat runtime tested';(O/('adult_'+name+'.object_config.json')).write_text(json.dumps(conf,indent=2))
assert topology[0]==topology[1]
setkeys(neutral);bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(O/'editable_character_v8_candidates.blend'))
(O/'expressions.json').write_text(json.dumps(expr,indent=2));(O/'build_metrics.json').write_text(json.dumps({'expressions':report,'same_evaluated_topology_all_expressions':True,'modifications':'smile-only candidate facial morphs/corrective'},indent=2));print('V8_CANDIDATES_BUILT',report)
