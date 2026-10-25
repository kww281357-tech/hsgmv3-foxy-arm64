import bpy,json,math,bmesh,numpy as np
from pathlib import Path
P=Path(__file__).resolve().parent;O=P/'v10';A=O/'verified_cc0_assets/eyelashes01/eyelashes/mindfront_eyelashes_03'
bpy.ops.wm.open_mainfile(filepath=str(P/'v9/editable_character_v9_D.blend'),load_ui=False,use_scripts=False);scene=bpy.context.scene
body=bpy.data.objects['Adult_Expression_Base'];keys=body.data.shape_keys.key_blocks;seed=np.array([v.co[:] for v in keys[0].data]);identity=seed.copy()
for k in list(keys)[1:]:
 if k.name.startswith('$'):identity+=(np.array([v.co[:] for v in k.data])-seed)*k.value
refs=[];weights=[];offsets=[];scales={};parsing=False
for raw in (A/'mindfront_eyelashes_03.mhclo').read_text().splitlines():
 row=raw.strip().split()
 if not row or row[0].startswith('#'):continue
 if row[0] in ['x_scale','y_scale','z_scale']:scales[row[0]]=(int(row[1]),int(row[2]),float(row[3]))
 if row[0]=='verts':parsing=True;continue
 if parsing:
  if len(row)==9:
   try:refs.append([int(x) for x in row[:3]]);weights.append([float(x) for x in row[3:6]]);offsets.append([float(x) for x in row[6:9]])
   except ValueError:break
  else:break
refs=np.array(refs);weights=np.array(weights);offsets=np.array(offsets);assert refs.max()<len(seed)
scale=[]
for key,axis in [('x_scale',0),('y_scale',2),('z_scale',1)]:
 a,b,dist=scales[key];scale.append(abs(identity[a,axis]-identity[b,axis])/(dist*.1))
off=offsets*np.array(scale)*.1;off=np.stack([off[:,0],-off[:,2],off[:,1]],axis=1)
def fit(co):return (co[refs]*weights[:,:,None]).sum(1)+off
coords=fit(identity);faces=[];obj_verts=0
for line in (A/'mind_eyelashes_03.obj').read_text().splitlines():
 if line.startswith('v '):obj_verts+=1
 if line.startswith('f '):faces.append([int(s.split('/')[0])-1 for s in line.split()[1:]])
assert len(coords)==obj_verts
mesh=bpy.data.meshes.new('Mindfront_CC0_lashes_03_adapted');mesh.from_pydata(coords.tolist(),[],faces);mesh.update();lash=bpy.data.objects.new('v10_Mindfront_CC0_lashes_03',mesh);scene.collection.objects.link(lash)
base=lash.shape_key_add(name='Basis',from_mix=False)
for k in list(keys)[1:]:
 if k.name.startswith('$'):continue
 dest=lash.shape_key_add(name=k.name,from_mix=False);delta=np.array([v.co[:] for v in k.data])-seed;new=coords+(delta[refs]*weights[:,:,None]).sum(1)
 for v,co in zip(dest.data,new):v.co=co
 dest.value=k.value
mat=bpy.data.materials.new('v10_natural_individual_lash_dark_brown');mat.use_nodes=True;node=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED');node.inputs['Base Color'].default_value=(.012,.007,.004,1);node.inputs['Roughness'].default_value=.74;mesh.materials.append(mat)
for f in mesh.polygons:f.use_smooth=True
rig=next(o for o in scene.objects if o.type=='ARMATURE');vg=lash.vertex_groups.new(name='head');vg.add(list(range(len(mesh.vertices))),1,'REPLACE');arm=lash.modifiers.new('Original head rig','ARMATURE');arm.object=rig
old=next(o for o in scene.objects if o.name.endswith('.eyelashes02'));bpy.data.objects.remove(old,do_unlink=True);bpy.context.view_layer.update()
(O/'lash_asset_H_metrics.json').write_text(json.dumps({'source_asset':'Mindfront eyelashes03','license':'CC0','source_page':'https://static.makehumancommunity.org/assets/assetpacks/eyelashes01.html','vertices':len(mesh.vertices),'faces':len(mesh.polygons),'proxy_ref_scale_xyz_makehuman':scale,'new_lash_shape_keys':len(lash.data.shape_keys.key_blocks),'bounds_base':[coords.min(0).tolist(),coords.max(0).tolist()],'base_identity_and_hand_geometry_unchanged':True},indent=2))
# Export evaluated static full body with fixed diagonals.
deps=bpy.context.evaluated_depsgraph_get();originals=[o for o in scene.objects if o.type=='MESH'];copies=[];bpy.ops.object.select_all(action='DESELECT')
for o in originals:
 data=bpy.data.meshes.new_from_object(o.evaluated_get(deps),preserve_all_data_layers=True,depsgraph=deps);ob=bpy.data.objects.new(o.name+'_H',data);scene.collection.objects.link(ob);ob.matrix_world=o.matrix_world.copy();copies.append(ob);ob.select_set(True);bm=bmesh.new();bm.from_mesh(data);bmesh.ops.triangulate(bm,faces=list(bm.faces),quad_method='FIXED',ngon_method='EAR_CLIP');bm.to_mesh(data);bm.free();data.update()
floor=min((ob.matrix_world@v.co).z for ob in copies for v in ob.data.vertices)
for ob in copies:ob.location.z-=floor
bpy.ops.export_scene.gltf(filepath=str(O/'adult_neutral_H.glb'),export_format='GLB',use_selection=True,export_animations=False)
for ob in copies:bpy.data.objects.remove(ob,do_unlink=True)
bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(O/'editable_character_v10_H.blend'));print('V10_CC0_LASHES_H_BUILT')
