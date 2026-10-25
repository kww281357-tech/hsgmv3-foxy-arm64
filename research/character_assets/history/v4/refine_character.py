"""v4 secondary-development adapter for the editable MPFB v3 character.
Usage: blender -b --python refine_character.py -- --source V3.blend --expressions expressions.json --out DIR
No external downloads, subprocesses, or network requests. New adapter code: CC0-1.0.
"""
import bpy,sys,os,json,argparse,math
import numpy as np
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--expressions',required=True);p.add_argument('--out',required=True);a=p.parse_args(args)
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);tex=out/'textures';tex.mkdir(exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=a.source);scene=bpy.context.scene
for o in list(scene.objects):
 if o.type in ['CAMERA','LIGHT']:bpy.data.objects.remove(o,do_unlink=True)
meshes=[o for o in scene.objects if o.type=='MESH'];base=next(o for o in meshes if o.name=='Adult_Expression_Base');rig=next(o for o in scene.objects if o.type=='ARMATURE')
changes=[]
def bsdf(m):return next(n for n in m.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
def image_from_array(name,data):
 h,w=data.shape[:2];im=bpy.data.images.new(name,width=w,height=h,alpha=False);im.colorspace_settings.name='Non-Color';rgba=np.ones((h,w,4),np.float32);rgba[:,:,:3]=data;im.pixels.foreach_set(rgba.ravel());im.filepath_raw=str(tex/(name+'.png'));im.file_format='PNG';im.save();im.colorspace_settings.name='Non-Color';im.pack();return im
def detail_material(m,label,strength,roughness):
 n=512 if label=='hair' else 1024;rng=np.random.default_rng(3407 if label=='skin' else 811)
 if label=='hair':
  yy,xx=np.mgrid[:n,:n];height=(np.sin(xx*.7+np.sin(yy*.017)*1.5)+.35*np.sin(xx*1.7+yy*.012))*.3
 else:
  height=rng.normal(0,1,(n,n));height=(height+np.roll(height,1,0)+np.roll(height,-1,0)+np.roll(height,1,1)+np.roll(height,-1,1))/5
 dy,dx=np.gradient(height);normal=np.stack((-dx*strength,-dy*strength,np.ones_like(height)),axis=-1);normal/=np.linalg.norm(normal,axis=-1,keepdims=True);normal=(normal+1)*.5
 im=image_from_array(label+'_micro_normal',normal);nt=m.node_tree;ntex=nt.nodes.new('ShaderNodeTexImage');ntex.name=label+'_export_normal';ntex.image=im;nm=nt.nodes.new('ShaderNodeNormalMap');nt.links.new(ntex.outputs['Color'],nm.inputs['Color']);nm.inputs['Strength'].default_value=1;nt.links.new(nm.outputs['Normal'],bsdf(m).inputs['Normal'])
 rough=np.clip(roughness+height*.025,.1,.9);imr=image_from_array(label+'_roughness',np.repeat(rough[:,:,None],3,axis=2));rt=nt.nodes.new('ShaderNodeTexImage');rt.name=label+'_export_roughness';rt.image=imr;nt.links.new(rt.outputs['Color'],bsdf(m).inputs['Roughness'])
for m in bpy.data.materials:
 if not m.use_nodes:continue
 node=bsdf(m)
 if m.name.endswith('.body'):
  node.inputs['Subsurface Weight'].default_value=.065;node.inputs['Subsurface Scale'].default_value=.035;node.inputs['Specular IOR Level'].default_value=.3;detail_material(m,'skin',.07,.47)
 elif m.name.endswith('.high-poly'):node.inputs['Roughness'].default_value=.16;node.inputs['Specular IOR Level'].default_value=.5
 elif m.name.endswith('.ponytail01'):detail_material(m,'hair',.075,.57);node.inputs['Specular IOR Level'].default_value=.3
 elif m.name.endswith('.shoes03'):node.inputs['Roughness'].default_value=.44;node.inputs['Specular IOR Level'].default_value=.32
 elif m.name.endswith('.teeth_base'):node.inputs['Roughness'].default_value=.3
 elif m.name.endswith('.female_elegantsuit01'):node.inputs['Roughness'].default_value=.85;node.inputs['Specular IOR Level'].default_value=.24
changes+=['Skin: exported tangent-space micro-normal and roughness textures, restrained SSS in editable source','Eyes: roughness 0.70 -> 0.16','Hair: directional normal/roughness textures','Shoes: roughness 0.20 -> 0.44; reduced plastic highlight']
# Slight finger curl, same pose for all expressions. No face identity/weight changes.
for side in ['l','r']:
 for finger in ['index','middle','ring','pinky']:
  for segment,ang in [(1,.12),(2,.19),(3,.10)]:
   b=rig.pose.bones.get(f'{finger}_{segment:02d}_{side}')
   if b:b.rotation_mode='XYZ';b.rotation_euler.x+=ang
changes.append('Body rig: gentle proximal/middle/distal finger curl, preserved face proportions and stance')
cloth=next(o for o in meshes if o.name.endswith('.female_elegantsuit01'))
solid=cloth.modifiers.new('v4 garment edge thickness','SOLIDIFY');solid.thickness=.0009;solid.offset=0;solid.use_even_offset=False
bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get()
cloth_bvh=BVHTree.FromObject(cloth,deps)
def basicmat(name,color,rough):
 m=bpy.data.materials.new(name);m.use_nodes=True;n=bsdf(m);n.inputs['Base Color'].default_value=(*color,1);n.inputs['Roughness'].default_value=rough;return m
buttonmat=basicmat('v4 ivory resin buttons',(.48,.44,.38),.32)
evaluated=cloth.evaluated_get(deps);cm=evaluated.to_mesh();cm.calc_loop_triangles();uvs=cm.uv_layers.active.data
def uv_position(uv):
 q=np.array(uv)
 for tr in cm.loop_triangles:
  v=np.array([uvs[i].uv[:] for i in tr.loops]);mat=np.column_stack((v[1]-v[0],v[2]-v[0]))
  if abs(np.linalg.det(mat))<1e-10:continue
  w=np.linalg.solve(mat,q-v[0])
  if min(w)>=-1e-5 and sum(w)<=1.00001:
   pos=sum((cm.vertices[idx].co*float(weight) for idx,weight in zip(tr.vertices,[1-sum(w),w[0],w[1]])),Vector())
   return cloth.matrix_world@pos, tr.normal
 return None,None
for idx,(u,iy) in enumerate([(194/1600,1229/1600),(193/1600,1280/1600),(195/1600,1326/1600),(199/1600,1481/1600)]):
 loc,norm=uv_position((u,1-iy))
 if loc is None:continue
 bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=8,radius=1,location=loc+Vector((0,-.0011,0)));o=bpy.context.object;o.name=f'v4_button_{idx+1}';o.scale=(.0028,.0009,.0028);bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);o.data.materials.append(buttonmat)
 for f in o.data.polygons:f.use_smooth=True
 print('BUTTON_UV_MATCH',idx,list(loc))
evaluated.to_mesh_clear()
changes.append('Garment: 0.9 mm physical edge thickness and four UV-aligned raised resin buttons')
# Deliberate groom-following strands lie just above the actual hair surface, not random fuzz.
hair=next(o for o in meshes if o.name.endswith('.ponytail01'));hbvh=BVHTree.FromObject(hair,deps);center=Vector((0,-.005,1.41));rng=np.random.default_rng(934)
strandmats=[basicmat('v4 hair strand '+str(i),color,.47) for i,color in enumerate([(.045,.020,.011),(.055,.026,.014),(.065,.032,.018)])]
strandcount=0
for i in range(48):
 x=-.74+1.48*i/47;points=[]
 for j in range(18):
  t=.28+1.45*j/17;direction=Vector((x*math.cos(t*.25),-math.sqrt(1-x*x)*math.cos(t),math.sqrt(1-x*x)*math.sin(t))).normalized();hit,norm,_,_=hbvh.ray_cast(center+direction*.35,-direction,.5)
  if hit is not None and hit.z>1.435:points.append(hit+norm*.00035)
 if len(points)<5:continue
 cu=bpy.data.curves.new('v4_groom_strand','CURVE');cu.dimensions='3D';cu.resolution_u=1;cu.bevel_depth=.000055+float(rng.uniform(0,.000035));cu.bevel_resolution=1;sp=cu.splines.new('POLY');sp.points.add(len(points)-1)
 for pt,co in zip(sp.points,points):pt.co=(*co,1)
 ob=bpy.data.objects.new('v4_groom_strand',cu);scene.collection.objects.link(ob);ob.data.materials.append(strandmats[i%3]);bpy.context.view_layer.objects.active=ob;bpy.ops.object.select_all(action='DESELECT');ob.select_set(True);bpy.ops.object.convert(target='MESH');strandcount+=1
changes.append(f'Hair: {strandcount} surface-following fine strand meshes, portable GLB geometry')
# Consolidate detail meshes to avoid one draw node per strand/button.
for prefix,label,bone in [('v4_groom_strand','v4_hair_detail','head'),('v4_button_','v4_buttons','spine_02')]:
 selected=[ob for ob in scene.objects if ob.type=='MESH' and ob.name.startswith(prefix)]
 if selected:
  bpy.ops.object.select_all(action='DESELECT')
  for ob in selected:ob.select_set(True)
  bpy.context.view_layer.objects.active=selected[0];bpy.ops.object.join();joined=bpy.context.object;joined.name=label
  vg=joined.vertex_groups.new(name=bone);vg.add(list(range(len(joined.data.vertices))),1.0,'REPLACE');arm=joined.modifiers.new('v4 follow body rig','ARMATURE');arm.object=rig

# Compact opaque textures with high-quality JPEG; alpha-dependent hair/brow/lash stay PNG.
for m in bpy.data.materials:
 if not m.use_nodes or any(s in m.name for s in ['ponytail','eyebrow','eyelashes','high-poly']):continue
 for n in list(m.node_tree.nodes):
  if n.type!='TEX_IMAGE' or n.name!='DiffuseTexture':continue
  old=n.image;path=tex/(old.name.rsplit('.',1)[0]+'.jpg');old.file_format='JPEG';old.filepath_raw=str(path);old.save();new=bpy.data.images.load(str(path),check_existing=True);new.pack();n.image=new
  for link in list(m.node_tree.links):
   if link.to_node.type=='BSDF_PRINCIPLED' and link.to_socket.name=='Alpha':m.node_tree.links.remove(link)
changes.append('Packaging: JPEG copies for opaque base-color textures; lossless alpha maps remain unchanged')
expr=json.load(open(a.expressions));(out/'expressions.json').write_text(json.dumps(expr,indent=2));(out/'source_changes.json').write_text(json.dumps(changes,indent=2))
allmesh=[o for o in scene.objects if o.type=='MESH'];report={}
for name,weights in expr.items():
 for o in allmesh:
  if o.data.shape_keys:
   for key in o.data.shape_keys.key_blocks:
    if key.name=='Basis' or key.name.startswith('$'):continue
    key.value=weights.get(key.name,0)
 bpy.context.view_layer.update();bpy.ops.object.select_all(action='DESELECT');copies=[]
 for o in allmesh:
  m=bpy.data.meshes.new_from_object(o.evaluated_get(deps),preserve_all_data_layers=True,depsgraph=deps);obj=bpy.data.objects.new(o.name+'_'+name,m);scene.collection.objects.link(obj);obj.matrix_world=o.matrix_world.copy();copies.append(obj);obj.select_set(True)
 points=[ob.matrix_world@v.co for ob in copies for v in ob.data.vertices];floor=min(v.z for v in points)
 for ob in copies:ob.location.z-=floor
 bpy.ops.export_scene.gltf(filepath=str(out/f'adult_{name}.glb'),export_format='GLB',use_selection=True,export_animations=False,export_extras=True)
 report[name]={'triangles':sum(len(poly.vertices)-2 for ob in copies for poly in ob.data.polygons),'mesh_count':len(copies),'height_m':max(v.z for v in points)-floor}
 for ob in copies:bpy.data.objects.remove(ob,do_unlink=True)
for o in allmesh:
 if o.data.shape_keys:
  for key in o.data.shape_keys.key_blocks:
   if key.name=='Basis' or key.name.startswith('$'):continue
   key.value=expr['neutral'].get(key.name,0)

for image in list(bpy.data.images):
 if image.users==0:bpy.data.images.remove(image)
bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(out/'expression_master_v4.blend'));(out/'geometry_validation.json').write_text(json.dumps(report,indent=2))
print('V4_COMPLETE',report)
