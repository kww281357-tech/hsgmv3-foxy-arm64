"""CC0 authored adapter. Offline CPU Blender; external scripts disabled. One fictional CC0 character."""
import bpy, json, math, argparse, sys
import numpy as np
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--source',default=str(P/'source/expression_master_v4.blend'));ap.add_argument('--expressions',default=str(P/'source/expressions.json'));ap.add_argument('--out',default=str(P/'output'));a=ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
O=Path(a.out).resolve();O.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(Path(a.source).resolve()),load_ui=False,use_scripts=False)
scene=bpy.context.scene
for o in list(scene.objects):
 if o.type in {'LIGHT','CAMERA'}:bpy.data.objects.remove(o,do_unlink=True)
expr=json.loads(Path(a.expressions).read_text());changes=[]
def setkeys(weights):
 for o in scene.objects:
  if o.type=='MESH' and o.data.shape_keys:
   for k in o.data.shape_keys.key_blocks:
    if k.name!='Basis' and not k.name.startswith('$'):k.value=weights.get(k.name,0)
 bpy.context.view_layer.update()
def export(name):
 deps=bpy.context.evaluated_depsgraph_get(); originals=[o for o in scene.objects if o.type=='MESH'];copies=[]
 bpy.ops.object.select_all(action='DESELECT')
 for o in originals:
  m=bpy.data.meshes.new_from_object(o.evaluated_get(deps),preserve_all_data_layers=True,depsgraph=deps);ob=bpy.data.objects.new(o.name+'_'+name,m);scene.collection.objects.link(ob);ob.matrix_world=o.matrix_world.copy();copies.append(ob);ob.select_set(True)
 floor=min((o.matrix_world@v.co).z for o in copies for v in o.data.vertices)
 for ob in copies:ob.location.z-=floor
 bpy.ops.export_scene.gltf(filepath=str(O/(name+'.glb')),export_format='GLB',use_selection=True,export_animations=False)
 stats={'vertices':sum(len(o.data.vertices) for o in copies),'triangles':sum(len(f.vertices)-2 for o in copies for f in o.data.polygons),'meshes':len(copies)}
 for ob in copies:bpy.data.objects.remove(ob,do_unlink=True)
 return stats
setkeys(expr['neutral']);report={'baseline':export('adult_neutral_before')}
def node(m):return next(n for n in m.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
def image_transform(im,name,fn):
 w,h=im.size;px=np.asarray(im.pixels[:],dtype=np.float32).reshape(h,w,4);px=fn(px)
 new=bpy.data.images.new(name,width=w,height=h,alpha=True);new.pixels.foreach_set(np.clip(px,0,1).ravel());new.filepath_raw=str(O/(name+'.png'));new.file_format='PNG';new.save();new.pack();return new
for m in bpy.data.materials:
 if not m.use_nodes:continue
 n=node(m)
 if m.name.endswith('.body'):
  old=m.node_tree.nodes['DiffuseTexture'].image
  def skin(px):
   px[:,:,:3]*=np.array([.94,.89,.86]);return px
  m.node_tree.nodes['DiffuseTexture'].image=image_transform(old,'v5_skin_albedo',skin)
  n.inputs['Subsurface Weight'].default_value=.035;n.inputs['Specular IOR Level'].default_value=.28
  changes.append('Skin albedo: preserve source detail while reducing pale uniform brightness; no painted screen-space highlights')
 elif m.name.endswith('.high-poly'):
  n.inputs['Roughness'].default_value=.09;n.inputs['IOR'].default_value=1.38;n.inputs['Coat Weight'].default_value=.22;n.inputs['Coat Roughness'].default_value=.08
  changes.append('Eyes: roughness .16 -> .09; restrained glTF clearcoat, fallback still uses standard PBR roughness')
 elif m.name.endswith('.ponytail01'):
  old=m.node_tree.nodes['DiffuseTexture'].image
  def hair(px):
   px[:,:,:3]*=np.array([.70,.65,.60]);return px
  new=image_transform(old,'v5_hair_albedo',hair)
  for nd in m.node_tree.nodes:
   if nd.type=='TEX_IMAGE' and nd.image==old:nd.image=new
  n.inputs['Specular IOR Level'].default_value=.22
  for nd in m.node_tree.nodes:
   if nd.type=='NORMAL_MAP':nd.inputs['Strength'].default_value=.65
  changes.append('Hair: darker natural brown, less broad shell-like highlight; existing strand normal strength .65')
 elif m.name.endswith('.female_elegantsuit01'):
  old=m.node_tree.nodes['DiffuseTexture'].image
  def cloth(px):
   lum=px[:,:,:3].mean(2);shade=.78+.22*np.clip(lum,0,1)
   px[:,:,:3]=shade[:,:,None]*np.array([.24,.36,.34]);return px
  m.node_tree.nodes['DiffuseTexture'].image=image_transform(old,'v5_muted_homewear',cloth)
  n.inputs['Roughness'].default_value=.88;n.inputs['Specular IOR Level'].default_value=.18
  changes.append('Homewear: replace high-contrast striped albedo by muted sage fabric; preserve seam normal map and garment geometry')
# Modest upper-lid resting position, propagated to matching dependent eye/brow/lash morphs.
expr['neutral'].update(eyeBlinkLeft=.055,eyeBlinkRight=.045,eyeSquintLeft=.025,eyeSquintRight=.025)
setkeys(expr['neutral']);changes.append('Neutral eyelid geometry: asymmetric 4.5–5.5% blink, 2.5% squint; all original morph topology retained')
# Surface-following fine groom: new narrow tapered strands lift slightly off silhouette, not large tubes.
hair=next(o for o in scene.objects if o.name.endswith('.ponytail01'));deps=bpy.context.evaluated_depsgraph_get();bvh=BVHTree.FromObject(hair,deps);center=Vector((0,-.005,1.41));rng=np.random.default_rng(501);created=[]
mat=bpy.data.materials.new('v5 fine flyaway dark brown');mat.use_nodes=True;node(mat).inputs['Base Color'].default_value=(.025,.012,.006,1);node(mat).inputs['Roughness'].default_value=.66
for i in range(90):
 x=-.91+1.82*i/89;points=[]
 for j in range(16):
  t=.32+1.38*j/15;d=Vector((x*math.cos(t*.25),-math.sqrt(1-x*x)*math.cos(t),math.sqrt(1-x*x)*math.sin(t))).normalized();hit,norm,_,_=bvh.ray_cast(center+d*.35,-d,.5)
  if hit is not None and hit.z>1.435:
   lift=.00035+.0015*math.sin(math.pi*j/15)**2
   points.append(hit+norm*lift)
 if len(points)<6:continue
 cu=bpy.data.curves.new('v5 fine groom','CURVE');cu.dimensions='3D';cu.bevel_depth=.00004;cu.bevel_resolution=0;s=cu.splines.new('POLY');s.points.add(len(points)-1)
 for j,(pt,co) in enumerate(zip(s.points,points)):pt.co=(*co,1);pt.radius=max(.12,math.sin(math.pi*(j+.5)/len(points))**.5)
 ob=bpy.data.objects.new('v5_groom',cu);scene.collection.objects.link(ob);ob.data.materials.append(mat);bpy.ops.object.select_all(action='DESELECT');ob.select_set(True);bpy.context.view_layer.objects.active=ob;bpy.ops.object.convert(target='MESH');created.append(bpy.context.object)
if created:
 bpy.ops.object.select_all(action='DESELECT')
 for ob in created:ob.select_set(True)
 bpy.context.view_layer.objects.active=created[0];bpy.ops.object.join();bpy.context.object.name='v5_fine_hair_geometry'
changes.append(f'Added {len(created)} tapered 0.08mm-diameter fine hair curves converted to one GLB mesh; no opaque random fuzz')
report['refined']=export('adult_neutral_refined')
for im in bpy.data.images:
 if im.size[0]>0 and im.source=='FILE' and not im.packed_file:im.pack()
bpy.ops.wm.save_as_mainfile(filepath=str(O/'editable_character_v5.blend'))
(O/'expressions.json').write_text(json.dumps(expr,indent=2));(O/'changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2));(O/'build_metrics.json').write_text(json.dumps(report,indent=2))
(O/'adult_neutral_refined.object_config.json').write_text(json.dumps({'render_asset':'adult_neutral_refined.glb','is_collidable':False,'scale':[1,1,1],'up':[0,1,0],'front':[0,0,1],'shader_type':'material','COM':[0,0,0],'units_to_meters':1.0},indent=2))
print('BUILD_FINISHED',report)
