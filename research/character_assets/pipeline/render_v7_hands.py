import bpy,json,struct,hashlib,math,argparse,sys
from pathlib import Path
from mathutils import Vector
ap=argparse.ArgumentParser();ap.add_argument('--assets',default=str(Path(__file__).resolve().parent/'v7'));a=ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []);P=Path(a.assets).resolve();report={}
def aim(o,p):o.rotation_euler=(Vector(p)-o.location).to_track_quat('-Z','Y').to_euler()
for name in ['adult_neutral']:
 path=P/(name+'.glb');raw=path.read_bytes();magic,version,total=struct.unpack_from('<4sII',raw);assert magic==b'glTF' and version==2 and total==len(raw)
 ln,_=struct.unpack_from('<II',raw,12);j=json.loads(raw[20:20+ln]);assert all('uri' not in im for im in j.get('images',[]))
 bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.gltf(filepath=str(path));sc=bpy.context.scene
 objects=[o for o in sc.objects if o.type=='MESH'];pts=[o.matrix_world@v.co for o in objects for v in o.data.vertices];mn=[min(v[i] for v in pts) for i in range(3)];mx=[max(v[i] for v in pts) for i in range(3)]
 assert all(math.isfinite(c) for v in pts for c in v);assert abs(mn[2])<1e-4
 report[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'bounds_blender_zup':[mn,mx],'embedded_images':len(j.get('images',[])),'meshes':len(objects),'import':'PASS','triangles':sum(len(f.vertices)-2 for o in objects for f in o.data.polygons),'gltf_extensions':j.get('extensionsUsed',[])}
 sc.render.engine='CYCLES';sc.cycles.device='CPU';sc.cycles.samples=48;sc.cycles.use_denoising=False;sc.render.resolution_x=600;sc.render.resolution_y=720;sc.render.resolution_percentage=100
 sc.view_settings.view_transform='AgX';sc.view_settings.look='AgX - Medium High Contrast'
 sc.world=bpy.data.worlds.new('Neutral studio');sc.world.use_nodes=True;sc.world.node_tree.nodes['Background'].inputs[0].default_value=(.12,.15,.2,1);sc.world.node_tree.nodes['Background'].inputs[1].default_value=.35
 for loc,power,size in [((2,-3,3),220,2),((-2,-2,2),110,2),((1,2,3),200,2)]:
  bpy.ops.object.light_add(type='AREA',location=loc);o=bpy.context.object;o.data.energy=power;o.data.size=size;aim(o,(0,0,1.4))
 bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.type='ORTHO';sc.camera=cam
 for view,loc,target,scale in [('full',(0,-4,.95),(0,0,.765),1.75),('left_hand',(.65,-1.8,.85),(.22,-.12,.755),.30),('right_hand',(-.65,-1.8,.85),(-.22,-.12,.755),.30)]:
  cam.location=loc;aim(cam,target);cam.data.ortho_scale=scale;sc.render.filepath=str(P/(name+'_'+view+'.png'));bpy.ops.render.render(write_still=True)
 (P/'roundtrip_validation.json').write_text(json.dumps(report,indent=2))
print('ALL_GLB_VALIDATED')
