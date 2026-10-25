import bpy,os,json,math,hashlib,struct
from mathutils import Vector
import argparse,sys
parser=argparse.ArgumentParser();parser.add_argument('--assets',required=True);opts=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);P=opts.assets;r={}
for name in ['neutral','smile','cry']:
 bpy.ops.wm.read_factory_settings(use_empty=True);path=P+'/adult_'+name+'.glb';d=open(path,'rb').read();magic,version,total=struct.unpack_from('<4sII',d);l,t=struct.unpack_from('<II',d,12);j=json.loads(d[20:20+l]);assert magic==b'glTF' and total==len(d) and version==2
 assert all('uri' not in im for im in j['images']);assert all('TEXCOORD_0' in pr['attributes'] for m in j['meshes'] for pr in m['primitives'] if 'baseColorTexture' in j['materials'][pr['material']].get('pbrMetallicRoughness',{}))
 bpy.ops.import_scene.gltf(filepath=path);objs=[o for o in bpy.context.scene.objects if o.type=='MESH'];pts=[o.matrix_world@v.co for o in objs for v in o.data.vertices];mn=[min(v[i] for v in pts) for i in range(3)];mx=[max(v[i] for v in pts) for i in range(3)];assert abs(mn[2])<.00001;assert all(math.isfinite(c) for v in pts for c in v);assert all(i.size[0]>0 for i in bpy.data.images)
 r[name]={'file_bytes':len(d),'sha256':hashlib.sha256(d).hexdigest(),'bounds_z_up_m':[mn,mx],'triangles':sum(len(p.vertices)-2 for o in objs for p in o.data.polygons),'images_loaded':len(bpy.data.images),'uv0_all_textured_primitives':True,'embedded_textures':True,'roundtrip_import':'PASS','armatures_in_static_glb':sum(o.type=='ARMATURE' for o in bpy.context.scene.objects)}
 scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.samples=48;scene.cycles.use_denoising=False;scene.render.resolution_x=700;scene.render.resolution_y=800;scene.render.resolution_percentage=100
 scene.world=bpy.data.worlds.new('World');scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.12,.15,.2,1);scene.world.node_tree.nodes['Background'].inputs[1].default_value=.35
 def aim(o,p):o.rotation_euler=(Vector(p)-o.location).to_track_quat('-Z','Y').to_euler()
 for loc,power,size in [((2,-3,3),220,2),((-2,-2,2),110,2),((1,2,3),200,2)]:
  bpy.ops.object.light_add(type='AREA',location=loc);o=bpy.context.object;o.data.energy=power;o.data.size=size;aim(o,(0,0,1.5))
 bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.type='ORTHO';cam.data.ortho_scale=.48;scene.camera=cam;cam.location=(0,-3,mx[2]-.16);aim(cam,(0,-.025,mx[2]-.19));scene.render.filepath=P+'/'+name+'_glb_verified.png';bpy.ops.render.render(write_still=True)
open(P+'/roundtrip_validation.json','w').write(json.dumps(r,indent=2))
