import bpy,os
from mathutils import Vector
import argparse,sys
parser=argparse.ArgumentParser();parser.add_argument('--assets',required=True);opts=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);P=opts.assets
for name in ['neutral','smile','cry']:
 bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.gltf(filepath=P+'/adult_'+name+'.glb')
 scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.samples=64;scene.cycles.use_denoising=False
 scene.render.resolution_x=900;scene.render.resolution_y=1500;scene.render.resolution_percentage=100
 scene.world=bpy.data.worlds.new('World');scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.12,.15,.2,1);scene.world.node_tree.nodes['Background'].inputs[1].default_value=.35
 def aim(o,p):o.rotation_euler=(Vector(p)-o.location).to_track_quat('-Z','Y').to_euler()
 for loc,power,size in [((2,-3,3),220,2),((-2,-2,2),110,2),((1,2,3),200,2)]:
  bpy.ops.object.light_add(type='AREA',location=loc);o=bpy.context.object;o.data.energy=power;o.data.size=size;aim(o,(0,0,1.1))
 bpy.ops.mesh.primitive_plane_add(size=200);floor=bpy.context.object;floor.name='Preview floor only';mat=bpy.data.materials.new('Studio floor');mat.diffuse_color=(.12,.15,.2,1);floor.data.materials.append(mat)
 bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.type='ORTHO';cam.data.ortho_scale=1.73;scene.camera=cam;cam.location=(0,-4,.95);aim(cam,(0,0,.765));scene.render.filepath=P+'/'+name+'_fullbody_v4.png';bpy.ops.render.render(write_still=True)
 if name=='neutral':
  cam.location=(3,-1,.95);aim(cam,(0,0,.765));scene.render.filepath=P+'/neutral_side_v4.png';bpy.ops.render.render(write_still=True)

