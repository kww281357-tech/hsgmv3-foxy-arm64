"""Export existing v4 editable character into three static GLBs. CC0-1.0."""
import bpy,sys,argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--expressions',required=True);p.add_argument('--out',required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=a.source);mesh=[o for o in bpy.context.scene.objects if o.type=='MESH'];expr=json.load(open(a.expressions))
for name,weights in expr.items():
 for o in mesh:
  if o.data.shape_keys:
   for key in o.data.shape_keys.key_blocks:
    if key.name!='Basis' and not key.name.startswith('$'):key.value=weights.get(key.name,0)
 bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get();bpy.ops.object.select_all(action='DESELECT');copies=[]
 for o in mesh:
  data=bpy.data.meshes.new_from_object(o.evaluated_get(deps),preserve_all_data_layers=True,depsgraph=deps);ob=bpy.data.objects.new(o.name+'_'+name,data);bpy.context.scene.collection.objects.link(ob);ob.matrix_world=o.matrix_world.copy();copies.append(ob);ob.select_set(True)
 floor=min((ob.matrix_world@v.co).z for ob in copies for v in ob.data.vertices)
 for ob in copies:ob.location.z-=floor
 bpy.ops.export_scene.gltf(filepath=str(out/('adult_'+name+'.glb')),export_format='GLB',use_selection=True,export_animations=False)
 for ob in copies:bpy.data.objects.remove(ob,do_unlink=True)
