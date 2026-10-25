import bpy,json
from pathlib import Path
from mathutils.bvhtree import BVHTree
P=Path(__file__).resolve().parent;O=P/'v7'
bpy.ops.wm.open_mainfile(filepath=str(O/'editable_character_v7.blend'),load_ui=False,use_scripts=False)
d=bpy.context.evaluated_depsgraph_get();base=bpy.data.objects['Adult_Expression_Base'];ev=base.evaluated_get(d);mesh=ev.to_mesh(preserve_all_data_layers=True,depsgraph=d)
verts=[base.matrix_world@v.co for v in mesh.vertices];names={g.index:g.name for g in base.vertex_groups}
cloth=next(o for o in bpy.context.scene.objects if o.type=='MESH' and 'elegantsuit' in o.name);ce=cloth.evaluated_get(d);cm=ce.to_mesh();cv=[cloth.matrix_world@v.co for v in cm.vertices];cb=BVHTree.FromPolygons(cv,[list(p.vertices) for p in cm.polygons])
report={'method':'Evaluated neutral mesh; hand/finger vertex groups >=0.5; BVH triangle overlaps against garment. This is a surface-intersection check, not a complete collision/inside-volume proof.','hands':{}}
for side in ['l','r']:
 selected={v.index for v in mesh.vertices if sum(g.weight for g in v.groups if names.get(g.group,'').endswith('_'+side) and names.get(g.group,'').startswith(('hand_','index_','middle_','ring_','pinky_','thumb_')))>=.5}
 faces=[list(p.vertices) for p in mesh.polygons if all(i in selected for i in p.vertices)]
 hb=BVHTree.FromPolygons(verts,faces);ds=[cb.find_nearest(verts[i])[3] for i in selected]
 report['hands'][side]={'selected_vertices':len(selected),'selected_faces':len(faces),'garment_triangle_intersections':len(hb.overlap(cb)),'minimum_vertex_to_garment_distance_m':min(ds)}
report['finger_pair_surface_intersections']={}
for side in ['l','r']:
 trees={}
 for finger in ['index','middle','ring','pinky','thumb']:
  selected={v.index for v in mesh.vertices if sum(g.weight for g in v.groups if names.get(g.group,'').endswith('_'+side) and names.get(g.group,'').startswith(finger+'_'))>=.5}
  faces=[list(p.vertices) for p in mesh.polygons if all(i in selected for i in p.vertices)]
  trees[finger]=BVHTree.FromPolygons(verts,faces)
 fs=list(trees)
 for i,a in enumerate(fs):
  for b in fs[i+1:]:report['finger_pair_surface_intersections'][side+':'+a+'-'+b]=len(trees[a].overlap(trees[b]))
(O/'hand_geometry_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
