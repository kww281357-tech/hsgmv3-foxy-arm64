import bpy,json,math,bmesh,numpy as np
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
P=Path(__file__).resolve().parent;O=P/'v10';O.mkdir(exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(P/'v9/editable_character_v9_D.blend'),load_ui=False,use_scripts=False);scene=bpy.context.scene
def bsdf(m):return next(n for n in m.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
def save_image(name,arr,noncolor=False):
 h,w=arr.shape[:2]
 if arr.ndim==2:arr=np.repeat(arr[:,:,None],3,axis=2)
 if arr.shape[2]==3:arr=np.concatenate([arr,np.ones((h,w,1),np.float32)],axis=2)
 im=bpy.data.images.new(name,width=w,height=h,alpha=True)
 if noncolor:im.colorspace_settings.name='Non-Color'
 im.pixels.foreach_set(np.clip(arr,0,1).astype(np.float32).ravel());im.filepath_raw=str(O/(name+'.png'));im.file_format='PNG';im.save();im.pack();return im
def pixels(im):return np.array(im.pixels[:],np.float32).reshape(im.size[1],im.size[0],4)
def link_image(m,im,socket,name):
 n=m.node_tree.nodes.new('ShaderNodeTexImage');n.name=name;n.image=im;m.node_tree.links.new(n.outputs['Color'],bsdf(m).inputs[socket]);return n
def smooth(a,n=1):
 for _ in range(n):a=(a*4+np.roll(a,1,0)+np.roll(a,-1,0)+np.roll(a,1,1)+np.roll(a,-1,1))/8
 return a
rig=next(o for o in scene.objects if o.type=='ARMATURE');deps=bpy.context.evaluated_depsgraph_get();hair=next(o for o in scene.objects if o.name.endswith('.ponytail01'))
hm0=bpy.data.meshes.new_from_object(hair.evaluated_get(deps),preserve_all_data_layers=True,depsgraph=deps);hm0.calc_loop_triangles();verts=[v.co.copy() for v in hm0.vertices];tris=[list(t.vertices) for t in hm0.loop_triangles];source_bvh=BVHTree.FromPolygons(verts,tris,all_triangles=True)
center=Vector((0,-.005,1.41));NP=128;NR=48;mins=[]
# Follow the existing styled hair envelope, not the head sphere used by rejected E.
for i in range(NP):
 phi=2*math.pi*i/NP;frontness=max(0,math.cos(phi));zmin=1.385+.047*frontness;found=[]
 for theta in np.linspace(-.35,1.50,100):
  d=Vector((math.sin(phi)*math.cos(theta),-math.cos(phi)*math.cos(theta),math.sin(theta)));hit,normal,idx,dist=source_bvh.ray_cast(center+d*.35,-d,.35)
  if hit is not None and hit.z>zmin and (hit-center).dot(d)>.045 and (hit-center).length<.145:found.append(float(theta))
 mins.append(min(found) if found else .45)
mins=np.array(mins)
for _ in range(2):mins=(np.roll(mins,1)+2*mins+np.roll(mins,-1))/4
radii=np.zeros((NR,NP));points=np.zeros((NR,NP,3));directions=np.zeros_like(points)
for j in range(NR):
 for i in range(NP):
  phi=2*math.pi*i/NP;theta=mins[i]+(math.pi/2-.015-mins[i])*j/(NR-1);d=Vector((math.sin(phi)*math.cos(theta),-math.cos(phi)*math.cos(theta),math.sin(theta)));hit,normal,idx,dist=source_bvh.ray_cast(center+d*.35,-d,.35);directions[j,i]=d
  radius=(hit-center).dot(d) if hit is not None else 0
  if not .045<radius<.145:radius=.085
  radii[j,i]=radius
# Smooth only millimetre-scale discontinuities across original card boundaries.
original=radii.copy()
for _ in range(4):
 sm=(np.roll(radii,1,1)+2*radii+np.roll(radii,-1,1))/4
 radii=np.clip(sm,original-.0025,original+.0025)
for j in range(1,NR-1):radii[j]=(radii[j-1]+2*radii[j]+radii[j+1])/4
points=np.array(center)[None,None,:]+directions*(radii[:,:,None]+.00035)
co=[Vector(p) for p in points.reshape(-1,3)];faces=[]
for j in range(NR-1):
 for i in range(NP):k=(i+1)%NP;faces.append([j*NP+i,j*NP+k,(j+1)*NP+k,(j+1)*NP+i])
co.append(Vector((0,-.005,float(points[-1,:,2].mean())+.0004)));pole=len(co)-1
for i in range(NP):faces.append([(NR-1)*NP+i,(NR-1)*NP+(i+1)%NP,pole])
mesh=bpy.data.meshes.new('v10_original_style_envelope');mesh.from_pydata(co,[],faces);mesh.update();cap=bpy.data.objects.new('v10_continuous_original_style_scalp',mesh);scene.collection.objects.link(cap)
for f in mesh.polygons:f.use_smooth=True
mat=bpy.data.materials.new('v10_hair_root_dark_brown');mat.use_nodes=True;bsdf(mat).inputs['Base Color'].default_value=(.024,.012,.006,1);bsdf(mat).inputs['Roughness'].default_value=.73;cap.data.materials.append(mat)
# Keep the old ponytail and lower side hair, remove only broad crown cards.
hair=next(o for o in scene.objects if o.name.endswith('.ponytail01'));hm=bpy.data.meshes.new_from_object(hair.evaluated_get(deps),preserve_all_data_layers=True,depsgraph=deps);b=bmesh.new();b.from_mesh(hm);remove=[f for f in b.faces if f.calc_center_median().z>1.420 and f.calc_center_median().y<.042];removed=len(remove);bmesh.ops.delete(b,geom=remove,context='FACES');b.to_mesh(hm);b.free();tail=bpy.data.objects.new('v9_retained_ponytail',hm);scene.collection.objects.link(tail);tail.matrix_world=hair.matrix_world.copy();bpy.data.objects.remove(hair,do_unlink=True)
for ob in list(scene.objects):
 if ob.name in ['v4_hair_detail','v5_fine_hair_geometry','v6_continuous_hair_groom']:bpy.data.objects.remove(ob,do_unlink=True)
bpy.context.view_layer.update();bvh=BVHTree.FromObject(cap,bpy.context.evaluated_depsgraph_get());center=Vector((0,-.005,1.41));created=[];rng=np.random.default_rng(912);mats=[]
for i,c in enumerate([(.022,.010,.005),(.035,.017,.008),(.044,.023,.011),(.028,.013,.006)]):
 m=bpy.data.materials.new('v9_brown_strand_'+str(i));m.use_nodes=True;bsdf(m).inputs['Base Color'].default_value=(*c,1);bsdf(m).inputs['Roughness'].default_value=.61;mats.append(m)
for i in range(800):
 x=-.985+1.97*i/799;points=[]
 for j in range(42):
  t=.05+2.83*j/41;wave=.012*math.sin(4*t+i*.17)*math.sin(math.pi*j/41);xx=max(-.99,min(.99,x+wave));d=Vector((xx*math.cos(t*.18),-math.sqrt(1-x*x)*math.cos(t),math.sqrt(1-x*x)*math.sin(t))).normalized();hit,norm,_,_=bvh.ray_cast(center+d*.35,-d,.45)
  if hit is not None and (hit-center).dot(d)>0:points.append(hit+norm*(.00025+.00035*math.sin(math.pi*j/41)))
 if len(points)<5:continue
 cu=bpy.data.curves.new('v9 individual tapered strand','CURVE');cu.dimensions='3D';cu.bevel_depth=float(rng.uniform(.00010,.00018));cu.bevel_resolution=0;sp=cu.splines.new('POLY');sp.points.add(len(points)-1)
 for k,(pnt,point) in enumerate(zip(sp.points,points)):pnt.co=(*point,1);pnt.radius=max(.12,math.sin(math.pi*(k+.5)/len(points))**.25)
 ob=bpy.data.objects.new('v9_strand',cu);scene.collection.objects.link(ob);ob.data.materials.append(mats[i%4]);bpy.ops.object.select_all(action='DESELECT');ob.select_set(True);bpy.context.view_layer.objects.active=ob;bpy.ops.object.convert(target='MESH');created.append(bpy.context.object)
bpy.ops.object.select_all(action='DESELECT')
for ob in created:ob.select_set(True)
bpy.context.view_layer.objects.active=created[0];bpy.ops.object.join();groom=bpy.context.object;groom.name='v9_continuous_groom'
for ob in [cap,tail,groom]:
 vg=ob.vertex_groups.new(name='head');vg.add(list(range(len(ob.data.vertices))),1.,'REPLACE');arm=ob.modifiers.new('Follow original head','ARMATURE');arm.object=rig
(O/'hair_structure_candidate_F.json').write_text(json.dumps({'scalp_vertices':len(mesh.vertices),'scalp_faces':len(mesh.polygons),'removed_old_crown_card_faces':removed,'new_strands':len(created),'retained':'original ponytail and lower side hair','new_external_asset':False,'status':'candidate requiring visual review'},indent=2))
report={}
def export(label):
 deps=bpy.context.evaluated_depsgraph_get();originals=[o for o in scene.objects if o.type=='MESH'];copies=[];bpy.ops.object.select_all(action='DESELECT')
 for o in originals:
  data=bpy.data.meshes.new_from_object(o.evaluated_get(deps),preserve_all_data_layers=True,depsgraph=deps);ob=bpy.data.objects.new(o.name+'_'+label,data);scene.collection.objects.link(ob);ob.matrix_world=o.matrix_world.copy();copies.append(ob);ob.select_set(True)
  bm=bmesh.new();bm.from_mesh(data);bmesh.ops.triangulate(bm,faces=list(bm.faces),quad_method='FIXED',ngon_method='EAR_CLIP');bm.to_mesh(data);bm.free();data.update()
 floor=min((ob.matrix_world@v.co).z for ob in copies for v in ob.data.vertices)
 for ob in copies:ob.location.z-=floor
 bpy.ops.export_scene.gltf(filepath=str(O/f'adult_neutral_{label}.glb'),export_format='GLB',use_selection=True,export_animations=False)
 report[label]={'vertices':sum(len(o.data.vertices) for o in copies),'triangles':sum(len(f.vertices)-2 for o in copies for f in o.data.polygons)}
 for ob in copies:bpy.data.objects.remove(ob,do_unlink=True)
export('F');bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(O/'editable_character_v10_F.blend'));print('V10_F_BUILT')
