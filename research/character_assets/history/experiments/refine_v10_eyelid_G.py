import bpy,json,math,bmesh,numpy as np
from pathlib import Path
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
# Rasterize face-coordinate masks into existing UV atlas; no screen-space repainting.
body=bpy.data.objects['Adult_Expression_Base'];ks=body.data.shape_keys.key_blocks;base=np.array([v.co[:] for v in ks[0].data]);mix=base.copy()
for k in list(ks)[1:]:mix+=(np.array([v.co[:] for v in k.data])-base)*k.value
N=2048;atlas=np.zeros((N,N,3),np.float32);valid=np.zeros((N,N),bool);uv=body.data.uv_layers.active.data;body.data.calc_loop_triangles()
for t in body.data.loop_triangles:
 xyz=mix[list(t.vertices)]
 if xyz[:,2].max()<1.27 or xyz[:,1].min()>-.05:continue
 q=np.array([uv[i].uv[:] for i in t.loops])*(N-1);lo=np.maximum(np.floor(q.min(0)).astype(int),0);hi=np.minimum(np.ceil(q.max(0)).astype(int),N-1)
 if np.any(hi<lo):continue
 xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1));v0=q[1]-q[0];v1=q[2]-q[0];det=v0[0]*v1[1]-v1[0]*v0[1]
 if abs(det)<1e-8:continue
 dx=xx-q[0,0];dy=yy-q[0,1];a=(dx*v1[1]-dy*v1[0])/det;b=(dy*v0[0]-dx*v0[1])/det;inside=(a>=0)&(b>=0)&(a+b<=1)
 sl=(slice(lo[1],hi[1]+1),slice(lo[0],hi[0]+1));vals=xyz[0]+a[:,:,None]*(xyz[1]-xyz[0])+b[:,:,None]*(xyz[2]-xyz[0]);atlas[sl][inside]=vals[inside];valid[sl][inside]=True
x,y,z=atlas.transpose(2,0,1);front=valid*np.clip((-y-.07)/.03,0,1)
def g(cx,cz,sx,sz):return np.exp(-.5*((np.abs(x)-cx)/sx)**2-.5*((z-cz)/sz)**2)*front
cheek=g(.038,1.356,.019,.021);nose=g(0,1.36,.011,.023);forehead=g(0,1.435,.046,.021);lip=g(0,1.329,.029,.009);lid=g(.029,1.384,.021,.007)
mat=bpy.data.materials['Adult_Expression_Base.body'];oldim=mat.node_tree.nodes['DiffuseTexture'].image;old=pixels(oldim);blurim=oldim.copy();blurim.scale(256,256);blurim.scale(2048,2048);blur=pixels(blurim)
# Reduce only the duplicated baked dark lash margin; actual 3D lashes remain.
lidband=g(.029,1.384,.020,.006);dark=np.clip((blur[:,:,:3].mean(2)-old[:,:,:3].mean(2)-.012)/.07,0,1);weight=.60*lidband*dark
new=old.copy();new[:,:,:3]=old[:,:,:3]*(1-weight[:,:,None])+blur[:,:,:3]*weight[:,:,None];mat.node_tree.nodes['DiffuseTexture'].image=save_image('v10_eyelid_albedo_delining',new)
# Millimetre-bounded upper-lid fold: a new reversible corrective, not an eye-size change.
xx,yy,zz=mix.T;ax=np.abs(xx);arch=1.3945+.0020*(1-np.minimum(1,((ax-.029)/.022)**2));support=np.exp(-.5*((ax-.029)/.020)**2)*np.clip((-yy-.080)/.025,0,1)
crease=np.exp(-.5*((zz-arch)/.00085)**2)*support;ridge=np.exp(-.5*((zz-(arch-.0015))/.0011)**2)*support;delta=np.zeros_like(base);delta[:,1]=.00045*crease-.00018*ridge;delta[(zz<1.389)|(zz>1.401)|(ax<.009)|(ax>.058)]=0
key=body.shape_key_add(name='v10_upper_lid_fold',from_mix=False)
for v,co in zip(key.data,base+delta):v.co=co
key.value=1.;bpy.context.view_layer.update()
(O/'eyelid_G_metrics.json').write_text(json.dumps({'new_corrective':'v10_upper_lid_fold','max_displacement_m':float(np.linalg.norm(delta,axis=1).max()),'affected_vertices':int(np.count_nonzero(np.linalg.norm(delta,axis=1)>1e-8)),'source_basis_unchanged':True,'baked_liner_reduction_max_weight':float(weight.max()),'body_hands_bones_unchanged':True,'status':'neutral candidate only; expression compatibility untested'},indent=2))
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
export('G');bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(O/'editable_character_v10_G.blend'));print('V10_G_BUILT')
