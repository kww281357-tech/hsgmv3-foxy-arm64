import bpy,json,math,bmesh,numpy as np
from pathlib import Path
P=Path(__file__).resolve().parent;O=P/'v9';O.mkdir(exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(P/'v8/editable_character_v8_candidates.blend'),load_ui=False,use_scripts=False);scene=bpy.context.scene
neutral=json.loads((P/'v7/expressions.json').read_text())['neutral']
for o in scene.objects:
 if o.type=='MESH' and o.data.shape_keys:
  for k in o.data.shape_keys.key_blocks:
   if k.name!='Basis' and not k.name.startswith('$'):k.value=neutral.get(k.name,0)
bpy.context.view_layer.update()
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
rng=np.random.default_rng(902);noise=rng.standard_normal((N,N)).astype(np.float32);fine=smooth(noise,1);broad=smooth(noise,12)
rough=np.clip(.59+.05*cheek-.14*nose-.095*forehead-.20*lip-.10*lid+.025*fine,.30,.72)
mat=bpy.data.materials['Adult_Expression_Base.body'];skin=mat.node_tree.nodes['DiffuseTexture'].image;old=pixels(skin)
link_image(mat,save_image('v9_skin_regional_roughness',rough,True),'Roughness','v9_region_roughness');bsdf(mat).inputs['Specular IOR Level'].default_value=.25;bsdf(mat).inputs['Subsurface Weight'].default_value=0 # No Blender-only realism claim.
# Differentiate iris from sclera using source texture, retaining original brown identity.
eye=bpy.data.materials['Adult_Expression_Base.high-poly'];ep=pixels(eye.node_tree.nodes['DiffuseTexture'].image);eh,ew=ep.shape[:2];ey,ex=np.mgrid[0:eh,0:ew];ex=ex/ew;ey=ey/eh
radius=np.minimum(np.hypot((ex-.29),ey-.29),np.hypot(ex-.71,ey-.70));iris=np.clip((.125-radius)/.012,0,1);limbus=np.exp(-.5*((radius-.112)/.008)**2)
ep[:,:,:3]*=(1-.18*limbus)[:,:,None];eye.node_tree.nodes['DiffuseTexture'].image=save_image('v9_eye_albedo',ep)
link_image(eye,save_image('v9_eye_roughness',.20-.125*iris,True),'Roughness','v9_eye_region_roughness');bsdf(eye).inputs['Coat Weight'].default_value=0;bsdf(eye).inputs['IOR'].default_value=1.38
# Enamel and gingiva retain source differences, avoid uniformly white teeth shader.
teeth=bpy.data.materials['Adult_Expression_Base.teeth_base'];tp=pixels(teeth.node_tree.nodes['DiffuseTexture'].image);enamel=np.clip(1-(tp[:,:,0]-tp[:,:,1])*8,0,1)*(tp[:,:,:3].mean(2)>.08)
tp[:,:,:3]*=1-enamel[:,:,None]*np.array([.02,.045,.075]);teeth.node_tree.nodes['DiffuseTexture'].image=save_image('v9_enamel_albedo',tp)
link_image(teeth,save_image('v9_enamel_roughness',.38-.13*enamel,True),'Roughness','v9_enamel_region_roughness');bsdf(teeth).inputs['Specular IOR Level'].default_value=.35
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
export('A')
# B: preserve photographic source detail; slightly strengthen high-frequency color variation.
blur=smooth(old[:,:,:3],6);detail=old[:,:,:3]-blur;rgb=old[:,:,:3]+detail*.35
blood=np.clip(.7*cheek+.30*nose,0,1);rgb*=1+blood[:,:,None]*np.array([.025,-.009,-.015]);rgb*=1+(fine*.007)[:,:,None]*front[:,:,None]
out=old.copy();out[:,:,:3]=rgb;mat.node_tree.nodes['DiffuseTexture'].image=save_image('v9_skin_albedo_detail',out)
# Tangent-space microrelief baked to PNG, not procedural shader nodes.
height=smooth(noise,2);dx=(np.roll(height,-1,1)-np.roll(height,1,1))*.10;dy=(np.roll(height,-1,0)-np.roll(height,1,0))*.10
normal=np.stack([-dx,-dy,np.ones_like(dx)],axis=2);normal/=np.linalg.norm(normal,axis=2)[:,:,None];normal=normal*.5+.5
nim=save_image('v9_skin_micro_normal',normal,True);mat.node_tree.nodes['skin_export_normal'].image=nim
for nd in mat.node_tree.nodes:
 if nd.type=='NORMAL_MAP':nd.inputs['Strength'].default_value=.55
export('B');bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(O/'editable_character_v9_candidates.blend'))
(O/'build_metrics.json').write_text(json.dumps(report,indent=2));(O/'texture_changes.json').write_text(json.dumps({'skin_roughness_range':[float(rough.min()),float(rough.max())],'skin_roughness_std':float(rough.std()),'skin_albedo_source_detail_amplification':1.35,'new_geometry':False,'all_new_textures_baked_png':True,'subsurface_not_used':True},indent=2));print('V9_CANDIDATES_BUILT')
