from pathlib import Path
import json,struct,numpy as np
P=Path(__file__).resolve().parent/'v7';data={}
for name in ['neutral','smile','cry']:
 b=(P/f'adult_{name}.glb').read_bytes();n=struct.unpack_from('<I',b,12)[0];j=json.loads(b[20:20+n]);binary=b[28+n:];arrays=[]
 for m in j['meshes']:
  if not m['name'].startswith('base.'):continue
  for pr in m['primitives']:
   a=j['accessors'][pr['attributes']['POSITION']];v=j['bufferViews'][a['bufferView']];assert a['componentType']==5126 and a['type']=='VEC3'
   off=v.get('byteOffset',0)+a.get('byteOffset',0);stride=v.get('byteStride',12)
   arr=np.ndarray((a['count'],3),dtype='<f4',buffer=binary,offset=off,strides=(stride,4)).copy();arrays.append(arr)
 data[name]=np.concatenate(arrays)
mask=data['neutral'][:,1]<1.1
assert np.array_equal(data['neutral'][mask],data['smile'][mask]) and np.array_equal(data['neutral'][mask],data['cry'][mask])
(P/'lower_body_pose_validation.json').write_text(json.dumps({'method':'Direct exported GLB base mesh POSITION arrays, local glTF Y < 1.1 m','vertices_checked':int(mask.sum()),'all_three_positions_bitwise_equal':True},indent=2));print('EXPORTED_LOWER_BODY_POSE_IDENTICAL')
