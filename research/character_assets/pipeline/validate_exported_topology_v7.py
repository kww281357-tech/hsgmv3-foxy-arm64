from pathlib import Path
import json,struct,hashlib
p=Path(__file__).resolve().parent/'v7';r={}
for e in ['neutral','smile','cry']:
 b=(p/f'adult_{e}.glb').read_bytes();n=struct.unpack_from('<I',b,12)[0];j=json.loads(b[20:20+n]);off=20+n;bn=struct.unpack_from('<I',b,off)[0];binary=b[off+8:off+8+bn];sig=[]
 for m in j['meshes']:
  for prim in m['primitives']:
   a=j['accessors'][prim['indices']];v=j['bufferViews'][a['bufferView']];start=v.get('byteOffset',0)+a.get('byteOffset',0);length=a['count']*{5121:1,5123:2,5125:4}[a['componentType']]
   sig.append({'position_count':j['accessors'][prim['attributes']['POSITION']]['count'],'index_count':a['count'],'index_hash':hashlib.sha256(binary[start:start+length]).hexdigest()})
 r[e]=sig
assert r['neutral']==r['smile']==r['cry'];(p/'exported_topology_validation.json').write_text(json.dumps({'all_three_exported_GLB_index_streams_and_position_counts_equal':True,'signatures':r},indent=2));print('EXPORTED_INDEX_STREAMS_IDENTICAL')
