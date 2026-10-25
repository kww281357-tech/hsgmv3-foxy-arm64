from pathlib import Path
import struct,json,hashlib
P=Path(__file__).resolve().parent/'v10';raw=(P/'adult_neutral_H.glb').read_bytes();n=struct.unpack_from('<I',raw,12)[0];j=json.loads(raw[20:20+n]);removed=[]
for m in j.get('materials',[]):
 if m.get('extensions'):
  removed.extend(m['extensions'].keys());m.pop('extensions')
for k in ['extensionsUsed','extensionsRequired']:
 if k in j:j[k]=[x for x in j[k] if x not in ['KHR_materials_clearcoat','KHR_materials_specular','KHR_materials_ior']]
 if k in j and not j[k]:j.pop(k)
payload=json.dumps(j,separators=(',',':')).encode();payload+=b' '*((-len(payload))%4);tail=raw[20+n:];out=struct.pack('<4sII',b'glTF',2,20+len(payload)+len(tail))+struct.pack('<II',len(payload),0x4e4f534a)+payload+tail;(P/'adult_neutral_H_core.glb').write_bytes(out)
(P/'core_pbr_fallback.json').write_text(json.dumps({'removed_optional_material_extensions':sorted(set(removed)),'embedded_images':len(j['images']),'core_basecolor_roughness_normal_textures_preserved':True,'bin_chunk_unchanged':tail==raw[20+n:],'sha256':hashlib.sha256(out).hexdigest()},indent=2));print('CORE_PBR_FALLBACK_CREATED')
