"""Source-only portable staging. Never downloads assets or runs Blender implicitly. CC0-1.0."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def verify_assets(asset_root):
 manifest=json.loads((ROOT/'external_assets.json').read_text());verified=[]
 for name in manifest['required_for_build']:
  f=Path(asset_root)/name
  if not f.is_file():raise ValueError('Required external asset missing: '+str(f)+'; see external_assets.json. Source-only checkout is not self-contained.')
  actual=hashlib.sha256(f.read_bytes()).hexdigest()
  if actual!=manifest['files'][name]['sha256']:raise ValueError('External asset SHA256 mismatch: '+name)
  verified.append(name)
 return verified

def prepare(asset_root,work):
 names=verify_assets(asset_root);out=Path(work).resolve()
 if out.exists():raise ValueError('Refusing to overwrite existing work directory: '+str(out))
 out.mkdir(parents=True)
 for f in (ROOT/'pipeline').glob('*.py'):shutil.copyfile(f,out/f.name)
 for name in names:
  dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(Path(asset_root)/name,dst)
 # Optional comparison input, copied only when its provenance hash is correct.
 m=json.loads((ROOT/'external_assets.json').read_text())
 for name in m['optional_for_comparison_render']:
  f=Path(asset_root)/name
  if f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest()==m['files'][name]['sha256']:shutil.copyfile(f,out/name)
 return out

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--work',type=Path);a=p.parse_args()
 if a.work:
  out=prepare(a.asset_root,a.work);print('Prepared:',out);print('Generation NOT executed. Explicit CPU command:');print('blender -b --disable-autoexec --threads 4 --python '+str(out/'refine_v7.py'))
 else:print(json.dumps({'verified_external_inputs':verify_assets(a.asset_root),'staged':False,'generation_executed':False},indent=2))
if __name__=='__main__':
 try:main()
 except ValueError as e:print(str(e),file=sys.stderr);sys.exit(2)
