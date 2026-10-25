import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
import prepare_rebuild as mod
class PrepareTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.old=mod.ROOT;mod.ROOT=self.root/'src';mod.ROOT.mkdir();(mod.ROOT/'pipeline').mkdir();(mod.ROOT/'pipeline/refine_v7.py').write_text('# inert test fixture\n')
  self.assets=self.root/'assets';(self.assets/'v6').mkdir(parents=True);(self.assets/'v6/a.blend').write_bytes(b'fixture')
  (mod.ROOT/'external_assets.json').write_text(json.dumps({'required_for_build':['v6/a.blend'],'optional_for_comparison_render':[],'files':{'v6/a.blend':{'sha256':hashlib.sha256(b'fixture').hexdigest()}}}))
 def tearDown(self):mod.ROOT=self.old;self.tmp.cleanup()
 def test_missing(self):
  with self.assertRaisesRegex(ValueError,'missing'):mod.verify_assets(self.root/'missing')
 def test_wrong_hash(self):
  (self.assets/'v6/a.blend').write_bytes(b'changed')
  with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):mod.verify_assets(self.assets)
 def test_prepare_no_execution(self):
  out=mod.prepare(self.assets,self.root/'work');self.assertEqual((out/'v6/a.blend').read_bytes(),b'fixture');self.assertTrue((out/'refine_v7.py').is_file());self.assertFalse((out/'v7').exists())
 def test_refuse_overwrite(self):
  out=self.root/'work';out.mkdir()
  with self.assertRaisesRegex(ValueError,'overwrite'):mod.prepare(self.assets,out)
if __name__=='__main__':unittest.main()
