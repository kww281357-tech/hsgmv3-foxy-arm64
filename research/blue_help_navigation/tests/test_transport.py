import unittest,json
from types import SimpleNamespace
from blue_help.providers.go_transport import *
from blue_help.trial.policy import audit_public
class FakeResponse:
    status=200
    def __init__(self,data=None,status=200):self.data=data or {'model':'deepseek-v4.1-flash','usage':{'cost':0,'opencode':{'billing_source':'go','allowance':.001}}};self.status=status
    def read(self,n):return json.dumps(self.data).encode()
class FakeConnection:
    def __init__(self,*a,**k):self.args=a;self.kwargs=k;self.sent=None;self.closed=False;self.response=FakeResponse()
    def request(self,*a,**k):self.sent=(a,k)
    def getresponse(self):return self.response
    def close(self):self.closed=True

def prepared(endpoint=ENDPOINT):return SimpleNamespace(endpoint=endpoint,body=json.dumps({'model':'deepseek-v4.1-flash','stream':False}).encode(),headers={'Content-Type':'application/json','x-opencode-session':'test','evil-header':'no'},image_count=1)
class TransportTests(unittest.TestCase):
    def test_requires_extra_usage_off(self):
        with self.assertRaises(TransportFailure):GoTransport('synthetic-test-key-not-real')
    def test_fixed_host_and_no_key_log(self):
        c=FakeConnection();t=GoTransport('synthetic-test-key-not-real',extra_usage_disabled_confirmed=True,connection_factory=lambda *a,**k:c)
        t(prepared());self.assertEqual(c.sent[0],('POST',PATH));self.assertEqual(c.sent[1]['headers']['User-Agent'],USER_AGENT);self.assertNotIn('evil-header',c.sent[1]['headers']);self.assertNotIn('synthetic-test-key',json.dumps(t.records));self.assertNotIn('synthetic-test-key',repr(t));self.assertTrue(c.closed)
    def test_redirect_never_followed(self):
        c=FakeConnection();c.response=FakeResponse(status=302);t=GoTransport('synthetic-test-key-not-real',extra_usage_disabled_confirmed=True,connection_factory=lambda *a,**k:c)
        with self.assertRaises(TransportFailure):t(prepared())
        with self.assertRaises(TransportFailure):t(prepared())
        self.assertEqual(t.requests,1)
    def test_wrong_destination_rejected_before_send(self):
        t=GoTransport('synthetic-test-key-not-real',extra_usage_disabled_confirmed=True)
        with self.assertRaises(TransportFailure):t(prepared('https://example.org'))
        self.assertEqual(t.requests,0)
    def test_one_probe_cap(self):
        t=GoTransport('synthetic-test-key-not-real',extra_usage_disabled_confirmed=True,connection_factory=FakeConnection);t(prepared())
        with self.assertRaises(TransportFailure):t(prepared())
    def test_balance_cost_stops(self):
        c=FakeConnection();c.response.data['usage']['cost']=.01
        t=GoTransport('synthetic-test-key-not-real',extra_usage_disabled_confirmed=True,connection_factory=lambda *a,**k:c)
        with self.assertRaises(TransportFailure):t(prepared())
    def test_private_nested_fields_rejected(self):
        with self.assertRaises(ValueError):audit_public({'history':[{'goal_position':[1,2,3]}]})
        self.assertTrue(audit_public({'last_delivered_answer':'Go about19meters toward the bed.'}))
if __name__=='__main__':unittest.main()
