"""Explicitly authorized, bounded Go transport. No key files/environment lookup.
The credential is supplied in memory by the one-trial entry point. No redirects,
provider/model fallback, retry, top-up, generic SDK UA, or HTTP debug logging.
"""
import http.client,json,time,math
from dataclasses import dataclass
HOST='opencode.ai'
ENDPOINT='https://opencode.ai/inference/go/openai/v1/chat/completions'
PATH='/inference/go/openai/v1/chat/completions'
USER_AGENT='r2rce-navigation-research/0.3'
class TransportFailure(RuntimeError): pass

@dataclass(frozen=True)
class Limits:
    max_requests:int=1
    min_interval_s:float=3.0
    max_allowance_usd:float=.5
    timeout_s:float=90.0
    def __post_init__(self):
        if type(self.max_requests) is not int or not 1<=self.max_requests<=160:raise ValueError('Request cap must be1..160')
        if not 3<=self.min_interval_s<=60 or not .001<=self.max_allowance_usd<=.5 or not 5<=self.timeout_s<=120:raise ValueError('Invalid bounded limits')

class GoTransport:
    def __init__(self,credential,*,extra_usage_disabled_confirmed=False,limits=Limits(),connection_factory=None,clock=time.monotonic,sleep=time.sleep):
        if extra_usage_disabled_confirmed is not True:raise TransportFailure('Extra Usage must be verified disabled before any request')
        if not isinstance(credential,str) or len(credential)<16 or any(x in credential for x in '\r\n'):raise TransportFailure('Invalid transient credential')
        self._credential=credential;self.limits=limits;self._factory=connection_factory or http.client.HTTPSConnection;self._clock=clock;self._sleep=sleep
        self.requests=0;self.allowance=0.;self.records=[];self._last=None;self._failed=False
    def __repr__(self):return f'GoTransport(requests={self.requests}, credential=<redacted>)'
    def close(self):self._credential=None
    def __call__(self,prepared):
        if self._failed or not self._credential:raise TransportFailure('Transport stopped; no retry')
        if prepared.endpoint!=ENDPOINT:raise TransportFailure('Unexpected endpoint; credential not sent')
        if self.requests>=self.limits.max_requests or self.allowance>=self.limits.max_allowance_usd:raise TransportFailure('Bounded trial limit reached')
        obj=json.loads(prepared.body)
        if obj.get('model')!='deepseek-v4.1-flash' or obj.get('stream') is not False:raise TransportFailure('Unexpected model/stream mode')
        if self._last is not None:self._sleep(max(0.,self.limits.min_interval_s-(self._clock()-self._last)))
        self._last=self._clock();self.requests+=1;connection=None
        record={'request':self.requests,'endpoint':ENDPOINT,'model_requested':obj['model'],'image_count':prepared.image_count,'extra_usage_disabled_confirmed':True,'automatic_retries':0}
        try:
            # Fixed official host/path and no redirect handling. Authorization is
            # never inserted into the messages, request audit or saved outputs.
            headers={k:v for k,v in prepared.headers.items() if k.lower() in {'content-type','x-opencode-session'}}
            headers['User-Agent']=USER_AGENT;headers['Authorization']='Bearer '+self._credential
            connection=self._factory(HOST,timeout=self.limits.timeout_s)
            connection.request('POST',PATH,body=prepared.body,headers=headers)
            response=connection.getresponse();record['http_status']=response.status
            raw=response.read(1024*1024+1)
            if len(raw)>1024*1024:raise TransportFailure('Response exceeds size bound')
            if response.status!=200:raise TransportFailure('Official endpoint returned HTTP '+str(response.status)+'; no retry/fallback')
            data=json.loads(raw);usage=data.get('usage',{});billing=usage.get('opencode',{})
            cost=usage.get('cost');allowance=billing.get('allowance')
            if type(cost) not in (int,float) or not math.isfinite(cost) or cost!=0 or billing.get('billing_source') not in ('go','go-plus'):
                raise TransportFailure('Response does not prove zero-balance-charge Go billing; stop')
            if type(allowance) not in (int,float) or not math.isfinite(allowance) or allowance<0:raise TransportFailure('Missing valid Go allowance cost; stop')
            self.allowance+=float(allowance)
            record.update(model_returned=data.get('model'),billing_source=billing['billing_source'],balance_cost_usd=cost,allowance_usd=allowance)
            return raw
        except Exception as exc:
            self._failed=True
            if isinstance(exc,TransportFailure):raise
            # Do not persist raw exception bodies, headers or potentially echoed key.
            raise TransportFailure('Request failed or timed out; outcome may be uncertain; no automatic retry') from None
        finally:
            record['elapsed_s']=self._clock()-self._last;self.records.append(record)
            if connection is not None:connection.close()
