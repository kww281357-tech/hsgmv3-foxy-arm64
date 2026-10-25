"""One explicitly authorized image/schema probe, never a full navigation claim.
Run only AFTER verifying Go Extra Usage is OFF. Reads approved key transiently
from the terminal with echo disabled; never from argv/env/files or saved config.
"""
from pathlib import Path
import argparse,sys,json,base64,getpass,hashlib
from blue_help.providers.go_contract import prepare_request,parse_response,session_id
from blue_help.providers.go_transport import GoTransport,Limits

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--confirmed-extra-usage-off',action='store_true');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--image',type=Path,required=True);args=ap.parse_args(argv)
    if not args.confirmed_extra_usage_off:raise SystemExit('BLOCKED: verify account Go Extra Usage OFF before sending any request')
    if args.output.exists():raise SystemExit('Choose a fresh result path; never overwrite an attempt')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    path=args.image
    picture=path.read_bytes()
    msg=[{'role':'system','content':'This is a development-time visual input/output test for an indoor navigation agent in a simulator. Describe only the attached current camera image. You have no goal, map or route. Output exactly a JSON object with keys visible_object, color, scene_evidence, action. action must be WAIT. Do not call tools or output coordinates.'},
      {'role':'user','content':[{'type':'text','text':'What is the most prominent object directly ahead? State one visible distinguishing feature. Return the requested JSON; do not infer an unseen destination.'},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(picture).decode()}}]}]
    request=prepare_request(msg,session=session_id('single-ep40-vision-probe','40','max3_ask'),max_tokens=1024)
    if not sys.stdin.isatty():raise SystemExit('Use an interactive no-echo terminal; do not pass a key file or command argument')
    key=getpass.getpass('Approved transient Go credential (input hidden; not saved): ')
    transport=GoTransport(key,extra_usage_disabled_confirmed=True,limits=Limits(max_requests=1));key=None
    report={'kind':'single_image_compatibility_probe_not_navigation','api_requests':0,'image_sha256':hashlib.sha256(picture).hexdigest(),'expected_model':'deepseek-v4.1-flash','credential_persisted':False,'max_requests':1,'extra_usage_off_verified_by_operator':True}
    try:
        response=parse_response(transport(request));data=json.loads(response.content)
        if set(data)!={'visible_object','color','scene_evidence','action'} or data['action']!='WAIT' or any(not isinstance(v,str) for v in data.values()):raise ValueError('Vision probe output schema mismatch')
        report.update(status='response_received_and_schema_valid',model_returned=response.model,answer=data,usage=response.usage,image_capability_evidence='Model accepted one image request and returned a schema-valid description; content still requires human comparison with the image')
    except Exception as exc:
        report.update(status='stopped',error_type=type(exc).__name__,error='Probe failed or outcome uncertain; no raw provider error/response or automatic retry')
    finally:
        report.update(api_requests=transport.requests,transport_audit=transport.records);transport.close();args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    return 0 if report.get('status')=='response_received_and_schema_valid' else 2
if __name__=='__main__':raise SystemExit(main())
