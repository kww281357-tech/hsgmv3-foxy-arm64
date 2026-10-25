"""Make labelled user-review media from existing REAL-run evidence; no simulator/model.

This is a presentation copy, never policy input. It does not invent observations,
modify navigation history, or add a success claim. Original raw/debug files remain.
"""
from pathlib import Path
import argparse,json,textwrap
from PIL import Image,ImageDraw


def make_review_media(run_dir,output):
    root=Path(run_dir);output=Path(output)
    if output.exists():raise ValueError('Do not overwrite prior review media')
    run=json.loads((root/'interaction_result.json').read_text())
    evidence=json.loads((root/'frame_evidence.private.json').read_text())
    frames=[];durations=[]
    for row in evidence:
        original=Image.open(root/f"debug_{row['index']:03d}.png").convert('RGB')
        canvas=Image.new('RGB',(740,440),(21,25,31));canvas.paste(original,(12,64))
        draw=ImageDraw.Draw(canvas)
        draw.text((12,12),'REAL HABITAT CPU INTERACTION TEST',fill=(110,210,255))
        draw.text((12,30),'Staged start + scripted text follower | NOT autonomous R2R-CE results',fill='white')
        draw.text((12,314),f"Scene: zsNo4HB9uLZ | Original task ID: {run['original_episode_id']}",fill='white')
        draw.text((12,332),f"Measured movement: {run['metrics']['path_length_m']:.3f} m | API calls: {run['api_attempts']}",fill='white')
        draw.text((12,350),'The debug circle is not in the model camera image.',fill=(230,210,130))
        draw.text((12,368),'A safety stop is not proof of reaching the goal.',fill=(230,210,130))
        draw.text((12,386),'One explicit, geometry-selected 0.2 m human relocation.',fill=(230,210,130))
        step=row['step'];event=row['event'];count=event['interaction']['delivered_answers']
        y=64
        lines=[f"Step {step} | {event['phase']}",f"Successful answers: {count}/3",'']
        if count==0:
            lines+=['DUMMY OFFER','Do you need any help? You can accept, decline, or ignore this offer and continue navigating.']
        else:
            q=next((x['text'] for x in run['dialogue'] if x['kind']=='question'),None)
            a=next((x['text'] for x in run['dialogue'] if x['kind']=='answer'),None)
            lines+=['VEHICLE QUESTION',q or '', '', 'DUMMY ANSWER',a or '']
        for paragraph in lines:
            wrapped=textwrap.wrap(paragraph,width=58) or ['']
            for line in wrapped:
                draw.text((350,y),line,fill='white');y+=13
        frames.append(canvas)
        durations.append(3000 if row['index'] in (0,1) else 700)
    if not frames:raise ValueError('No actual frames')
    frames[0].save(output,save_all=True,append_images=frames[1:],duration=durations,loop=0)
    frames[1 if len(frames)>1 else 0].save(output.with_suffix('.png'))
    return {'gif':str(output),'png':str(output.with_suffix('.png')),'frames':len(frames),'source_is_preserved':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();print(json.dumps(make_review_media(a.run_dir,a.output)))
