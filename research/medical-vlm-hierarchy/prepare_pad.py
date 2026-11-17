"""Official PAD-UFES-20 metadata audit with patient-isolated multi-label groups."""
import argparse
import csv
import hashlib
import io
import json
import random
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

BASE='https://data.mendeley.com/public-api/datasets/zr7vgbcyr2'
RATIOS={'train':.4,'dev':.1,'calibration':.25,'test':.25}


def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':'MedicalVLMResearch/0.2'})
    with urllib.request.urlopen(req,timeout=45) as response:return response.read()


def patient_splits(rows, seed):
    groups=defaultdict(list)
    for r in rows:
        if not r['patient_id']:raise ValueError('Missing patient ID')
        groups[r['patient_id']].append(r)
    # Balance lesion-level diagnosis counts, without pretending a patient has one label.
    vectors={}
    lesions={}
    for group,members in groups.items():
        unique={}
        for r in members:
            key=(r['patient_id'],r['lesion_id'])
            if key in lesions and lesions[key]!=r['diagnostic']:raise ValueError('Conflicting diagnosis within patient-lesion key')
            lesions[key]=r['diagnostic']
            unique[key]=r['diagnostic']
        vectors[group]=Counter(unique.values())
    total=Counter()
    for counts in vectors.values():total.update(counts)
    rng=random.Random(seed);order=sorted(groups);rng.shuffle(order)
    # Larger/rarer patient groups assigned first; shuffle supplies reproducible ties.
    order.sort(key=lambda g:sum(vectors[g][y]/total[y] for y in vectors[g]),reverse=True)
    assigned={};current={s:Counter() for s in RATIOS};sizes=Counter()
    for group in order:
        counts=vectors[group]
        def cost(s):
            label_delta=sum((2*(current[s][y]-RATIOS[s]*total[y])*n+n*n)/max(1,RATIOS[s]*total[y]) for y,n in counts.items())
            group_delta=(2*(sizes[s]-RATIOS[s]*len(groups))+1)/max(1,RATIOS[s]*len(groups))
            return label_delta+group_delta
        split=min(RATIOS,key=cost);assigned[group]=split;sizes[split]+=1;current[split].update(counts)
    return assigned


def conservative_rows(rows):
    """Link patients sharing ambiguous raw lesion IDs; don't assume they are unique."""
    parent={r['patient_id']:r['patient_id'] for r in rows}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    owner={}
    for r in rows:
        p=r['patient_id'];key=r['lesion_id']
        if key in owner:
            a,b=find(p),find(owner[key]);parent[max(a,b)]=min(a,b)
        else:owner[key]=p
    result=[]
    for r in rows:
        result.append({**r,'original_patient_id':r['patient_id'],
                       'lesion_id':r['patient_id']+'::'+r['lesion_id'],
                       'patient_id':find(r['patient_id'])})
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--source-dir');args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    report={'status':'in_progress','source':BASE,'split_algorithm':'shared-id-linked-relative-deficit-v3','seeds':[17,29,43],'cache_reused':args.source_dir}
    try:
        cache=Path(args.source_dir) if args.source_dir else None
        metadata=json.loads((cache/'dataset_api.json').read_text(encoding='utf-8')) if cache else json.loads(fetch(BASE))
        inventory=json.loads((cache/'files_api.json').read_text(encoding='utf-8')) if cache else json.loads(fetch(BASE+'/files?folder_id=root&version=1'))
        (out/'dataset_api.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        (out/'files_api.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
        report['license']=metadata.get('data_licence',metadata.get('license',metadata.get('licence')))
        report['doi']=metadata.get('doi');report['version']=metadata.get('version',1)
        f=next(f for f in inventory if f['filename']=='metadata.csv');content=f['content_details']
        payload=(cache/'metadata.csv').read_bytes() if cache else fetch(content['download_url']);actual=hashlib.sha256(payload).hexdigest()
        if actual!=content['sha256_hash']:raise ValueError('Official metadata SHA256 mismatch')
        (out/'metadata.csv').write_bytes(payload)
        report['metadata_sha256']=actual;report['metadata_bytes']=len(payload)
        rows=list(csv.DictReader(io.StringIO(payload.decode('utf-8-sig'))))
        required={'patient_id','lesion_id','img_id','diagnostic'}
        if not rows or not required.issubset(rows[0]):raise ValueError('Unexpected PAD metadata schema')
        if len({r['img_id'] for r in rows})!=len(rows):raise ValueError('Repeated image IDs')
        report.update(images=len(rows),patients=len({r['patient_id'] for r in rows}),lesions=len({r['lesion_id'] for r in rows}),
                      image_label_counts=dict(Counter(r['diagnostic'] for r in rows)),splits={})
        raw_groups=defaultdict(list)
        for r in rows:raw_groups[r['lesion_id']].append(r)
        report['compound_patient_lesion_keys']=len({(r['patient_id'],r['lesion_id']) for r in rows})
        report['ambiguous_raw_lesion_ids']=sum(len({r['patient_id'] for r in members})>1 for members in raw_groups.values())
        report['raw_ids_with_multiple_labels']=sum(len({r['diagnostic'] for r in members})>1 for members in raw_groups.values())
        report['lesion_semantics']='raw ID and compound key counts differ; biological identity unresolved; conservatively link shared raw IDs'
        rows=conservative_rows(rows)
        report['split_components']=len({r['patient_id'] for r in rows})
        reps={}
        for r in sorted(rows,key=lambda r:r['img_id']):reps.setdefault(r['lesion_id'],r['img_id'])
        for seed in report['seeds']:
            assigned=patient_splits(rows,seed);target=out/f'split_seed{seed}.csv'
            with target.open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['image_id','patient_id','lesion_id','split_group','label','split','representative'])
                writer.writeheader()
                for r in sorted(rows,key=lambda r:r['img_id']):
                    writer.writerow({'image_id':r['img_id'],'patient_id':r['original_patient_id'],'lesion_id':r['lesion_id'],'split_group':r['patient_id'],
                                     'label':r['diagnostic'],'split':assigned[r['patient_id']],'representative':int(reps[r['lesion_id']]==r['img_id'])})
            report['splits'][str(seed)]={'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                'component_counts':dict(Counter(assigned.values())),
                'representative_label_counts':{s:dict(Counter(r['diagnostic'] for r in rows if assigned[r['patient_id']]==s and reps[r['lesion_id']]==r['img_id'])) for s in RATIOS},
                'patient_overlap_count':0}
        report['status']='metadata_and_patient_splits_complete_no_images'
    except Exception as exc:
        report['status']='failed';report['error']=str(exc);raise
    finally:
        report['elapsed_seconds']=round(time.perf_counter()-start,3)
        (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
