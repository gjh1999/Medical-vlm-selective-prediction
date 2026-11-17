"""Acquire official HAM10000 metadata; group split before any image inference."""
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

API = 'https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/DBW86T'


def download(url, target, expected_md5=None):
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}')
    request = urllib.request.Request(url,headers={'User-Agent':'MedicalVLMResearch/0.1'})
    partial = target.with_suffix(target.suffix+'.partial')
    if partial.exists():
        raise FileExistsError(f'Inspect incomplete download before retry: {partial}')
    md5=hashlib.md5(); sha=hashlib.sha256(); size=0
    with urllib.request.urlopen(request,timeout=60) as r, partial.open('xb') as f:
        while chunk:=r.read(1024*1024):
            f.write(chunk);md5.update(chunk);sha.update(chunk);size+=len(chunk)
    if expected_md5 and md5.hexdigest()!=expected_md5:
        raise ValueError(f'MD5 mismatch for {partial}; preserved for inspection')
    partial.rename(target)
    return {'bytes':size,'md5':md5.hexdigest(),'sha256':sha.hexdigest(),'url':url}


def grouped_splits(rows, seed):
    groups=defaultdict(list)
    for row in rows:groups[row['lesion_id']].append(row)
    strata=defaultdict(list)
    for group, members in groups.items():
        labels={r['dx'] for r in members}
        if len(labels)!=1: raise ValueError(f'Conflicting labels in lesion {group}')
        strata[next(iter(labels))].append(group)
    assigned={}; rng=random.Random(seed)
    for label, ids in sorted(strata.items()):
        ids=sorted(ids);rng.shuffle(ids)
        # Largest remainder allocation, preserving exact counts and group isolation.
        ratios={'train':.4,'dev':.1,'calibration':.25,'test':.25}
        counts={s:int(len(ids)*ratio) for s,ratio in ratios.items()}
        remaining=len(ids)-sum(counts.values())
        order=sorted(ratios,key=lambda s:-(len(ids)*ratios[s]-counts[s]))
        for s in order[:remaining]:counts[s]+=1
        cursor=0
        for split,count in counts.items():
            for group in ids[cursor:cursor+count]: assigned[group]=split
            cursor+=count
    return assigned


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True)
    ap.add_argument('--download-images',action='store_true',help='Also download 2.77GB original image archives; no extraction')
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();report={'status':'in_progress','source':API}
    try:
        req=urllib.request.Request(API,headers={'User-Agent':'MedicalVLMResearch/0.1'})
        with urllib.request.urlopen(req,timeout=45) as r: meta=json.load(r)
        (out/'dataset_api.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        v=meta['data']['latestVersion'];files=[f['dataFile'] for f in v['files']]
        report.update(version=f"{v['versionNumber']}.{v['versionMinorNumber']}",terms=v.get('termsOfUse'),file_inventory=files)
        f=next(f for f in files if f['filename'].startswith('HAM10000_metadata.'))
        original=out/'HAM10000_metadata.original'
        # Preserve both checksum declarations; original exports need not match tab exports.
        report['metadata_download']=download(f"https://dataverse.harvard.edu/api/access/datafile/{f['id']}?format=original",original)
        report['metadata_download']['official_checksum']=f.get('checksum')
        report['metadata_download']['official_md5_matches']=(f.get('checksum',{}).get('type')=='MD5' and f['checksum']['value']==report['metadata_download']['md5'])
        text=original.read_text(encoding='utf-8-sig'); dialect=csv.Sniffer().sniff(text[:4096],delimiters=',\t')
        rows=list(csv.DictReader(io.StringIO(text),dialect=dialect))
        required={'lesion_id','image_id','dx'}
        if not rows or not required.issubset(rows[0]):raise ValueError('Unexpected metadata columns')
        if len({r['image_id'] for r in rows})!=len(rows):raise ValueError('Repeated image IDs')
        if any(not r['lesion_id'] or not r['dx'] for r in rows):raise ValueError('Missing group/label')
        report['image_count']=len(rows);report['lesion_count']=len({r['lesion_id'] for r in rows})
        report['label_counts']=dict(sorted(Counter(r['dx'] for r in rows).items()))
        report['patient_id_available']='patient_id' in rows[0]
        report['image_hash_audit']='not_run_no_images';report['near_duplicate_audit']='not_run_no_images'
        report['splits']={}
        for seed in [17,29,43]:
            assigned=grouped_splits(rows,seed)
            # Label-independent representative by lexicographic image ID per lesion.
            reps={}
            for r in sorted(rows,key=lambda r:r['image_id']):reps.setdefault(r['lesion_id'],r['image_id'])
            target=out/f'split_seed{seed}.csv'
            with target.open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['image_id','lesion_id','label','split','representative'])
                writer.writeheader()
                for r in sorted(rows,key=lambda r:r['image_id']):
                    writer.writerow({'image_id':r['image_id'],'lesion_id':r['lesion_id'],'label':r['dx'],'split':assigned[r['lesion_id']],'representative':int(reps[r['lesion_id']]==r['image_id'])})
            report['splits'][str(seed)]={'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'group_counts':dict(Counter(assigned.values())),
                'representative_labels':{s:dict(Counter(r['dx'] for r in rows if assigned[r['lesion_id']]==s and reps[r['lesion_id']]==r['image_id'])) for s in ['train','dev','calibration','test']},'group_leakage_count':0}
        if args.download_images:
            report['image_downloads']={}
            for f in files:
                if f['filename'] in ['HAM10000_images_part_1.zip','HAM10000_images_part_2.zip']:
                    report['image_downloads'][f['filename']]=download(f"https://dataverse.harvard.edu/api/access/datafile/{f['id']}",out/f['filename'],f['checksum']['value'])
        report['status']='metadata_and_group_splits_complete_images_not_audited'
    except Exception as exc:
        report['status']='failed';report['error']=str(exc)
        raise
    finally:
        report['elapsed_seconds']=round(time.perf_counter()-start,3)
        (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ['terms','file_inventory']},ensure_ascii=True,indent=2))


if __name__=='__main__':main()
