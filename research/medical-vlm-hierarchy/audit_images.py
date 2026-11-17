"""Verify images against a split manifest, hash contents and flag near duplicates.

Requires Pillow only. Never deletes or silently merges a duplicate candidate.
"""
import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path


def dhash(image):
    from PIL import Image
    small=image.convert('L').resize((9,8),Image.Resampling.LANCZOS)
    pixels=list(small.getdata());bits=0
    for row in range(8):
        for col in range(8):bits=(bits<<1)|int(pixels[row*9+col]>pixels[row*9+col+1])
    return bits


def duplicate_candidates(records, radius=3):
    if radius!=3:raise ValueError('Index configured for Hamming radius 3 only')
    buckets=defaultdict(list);pairs=[]
    for i,row in enumerate(records):
        h=int(row['dhash'],16);seen=set()
        # Four disjoint 16-bit chunks ensure a shared chunk if <=3 bits differ.
        for part in range(4):seen.update(buckets[(part,(h>>(16*part))&65535)])
        for j in sorted(seen):
            other=records[j];distance=(h^int(other['dhash'],16)).bit_count()
            if distance<=radius:
                pairs.append({'first':other['image_id'],'second':row['image_id'],'hamming':distance,
                              'cross_split':other['split']!=row['split'],
                              'same_bytes':other['sha256']==row['sha256'],
                              'same_decoded_pixels':other['pixel_sha256']==row['pixel_sha256']})
        for part in range(4):buckets[(part,(h>>(16*part))&65535)].append(i)
    return pairs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--images',required=True);ap.add_argument('--split',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();report={'status':'in_progress','images_root':str(Path(args.images).resolve()),'split':str(Path(args.split).resolve())}
    try:
        from PIL import Image,ImageOps
        report['pillow_version']=Image.__version__
        split_payload=Path(args.split).read_bytes();report['split_sha256']=hashlib.sha256(split_payload).hexdigest()
        with Path(args.split).open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
        if not rows or not {'image_id','label','split','representative'}<=set(rows[0]):raise ValueError('Invalid split schema')
        if len({r['image_id'] for r in rows})!=len(rows):raise ValueError('Repeated image ID in split')
        index={}
        for p in sorted(Path(args.images).rglob('*')):
            if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'}:
                for key in {p.name,p.stem}:
                    if key in index and index[key]!=p:raise ValueError(f'Ambiguous image basename: {key}')
                    index[key]=p
        records=[];missing=[];corrupt=[]
        for row in rows:
            path=index.get(row['image_id'])
            if path is None:missing.append(row['image_id']);continue
            try:
                with Image.open(path) as image:
                    image=ImageOps.exif_transpose(image).convert('RGB');image.load()
                    pixel_hash=hashlib.sha256(f'{image.width}x{image.height}:RGB:'.encode()+image.tobytes()).hexdigest()
                    records.append({**row,'image_path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'pixel_sha256':pixel_hash,'width':image.width,'height':image.height,'dhash':f'{dhash(image):016x}'})
            except (OSError,ValueError) as exc:corrupt.append({'image_id':row['image_id'],'error':str(exc)})
        pairs=duplicate_candidates(records)
        if records:
            with (out/'manifest.csv').open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
            report['manifest_sha256']=hashlib.sha256((out/'manifest.csv').read_bytes()).hexdigest()
        (out/'duplicate_candidates.json').write_text(json.dumps(pairs,indent=2),encoding='utf-8')
        report.update(expected=len(rows),decoded=len(records),missing=missing,corrupt=corrupt,
                      near_candidate_pairs=len(pairs),cross_split_candidates=sum(p['cross_split'] for p in pairs),
                      exact_cross_split_pairs=sum(p['cross_split'] and (p['same_bytes'] or p['same_decoded_pixels']) for p in pairs),
                      deduplication='candidates_only_no_deletion',
                      status='incomplete_images' if missing or corrupt else ('review_required' if pairs else 'passed_no_candidates'))
    except Exception as exc:
        report['status']='failed';report['error']=str(exc);raise
    finally:
        report['seconds']=round(time.perf_counter()-start,3)
        (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ['missing','corrupt']},ensure_ascii=True,indent=2))
    return 2 if report['status']=='incomplete_images' else 0


if __name__=='__main__':raise SystemExit(main())
