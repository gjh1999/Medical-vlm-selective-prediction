"""Frozen image features only. Needs a complete image audit; no labels/model fitting.

Both model and preprocessing are pinned from local verified weights. No online
fallback, no pickle feature cache, no test-dependent text/prompt selection.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path


def read_inputs(audit_dir):
    audit=json.loads((audit_dir/'audit.json').read_text(encoding='utf-8'))
    if audit.get('status') not in {'passed_no_candidates','review_required'}:raise ValueError('Image audit incomplete/failed')
    if audit.get('expected')!=audit.get('decoded'):raise ValueError('Incomplete image coverage')
    manifest=audit_dir/'manifest.csv'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=audit['manifest_sha256']:raise ValueError('Manifest changed after audit')
    with manifest.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=audit['decoded']:raise ValueError('Manifest row count mismatch')
    return audit,rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--audit',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--weights',required=True);ap.add_argument('--weights-sha256',required=True)
    ap.add_argument('--model',choices=['clip','biomedclip'],required=True)
    ap.add_argument('--model-config',help='BiomedCLIP open_clip_config.json from pinned revision')
    ap.add_argument('--revision',required=True);ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--device',choices=['cuda','cpu'],default='cuda');args=ap.parse_args()
    if args.batch_size<1:ap.error('batch-size must be positive')
    audit,rows=read_inputs(Path(args.audit));weights=Path(args.weights)
    with weights.open('rb') as stream:weight_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    if weight_hash!=args.weights_sha256:raise ValueError('Weight SHA256 mismatch')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    report={'status':'in_progress','model':args.model,'revision':args.revision,'weights_sha256':args.weights_sha256,
            'audit_sha256':hashlib.sha256((Path(args.audit)/'audit.json').read_bytes()).hexdigest(),
            'manifest_sha256':audit['manifest_sha256'],'batch_size':args.batch_size,'device':args.device,
            'feature_dtype':'float32','normalization':'L2','precision':'float32',
            'duplicates_review_pending':audit['status']=='review_required','model_training':False}
    start=time.perf_counter()
    try:
        import numpy as np
        import torch
        import open_clip
        from PIL import Image,ImageOps
        if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
        device=torch.device(args.device)
        report['versions']={p:importlib.metadata.version(p) for p in ['torch','torchvision','open_clip_torch','numpy','pillow','transformers']}
        report['python']=platform.python_version();report['cuda_build']=torch.version.cuda
        if args.model=='clip':
            model,_,preprocess=open_clip.create_model_and_transforms('ViT-B-16',pretrained=str(weights),device=device)
        else:
            if not args.model_config:raise ValueError('BiomedCLIP needs its pinned model config')
            config_bytes=Path(args.model_config).read_bytes();config=json.loads(config_bytes)
            report['model_config_sha256']=hashlib.sha256(config_bytes).hexdigest()
            # Follows the official BiomedCLIP local-loading recipe. Version-sensitive.
            from open_clip.factory import _MODEL_CONFIGS
            _MODEL_CONFIGS['biomedclip_local']=config['model_cfg']
            model,_,preprocess=open_clip.create_model_and_transforms('biomedclip_local',pretrained=str(weights),
                device=device,**{f'force_{k}':v for k,v in config.get('preprocess_cfg',{}).items() if k=='image_size'},
                **{f'image_{k}':v for k,v in config.get('preprocess_cfg',{}).items() if k in {'mean','std'}})
        model.eval();model.requires_grad_(False);report['preprocess']=repr(preprocess)
        if args.device=='cuda':
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();report['gpu']=torch.cuda.get_device_name()
        features=[];inference_start=time.perf_counter()
        for offset in range(0,len(rows),args.batch_size):
            images=[]
            for row in rows[offset:offset+args.batch_size]:
                p=Path(row['image_path'])
                if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Image changed since audit')
                with Image.open(p) as image:images.append(preprocess(ImageOps.exif_transpose(image).convert('RGB')))
            with torch.inference_mode():
                f=model.encode_image(torch.stack(images).to(device));f=f/f.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            features.append(f.cpu().float().numpy())
            print(f'Encoded {min(offset+args.batch_size,len(rows))}/{len(rows)}',flush=True)
        if args.device=='cuda':torch.cuda.synchronize()
        array=np.concatenate(features);np.save(out/'image_features.npy',array,allow_pickle=False)
        with (out/'manifest.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        report.update(status='complete',shape=list(array.shape),encoding_seconds=time.perf_counter()-inference_start,
                      peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
                      features_sha256=hashlib.sha256((out/'image_features.npy').read_bytes()).hexdigest())
    except Exception as exc:
        report['status']='failed';report['error']=str(exc);raise
    finally:
        report['total_seconds']=round(time.perf_counter()-start,3)
        (out/'cache.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
