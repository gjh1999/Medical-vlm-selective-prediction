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

from prepare_openai_clip import WEIGHT_SHA256
from prepare_biomedclip import CONFIG_BLOBS, WEIGHT_SHA256 as BIOMED_SHA256, validate_config_file


def read_inputs(audit_dir):
    audit=json.loads((audit_dir/'audit.json').read_text(encoding='utf-8'))
    if audit.get('status') not in {'passed_no_candidates','review_required'}:raise ValueError('Image audit incomplete/failed')
    if audit.get('expected')!=audit.get('decoded'):raise ValueError('Incomplete image coverage')
    manifest=audit_dir/'manifest.csv'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=audit['manifest_sha256']:raise ValueError('Manifest changed after audit')
    with manifest.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=audit['decoded']:raise ValueError('Manifest row count mismatch')
    return audit,rows


def load_encoder(name, weights, device, config_path, report, with_tokenizer=False):
    expected = WEIGHT_SHA256 if name == "clip" else BIOMED_SHA256
    with weights.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError(f"Unexpected checkpoint for {name}")
    report["weights_sha256"] = expected
    tokenizer = None
    if name == "clip":
        import clip
        model, preprocess = clip.load(str(weights.resolve()), device=device, jit=False)
        model.float()
        report["loader"] = "openai/CLIP clip.load"
        origin = json.loads(importlib.metadata.distribution("clip").read_text("direct_url.json") or "{}")
        report["loader_commit"] = origin.get("vcs_info", {}).get("commit_id")
        report["visual_activations"] = sorted({
            type(block.mlp.gelu).__name__ for block in model.visual.transformer.resblocks
        })
        if report["visual_activations"] != ["QuickGELU"]:
            raise ValueError("Unexpected visual activation in the OpenAI ViT-B/16 checkpoint")
        if with_tokenizer:
            tokenizer = clip.tokenize
    elif name == "biomedclip":
        import open_clip
        from open_clip.factory import _MODEL_CONFIGS
        if config_path is None:
            raise ValueError("BiomedCLIP requires open_clip_config.json")
        config_path = Path(config_path)
        report["model_config_sha256"] = validate_config_file(config_path, "open_clip_config.json")
        config = json.loads(config_path.read_bytes())
        text_dir = config_path.resolve().parent / "text"
        if text_dir.is_dir():
            report["text_config_sha256"] = {key: validate_config_file(text_dir / key, key)
                                            for key in CONFIG_BLOBS if key != "open_clip_config.json"}
            config["model_cfg"]["text_cfg"].update(hf_model_name=str(text_dir), hf_tokenizer_name=str(text_dir))
        _MODEL_CONFIGS["biomedclip_local"] = config["model_cfg"]
        preprocess_cfg = config["preprocess_cfg"]
        model, _, preprocess = open_clip.create_model_and_transforms(
            "biomedclip_local", pretrained=str(weights.resolve()), device=device, pretrained_hf=False,
            image_mean=preprocess_cfg["mean"], image_std=preprocess_cfg["std"])
        report["loader"] = "open_clip.create_model_and_transforms"
        if with_tokenizer:
            tokenizer = open_clip.get_tokenizer("biomedclip_local")
    else:
        raise ValueError(f"Unknown encoder {name}")
    model.eval()
    model.requires_grad_(False)
    report["preprocess"] = repr(preprocess)
    return model, preprocess, tokenizer


def encode_images(model, preprocess, rows, device):
    import torch
    from PIL import Image, ImageOps
    images = []
    for row in rows:
        path = Path(row["image_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Image changed since audit")
        with Image.open(path) as image:
            images.append(preprocess(ImageOps.exif_transpose(image).convert("RGB")))
    with torch.inference_mode():
        features = model.encode_image(torch.stack(images).to(device)).float()
        norms = features.norm(dim=-1, keepdim=True)
        if not torch.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError("Invalid image embedding norm")
        return (features / norms).cpu().numpy()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--audit',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--weights',required=True);ap.add_argument('--weights-sha256',required=True)
    ap.add_argument('--model',choices=['clip','biomedclip'],required=True)
    ap.add_argument('--model-config',help='BiomedCLIP open_clip_config.json from pinned revision')
    ap.add_argument('--revision',required=True);ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--smoke-only', action='store_true',
                    help='Encode at most one batch of train representatives, sorted by image ID')
    ap.add_argument('--device',choices=['cuda','cpu'],default='cuda');args=ap.parse_args()
    if args.batch_size<1:ap.error('batch-size must be positive')
    audit,rows=read_inputs(Path(args.audit));weights=Path(args.weights)
    expected = WEIGHT_SHA256 if args.model == 'clip' else BIOMED_SHA256
    if args.weights_sha256 != expected:raise ValueError('Expected the pinned checkpoint SHA256')
    if args.smoke_only:
        rows = sorted((r for r in rows if r['split'] == 'train' and r['representative'] == '1'),
                      key=lambda r: r['image_id'])[:args.batch_size]
        if not rows:
            raise ValueError('No train representatives in the audited manifest')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    report={'status':'in_progress','model':args.model,'revision':args.revision,'weights_sha256':args.weights_sha256,
            'audit_sha256':hashlib.sha256((Path(args.audit)/'audit.json').read_bytes()).hexdigest(),
            'manifest_sha256':audit['manifest_sha256'],'batch_size':args.batch_size,'device':args.device,
            'feature_dtype':'float32','normalization':'L2','precision':'float32',
            'duplicates_review_pending':audit['status']=='review_required','model_training':False,
            'scope':'train_representative_smoke' if args.smoke_only else 'all_audited_images',
            'input_count':len(rows)}
    start=time.perf_counter()
    try:
        import numpy as np
        import torch
        if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
        device=torch.device(args.device)
        packages = ['torch', 'torchvision', 'numpy', 'pillow']
        packages += ['clip'] if args.model == 'clip' else ['open_clip_torch', 'transformers']
        report['versions'] = {p: importlib.metadata.version(p) for p in packages}
        report['python']=platform.python_version();report['cuda_build']=torch.version.cuda
        model, preprocess, _ = load_encoder(args.model, weights, device, args.model_config, report)
        if args.device=='cuda':
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();report['gpu']=torch.cuda.get_device_name()
        features=[];inference_start=time.perf_counter()
        for offset in range(0,len(rows),args.batch_size):
            features.append(encode_images(model, preprocess, rows[offset:offset+args.batch_size], device))
            print(f'Encoded {min(offset+args.batch_size,len(rows))}/{len(rows)}',flush=True)
        if args.device=='cuda':torch.cuda.synchronize()
        array=np.concatenate(features);np.save(out/'image_features.npy',array,allow_pickle=False)
        with (out/'manifest.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        report.update(status='smoke_complete' if args.smoke_only else 'complete',
                      shape=list(array.shape),encoding_seconds=time.perf_counter()-inference_start,
                      peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
                      features_sha256=hashlib.sha256((out/'image_features.npy').read_bytes()).hexdigest())
    except Exception as exc:
        report['status']='failed';report['error']=str(exc);raise
    finally:
        report['total_seconds']=round(time.perf_counter()-start,3)
        (out/'cache.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
