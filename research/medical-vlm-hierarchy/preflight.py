"""Record real local hardware/software and original-source access; no ML install."""
import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SOURCES = {
    'ham_dataverse': 'https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/DBW86T',
    'pad_mendeley': 'https://data.mendeley.com/public-api/datasets/zr7vgbcyr2/versions/1',
    'biomedclip_model': 'https://huggingface.co/api/models/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
              'executable': sys.executable, 'platform': platform.platform(),
              'cpu_count': os.cpu_count(), 'free_disk_bytes': shutil.disk_usage(out.resolve()).free,
              'target_gpu': 'RTX 5080 on separate home workstation; not inspected',
              'modules': {m: bool(importlib.util.find_spec(m)) for m in ['torch','numpy','PIL','scipy','sklearn']},
              'network': {}}
    try:
        r = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=15)
        report['nvidia_smi'] = {'exit_code': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        report['nvidia_smi'] = {'error': str(exc)}
    if args.network:
        for name, url in SOURCES.items():
            then = time.perf_counter()
            try:
                request = urllib.request.Request(url, headers={'User-Agent':'MedicalVLMResearch/0.1'})
                with urllib.request.urlopen(request, timeout=25) as response:
                    payload = response.read(8*1024*1024 + 1)
                    if len(payload) > 8*1024*1024:
                        raise ValueError('Metadata response exceeds 8 MiB')
                    parsed = json.loads(payload)
                    (out / f'{name}.json').write_text(json.dumps(parsed,ensure_ascii=False,indent=2),encoding='utf-8')
                    report['network'][name] = {'url':url,'status':response.status,'bytes':len(payload),'format':'json'}
            except (OSError, ValueError) as exc:
                report['network'][name] = {'url':url,'error':str(exc)}
            report['network'][name]['seconds'] = round(time.perf_counter()-then,3)
    report['elapsed_seconds'] = round(time.perf_counter()-start,3)
    (out / 'environment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))
    if args.network and any('error' in v for v in report['network'].values()):
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
