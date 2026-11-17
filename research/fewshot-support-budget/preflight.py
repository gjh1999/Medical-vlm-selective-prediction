"""Audit environment and official UniverSeg sources without installing ML packages."""
import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {
        'utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'platform': platform.platform(),
        'modules': {name: bool(importlib.util.find_spec(name))
                    for name in ['torch', 'numpy', 'PIL']},
        'sources': {}, 'model_inference_run': False,
    }
    try:
        proc = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=15)
        report['gpu'] = {'exit_code': proc.returncode, 'stdout': proc.stdout,
                         'stderr': proc.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        report['gpu'] = {'error': str(exc)}

    def fetch(name, url):
        tick = time.perf_counter()
        entry = {'url': url}
        report['sources'][name] = entry
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'SupportBudgetResearch/0.1'})
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = response.read(4 * 1024 * 1024 + 1)
                if len(payload) > 4 * 1024 * 1024:
                    raise ValueError('Source exceeds 4 MiB metadata limit')
                entry.update(status=response.status, final_url=response.url,
                             bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            (out / name).write_bytes(payload)
            return payload
        except (OSError, ValueError) as exc:
            entry['error'] = str(exc)
            return None
        finally:
            entry['seconds'] = round(time.perf_counter() - tick, 3)

    if args.network:
        raw = fetch('universeg_commit.json', 'https://api.github.com/repos/JJGO/UniverSeg/commits/main')
        if raw:
            try:
                revision = json.loads(raw)['sha']
                if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
                    raise ValueError('Invalid revision')
                report['universeg_revision'] = revision
                for filename in ['README.md', 'requirements.txt', 'LICENSE-model', 'universeg/model.py']:
                    fetch(filename.replace('/', '_'),
                          f'https://raw.githubusercontent.com/JJGO/UniverSeg/{revision}/{filename}')
                fetch('universeg_tree.json',
                      f'https://api.github.com/repos/JJGO/UniverSeg/git/trees/{revision}?recursive=1')
            except (ValueError, KeyError, TypeError) as exc:
                report['revision_error'] = str(exc)
        fetch('context_matters.html', 'https://arxiv.org/html/2608.05333v1')
    report['elapsed_seconds'] = round(time.perf_counter() - start, 3)
    failed = any('error' in item for item in report['sources'].values()) or 'revision_error' in report
    report['status'] = 'partial_failure' if failed else ('source_audit_complete' if args.network else 'environment_only')
    (out / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 2 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
