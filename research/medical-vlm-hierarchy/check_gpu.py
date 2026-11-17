"""On-home-workstation CUDA acceptance check. Does not start model experiments."""
import argparse
import json
import platform
import time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    report={'status':'in_progress','python':platform.python_version(),'os':platform.platform()}
    start=time.perf_counter()
    try:
        import torch
        report.update(torch=torch.__version__,cuda_build=torch.version.cuda,cuda_available=torch.cuda.is_available())
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; stopping without CPU fallback')
        report.update(gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
                      compiled_architectures=torch.cuda.get_arch_list(),total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
        torch.manual_seed(17);torch.cuda.reset_peak_memory_stats()
        a=torch.randn(256,256,device='cuda');b=a@a.T;torch.cuda.synchronize()
        if not torch.isfinite(b).all():raise RuntimeError('Non-finite CUDA matmul')
        # Compare against FP32 CPU reference; no clinical data or learned model involved.
        reference=a.cpu()@a.cpu().T
        if not torch.allclose(b.cpu(),reference,atol=.02,rtol=.002):raise RuntimeError('CUDA/CPU matmul mismatch')
        report.update(status='passed_cuda_smoke_only',peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        report['seconds']=round(time.perf_counter()-start,3)
        (out/'gpu_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
