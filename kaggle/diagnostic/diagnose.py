"""Isolate Kaggle CPU/GPU memory growth and throughput in fresh processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

WORK = Path('/kaggle/working/diagnostic')
REVISION = '0d210f23b370664d62d43111185ee284e1b1ea44'


def snapshot(pid):
    result = {'time': time.time(), 'pid': pid}
    try:
        result['process'] = {line.split(':')[0]: line.split(':')[1].strip()
                             for line in Path(f'/proc/{pid}/status').read_text().splitlines()
                             if line.startswith(('VmRSS:', 'RssAnon:', 'RssFile:', 'Threads:'))}
    except OSError:
        pass
    for name in ('memory.current', 'memory.max', 'memory.events', 'memory.stat'):
        path = Path('/sys/fs/cgroup') / name
        if path.exists():
            result[name] = path.read_text().strip()
    for name in ('memory.usage_in_bytes', 'memory.limit_in_bytes', 'memory.failcnt', 'memory.oom_control'):
        path = Path('/sys/fs/cgroup/memory') / name
        if path.exists():
            result[name] = path.read_text().strip()
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True)
    result['gpus'] = gpu.stdout.strip()
    return result


def worker(mode, steps):
    import torch
    import torch.distributed as dist
    from model import build_resonance_model, resonance_loss
    rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(rank)
    if mode.endswith('threads2') or mode == 'ddp':
        torch.set_num_threads(2)
    device = torch.device('cuda', rank)
    if mode == 'ddp':
        dist.init_process_group('nccl')
    model = build_resonance_model().to(device)
    if mode.startswith('dp'):
        model = torch.nn.DataParallel(model)
    elif mode == 'ddp':
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    torch.manual_seed(42)
    batch = 4 if mode == 'ddp' else 8
    clean = torch.randn(batch, 2, 128, 256, device=device) / (65536 ** .5)
    noisy = clean + torch.randn_like(clean) * .001
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-7)
    scaler = torch.amp.GradScaler('cuda')
    loss_fn = resonance_loss()
    start = time.monotonic()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            prediction = model(noisy)
        loss = loss_fn(clean, prediction)
        if not torch.isfinite(loss):
            raise FloatingPointError(step)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(optimizer)
        scaler.update()
        if step == 0 or (step + 1) % 10 == 0:
            torch.cuda.synchronize()
            record = {'mode': mode, 'rank': rank, 'step': step + 1,
                      'elapsed': time.monotonic() - start, 'loss': loss.item(),
                      'threads': torch.get_num_threads(),
                      'allocated_mb': torch.cuda.memory_allocated() / 2**20,
                      'reserved_mb': torch.cuda.memory_reserved() / 2**20,
                      **snapshot(os.getpid())}
            print(json.dumps(record), flush=True)
    if mode == 'ddp':
        dist.destroy_process_group()


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    for name in ('model.py',):
        urllib.request.urlretrieve(f'https://raw.githubusercontent.com/GuessWho121/LocandKey/{REVISION}/{name}', WORK / name)
    script = Path(__file__).resolve()
    env = {**os.environ, 'PYTHONPATH': str(WORK), 'PYTHONUNBUFFERED': '1'}
    print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout, flush=True)
    for mode, steps in (('dp', 120), ('dp_threads2', 120), ('single_threads2', 120), ('ddp', 600)):
        command = [sys.executable, '-u', str(script), mode, str(steps)]
        if mode == 'ddp':
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                       str(script), mode, str(steps)]
        with (WORK / f'{mode}.log').open('w') as output:
            process = subprocess.Popen(command, cwd=WORK, env=env, stdout=output, stderr=subprocess.STDOUT)
            started = time.monotonic()
            while process.poll() is None:
                print(json.dumps({'mode': mode, **snapshot(process.pid)}), flush=True)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if time.monotonic() - started > 1200:
                        process.terminate()
                        process.wait(timeout=30)
            print(f'{mode} exit={process.returncode}', flush=True)
        print((WORK / f'{mode}.log').read_text()[-16000:], flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        worker(sys.argv[1], int(sys.argv[2]))
    else:
        main()
