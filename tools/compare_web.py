"""Local Loom / ComfyUI comparison UI. Run: python tools/compare_web.py"""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse
import uuid

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'build/comparison-ui'
CONTAINER = 'amd-strix-halo-comfyui'
LOCK = threading.Lock()
BUSY = threading.Lock()
JOBS = {}


def update(job, **values):
    with LOCK:
        job.update(values)
        path = OUTPUT / job['id'] / 'job.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(job, indent=2))
        temporary.replace(path)


def run_job(job):
    directory = OUTPUT / job['id']
    try:
        update(job, status='comfy', message='Generating ComfyUI image…')
        with (directory / 'comfy.log').open('w') as log:
            subprocess.run(['podman', 'start', CONTAINER], stdout=log, stderr=subprocess.STDOUT, check=True)
            command = ['podman', 'exec', '--user', 'zach', '-e',
                       f'PYTHONPATH={ROOT}/build/comfy-bench-deps', '-w',
                       str(Path.home() / 'code/ComfyUI'), CONTAINER,
                       '/opt/venv/bin/python', str(ROOT / 'tools/bench_comfyui.py'),
                       '--runs', '1', '--size', str(job['size']), '--steps', str(job['steps']),
                       '--seed', str(job['seed']), '--prompt', job['prompt'],
                       '--output-dir', str(directory)]
            started = time.monotonic()
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        comfy = json.loads((directory / 'comfy.json').read_text())
        comfy['wall_seconds'] = time.monotonic() - started
        update(job, status='loom', message='ComfyUI ready. Generating Loom image…', comfy=comfy)
        environment = os.environ.copy()
        environment.pop('LD_LIBRARY_PATH', None)
        environment.pop('KREA2_NATIVE_PROFILE', None)
        environment.setdefault('LOOM_COMPILE', str(Path.home() /
            'code/hrx-system/build-cuda/loom/src/loom/tools/loom-compile/loom-compile'))
        with (directory / 'loom.log').open('w') as log:
            started = time.monotonic()
            subprocess.run([sys.executable, str(ROOT / 'tools/compare_native.py'), str(directory)],
                           env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        loom = json.loads((directory / 'loom.json').read_text())
        loom['wall_seconds'] = time.monotonic() - started
        if loom['noise_sha256'] != comfy['noise_sha256']:
            raise RuntimeError('Initial noise checksums do not match')
        update(job, status='done', message='Both images ready · identical initial noise', loom=loom)
    except Exception as error:
        stage = job['status']
        log = directory / f'{stage}.log'
        details = log.read_text(errors='replace')[-5000:] if log.exists() else ''
        update(job, status='error', message=str(error), error=details)
    finally:
        BUSY.release()


def validate(data):
    prompt = data.get('prompt')
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
        raise ValueError('Enter a prompt of 1–8000 characters.')
    for field in ('size', 'steps', 'seed'):
        if type(data.get(field)) is not int:
            raise ValueError(f'{field} must be an integer.')
    if data['size'] not in (256, 512, 768, 1024, 1536, 2048):
        raise ValueError('Choose a supported image size.')
    if not 1 <= data['steps'] <= 100 or not 0 <= data['seed'] <= 2**53 - 1:
        raise ValueError('Steps must be 1–100 and seed 0–9007199254740991.')
    return dict(prompt=prompt.strip(), size=data['size'], steps=data['steps'], seed=data['seed'])


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, data, content_type='application/json'):
        if not isinstance(data, bytes):
            data = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            return self.reply(200, (ROOT / 'tools/compare_ui.html').read_bytes(), 'text/html; charset=utf-8')
        if path == '/api/jobs':
            with LOCK:
                payload = json.dumps(sorted(JOBS.values(), key=lambda j: j['created'], reverse=True)).encode()
            return self.reply(200, payload)
        match = re.fullmatch(r'/images/([a-f0-9]{16})/(loom|comfy)\.png', path)
        if match:
            file = OUTPUT / match[1] / f'{match[2]}.png'
            if file.is_file():
                return self.reply(200, file.read_bytes(), 'image/png')
        self.reply(404, {'error': 'Not found'})

    def do_POST(self):
        if self.path != '/api/jobs':
            return self.reply(404, {'error': 'Not found'})
        # Only same-origin browser requests can start expensive GPU work.
        origin = self.headers.get('Origin')
        if origin and urlparse(origin).netloc != self.headers.get('Host'):
            return self.reply(403, {'error': 'Origin mismatch'})
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            return self.reply(415, {'error': 'Expected application/json'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 40000:
                raise ValueError('Invalid request length')
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError('Expected a JSON object')
            values = validate(data)
        except (ValueError, TypeError) as error:
            return self.reply(400, {'error': str(error)})
        if not BUSY.acquire(blocking=False):
            return self.reply(409, {'error': 'A comparison is already running.'})
        try:
            job = dict(id=uuid.uuid4().hex[:16], created=datetime.now(timezone.utc).isoformat(),
                       status='queued', message='Starting comparison…', **values)
            (OUTPUT / job['id']).mkdir()
            update(job)
            with LOCK:
                JOBS[job['id']] = job
            threading.Thread(target=run_job, args=(job,), daemon=True).start()
        except Exception:
            BUSY.release()
            raise
        self.reply(202, {'id': job['id']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=7865)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT.glob('*/job.json'):
        job = json.loads(path.read_text())
        if job['status'] not in ('done', 'error'):
            update(job, status='error', message='Server restarted before this comparison finished.')
        JOBS[job['id']] = job
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Comparison UI: http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
