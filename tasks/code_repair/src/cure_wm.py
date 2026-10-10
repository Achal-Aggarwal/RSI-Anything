from __future__ import annotations
import os, resource, shutil, signal, subprocess, sys, tempfile, time
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class TestCase:
    input: str
    expected: str
    test_id: str

@dataclass(frozen=True)
class TestResult:
    test_id: str
    status: str
    actual: str
    expected: str
    stderr: str
    seconds: float

@dataclass(frozen=True)
class Execution:
    results: tuple[TestResult, ...]
    wall_seconds: float

    @property
    def all_pass(self) -> bool:
        return bool(self.results) and all((r.status == 'AC' for r in self.results))

    @property
    def pass_fraction(self) -> float:
        return sum((r.status == 'AC' for r in self.results)) / len(self.results) if self.results else 0.0

@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    code: str

def _limits():
    if sys.platform != 'linux':
        return
    resource.setrlimit(resource.RLIMIT_CPU, (10, 11))
    resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 ** 2, 1024 ** 2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

def _run_one(script: Path, test: TestCase, timeout: float, sandbox: str) -> TestResult:
    started = time.monotonic()
    if sandbox == 'required':
        if not shutil.which('runuser') or not shutil.which('unshare'):
            raise RuntimeError('Required runuser/unshare sandbox is unavailable')
        cmd = [shutil.which('runuser'), '-u', 'nobody', '--', shutil.which('unshare'), '-Urn', '--', sys.executable, '-I', str(script)]
    elif sandbox == 'local_test':
        cmd = [sys.executable, '-I', str(script)]
    else:
        raise ValueError(f'unknown sandbox: {sandbox}')
    env = {'PATH': '/usr/bin:/bin', 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1'}
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True, preexec_fn=_limits, env=env)
    try:
        stdout, stderr = proc.communicate(test.input, timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        return TestResult(test.test_id, 'TLE', '', test.expected, 'timeout', time.monotonic() - started)
    elapsed = time.monotonic() - started
    if proc.returncode:
        return TestResult(test.test_id, 'RE', stdout[-1024:], test.expected, stderr[-2048:], elapsed)
    status = 'AC' if stdout.strip() == test.expected.strip() else 'WA'
    return TestResult(test.test_id, status, stdout[-1024:], test.expected, stderr[-2048:], elapsed)

def execute(code: str, tests: tuple[TestCase, ...], *, timeout: float=3.0, sandbox: str='required') -> Execution:
    started = time.monotonic()
    if not tests:
        raise ValueError('Cannot execute an empty test suite')
    if not code.strip():
        results = tuple((TestResult(t.test_id, 'CE', '', t.expected, 'Empty candidate program', 0.0) for t in tests))
        return Execution(results, time.monotonic() - started)
    try:
        compile(code, 'candidate.py', 'exec')
    except SyntaxError as exc:
        msg = f'{type(exc).__name__}: {exc.msg} (line {exc.lineno})'
        results = tuple((TestResult(t.test_id, 'CE', '', t.expected, msg, 0.0) for t in tests))
        return Execution(results, time.monotonic() - started)
    with tempfile.TemporaryDirectory(prefix='cure_wm_') as tmp:
        tmp_dir = Path(tmp)
        tmp_dir.chmod(493)
        script = tmp_dir / 'candidate.py'
        script.write_text(code, encoding='utf-8')
        script.chmod(420)
        results = tuple((_run_one(script, test, timeout, sandbox) for test in tests))
    return Execution(results, time.monotonic() - started)
