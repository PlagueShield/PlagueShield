"""Scheduled X publishing with a durable, conservative thread outbox."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import unicodedata
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import httpx
from authlib.integrations.httpx_client import OAuth1Client

from plagueshield.data import load_case
from server.worker import AGENT_NAMES, CORE_AGENT_NAMES

PUBLISHING_POLICY = (
    "Publish only completed bundled research records. Summarize the recorded "
    "findings of every research agent without inventing results or clinical advice. "
    "Label synthetic/reconstructed provenance and experimental research. "
    "Use ASCII text within standard post limits. Never publish case input records, "
    "credentials, private prompts, or patient identifiers. Link to the public research site. "
    "Skip duplicate findings. No likes, follows, mentions, or unsolicited replies."
)


def ascii_excerpt(value: str, limit: int) -> str:
    text = ' '.join(unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode().split())
    text = text.replace('@', '').replace('#', '').replace('*', '').replace('https://', '').replace('http://', '')
    return text if len(text) <= limit else text[:limit - 3].rstrip() + '...'


def compose_thread(record: dict, public_url: str) -> list[str]:
    case = load_case(record['case_id'])
    verdicts = record.get('verdicts') or {}
    origin = case.provenance.origin.value
    provenance = 'Synthetic vignette' if origin == 'synthetic' else 'Reconstructed research vignette'
    names = [name for name in AGENT_NAMES if name in verdicts]
    root = f"PlagueShield | {case.case_id}\n{provenance}. Automated experimental research, not clinical advice.\n{len(names)}-agent findings:"
    if public_url:
        root += f"\n{public_url}/cases"
    lines = []
    for name in names:
        verdict = verdicts.get(name, {})
        data = verdict.get('data') or {}
        finding = verdict.get('abstain_reason') if verdict.get('abstained') else data.get('synthesis') or verdict.get('headline')
        if not verdict.get('abstained'):
            if name == 'uncertainty':
                score = data.get('reliability')
                finding = f'Uncalibrated heuristic score: {score:.0%}; not accuracy' if isinstance(score, (int, float)) else 'Heuristic confidence; not measured accuracy'
            elif name == 'meta_review':
                review = data.get('review') or {}
                finding = next(iter(review.get('methodology_findings') or []), None) or review.get('summary') or finding
            elif name == 'summary':
                finding = next(iter(verdict.get('rationale') or []), finding)
            elif name == 'analysis':
                finding = next((line.strip() for line in str(finding).splitlines() if line.strip() and not line.lstrip().startswith('#')), finding)
            elif name == 'diagnostic':
                finding = 'Model estimate: ' + str(finding)
        prefix = name.replace('_', '-') + (': ABSTAINED ' if verdict.get('abstained') else ': ')
        lines.append(prefix + ascii_excerpt(finding or 'No published finding.', 82 - len(prefix)))
    posts = [root, *['\n'.join(lines[index:index + 3]) for index in range(0, len(lines), 3)]]
    if any(len(post) > 280 or not post.isascii() for post in posts):
        raise ValueError('Thread exceeds conservative post limits')
    return posts


class XAPIError(RuntimeError):
    def __init__(self, message: str, *, uncertain: bool = False, retry_at: float = 0):
        super().__init__(message)
        self.uncertain = uncertain
        self.retry_at = retry_at


class XPublisher:
    def __init__(self, records: Callable[[], list[dict]], store: Path, *, client_factory=None):
        self.records = records
        self.store = store
        self.client_factory = client_factory
        self.enabled = os.getenv('PLAGUESHIELD_X_ENABLED', 'false').lower() == 'true'
        self.dry_run = os.getenv('PLAGUESHIELD_X_DRY_RUN', 'true').lower() != 'false'
        self.interval = max(300, int(os.getenv('PLAGUESHIELD_X_INTERVAL_SECONDS', '300')))
        self.account = os.getenv('PLAGUESHIELD_X_ACCOUNT', 'plagueshieldX').lstrip('@')
        self.public_url = os.getenv('PLAGUESHIELD_PUBLIC_URL', '').rstrip('/')
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.RLock()
        self._next = 0.0
        self._state = 'disabled'
        self._error = None
        with self._db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS batches (fingerprint TEXT PRIMARY KEY, case_id TEXT, created REAL, mode TEXT, status TEXT, posts TEXT, ids TEXT, error TEXT, retry_at REAL DEFAULT 0)')
            # A crash after sending can leave the result unknown: never replay blindly.
            db.execute("UPDATE batches SET status='uncertain', error='Process stopped during a send; reconcile with X before resuming.' WHERE status='sending'")

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.store)
        try:
            with db:
                yield db
        finally:
            db.close()

    def configured(self) -> bool:
        return all(os.getenv(key) for key in ('X_API_KEY', 'X_API_KEY_SECRET', 'X_ACCESS_TOKEN', 'X_ACCESS_TOKEN_SECRET'))

    def _client(self):
        if self.client_factory:
            return self.client_factory()
        return OAuth1Client(
            os.environ['X_API_KEY'], client_secret=os.environ['X_API_KEY_SECRET'],
            token=os.environ['X_ACCESS_TOKEN'], token_secret=os.environ['X_ACCESS_TOKEN_SECRET'],
            force_include_body=True, timeout=20.0,
        )

    def snapshot(self) -> dict:
        with self._lock, self._db() as db:
            rows = db.execute('SELECT case_id, created, mode, status, posts, ids, error FROM batches ORDER BY created DESC LIMIT 20').fetchall()
            history = [{
                'case_id': row[0], 'created_at': datetime.fromtimestamp(row[1], timezone.utc).isoformat(),
                'mode': row[2], 'status': row[3], 'posts': json.loads(row[4]),
                'post_urls': [f'https://x.com/i/web/status/{post_id}' for post_id in json.loads(row[5])],
                'error': row[6],
            } for row in rows]
            return {
                'enabled': self.enabled, 'dry_run': self.dry_run, 'configured': self.configured(),
                'account': self.account, 'interval_seconds': self.interval,
                'status': self._state, 'error': self._error,
                'next_check_at': datetime.fromtimestamp(self._next, timezone.utc).isoformat() if self._next else None,
                'alive': bool(self._thread and self._thread.is_alive()), 'history': history,
                'policy': PUBLISHING_POLICY,
            }

    def preview(self) -> dict | None:
        record = self._latest()
        if not record:
            return None
        return {'case_id': record['case_id'], 'posts': compose_thread(record, self.public_url)}

    def _latest(self) -> dict | None:
        for record in sorted(self.records(), key=lambda row: row.get('assessed_at', ''), reverse=True):
            if record.get('_publisher_source') != 'research_worker':
                continue
            if not all(name in (record.get('verdicts') or {}) for name in CORE_AGENT_NAMES):
                continue
            try:
                load_case(record['case_id'])
            except (FileNotFoundError, KeyError):
                continue
            verdicts = record.get('verdicts') or {}
            # Do not fall back to stale successful runs when the latest run failed.
            if any(v.get('abstained', True) for name in ('analysis', 'meta_review') for v in [verdicts.get(name, {})]):
                return None
            if not verdicts['analysis'].get('data', {}).get('synthesis') or not verdicts['meta_review'].get('data', {}).get('review', {}).get('summary'):
                return None
            if any(flag.get('code') == 'AGENT_FAILURE' for verdict in verdicts.values() for flag in verdict.get('flags', [])):
                return None
            return record
        return None

    @staticmethod
    def _check(response: httpx.Response, *, write: bool = False):
        if response.status_code == 429:
            try:
                retry_at = max(time.time() + 300, float(response.headers.get('x-rate-limit-reset', 0)), time.time() + float(response.headers.get('retry-after', 0)))
            except ValueError:
                retry_at = time.time() + 300
            raise XAPIError('X rate limit reached; waiting before retry.', retry_at=retry_at)
        if response.status_code >= 400:
            raise XAPIError(f'X API HTTP {response.status_code}; check account permissions and API access.', uncertain=write and response.status_code >= 500)

    def tick(self):
        with self._lock:
            if not self.enabled:
                self._state = 'disabled'
                return
            if not self.dry_run and (not self.configured() or not self.public_url):
                self._state = 'configuration_required'
                self._error = 'Live posting requires X user credentials and PLAGUESHIELD_PUBLIC_URL.'
                return
            if self.public_url:
                parsed = urlparse(self.public_url)
                if parsed.scheme != 'https' or not parsed.hostname or parsed.username or len(self.public_url) > 100 or not self.public_url.isascii():
                    self._state = 'configuration_required'
                    self._error = 'Public site URL must be an ASCII HTTPS URL of at most 100 characters.'
                    return
            mode = 'dry_run' if self.dry_run else 'live'
            with self._db() as db:
                if db.execute("SELECT 1 FROM batches WHERE mode=? AND status IN ('uncertain','sending','blocked')", (mode,)).fetchone():
                    self._state = 'reconciliation_required'
                    return
                batch = db.execute("SELECT fingerprint, posts, ids, retry_at FROM batches WHERE mode=? AND status='ready' ORDER BY created LIMIT 1", (mode,)).fetchone()
                if not batch:
                    record = self._latest()
                    if not record:
                        self._state = 'waiting_for_research'
                        return
                    posts = compose_thread(record, self.public_url)
                    fingerprint = hashlib.sha256(json.dumps([mode, self.account, posts]).encode()).hexdigest()
                    if db.execute('SELECT 1 FROM batches WHERE fingerprint=?', (fingerprint,)).fetchone():
                        self._state = 'no_new_findings'
                        return
                    db.execute('INSERT INTO batches (fingerprint, case_id, created, mode, status, posts, ids) VALUES (?,?,?,?,?,?,?)', (fingerprint, record['case_id'], time.time(), mode, 'ready', json.dumps(posts), '[]'))
                    batch = (fingerprint, json.dumps(posts), '[]', 0)
            fingerprint, serialized_posts, serialized_ids, retry_at = batch
            if time.time() < retry_at:
                self._state = 'rate_limited'
                return
            posts, ids = json.loads(serialized_posts), json.loads(serialized_ids)
            if self.dry_run:
                with self._db() as db:
                    db.execute("UPDATE batches SET status='previewed' WHERE fingerprint=?", (fingerprint,))
                self._state, self._error = 'previewed', None
                return
            self._state = 'publishing'
        # Network requests must not hold the public snapshot lock.
        sending = False
        try:
            with self._client() as client:
                identity = client.get('https://api.x.com/2/users/me')
                self._check(identity)
                if identity.json().get('data', {}).get('username', '').lower() != self.account.lower():
                    raise XAPIError('Authenticated X account does not match PLAGUESHIELD_X_ACCOUNT.')
                for text in posts[len(ids):]:
                    if self._stop.is_set():
                        return
                    payload = {'text': text}
                    if ids:
                        payload['reply'] = {'in_reply_to_tweet_id': ids[-1]}
                    with self._db() as db:
                        db.execute("UPDATE batches SET status='sending' WHERE fingerprint=?", (fingerprint,))
                    sending = True
                    response = client.post('https://api.x.com/2/tweets', json=payload)
                    self._check(response, write=True)
                    post_id = response.json().get('data', {}).get('id')
                    if not isinstance(post_id, str) or not post_id.isdigit():
                        raise XAPIError('X did not return a valid post ID; reconcile before resuming.', uncertain=True)
                    ids.append(post_id)
                    with self._db() as db:
                        db.execute("UPDATE batches SET ids=?, status='ready' WHERE fingerprint=?", (json.dumps(ids), fingerprint))
                    sending = False
            with self._db() as db:
                db.execute("UPDATE batches SET status='published', error=NULL WHERE fingerprint=?", (fingerprint,))
            with self._lock:
                self._state, self._error = 'published', None
        except (XAPIError, httpx.RequestError, ValueError) as exc:
            uncertain = getattr(exc, 'uncertain', False) or (sending and not isinstance(exc, XAPIError))
            retry_at = getattr(exc, 'retry_at', 0)
            status = 'uncertain' if uncertain else 'ready' if retry_at else 'blocked'
            error = str(exc) if isinstance(exc, XAPIError) else 'X request failed; reconcile any unconfirmed sends before retrying.'
            with self._db() as db:
                db.execute('UPDATE batches SET status=?, error=?, retry_at=? WHERE fingerprint=?', (status, error, retry_at, fingerprint))
            with self._lock:
                self._state, self._error = status, error

    def start(self):
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._state = 'waiting'
        self._thread = threading.Thread(target=self._loop, name='plagueshield-x-publisher', daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=100)

    def _loop(self):
        # Wait one full interval before the first externally visible post.
        while not self._stop.is_set():
            with self._lock:
                self._next = time.time() + self.interval
            if self._stop.wait(self.interval):
                break
            try:
                self.tick()
            except Exception:
                with self._lock:
                    self._state, self._error = 'error', 'Publishing cycle failed; inspect backend configuration.'
