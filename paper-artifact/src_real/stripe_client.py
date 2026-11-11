"""Minimal Stripe v1 client for the real-API validation slice (stdlib only).

Safety (hard checks, not options):
  * the key must start with "sk_test_" (test mode); anything else raises
    UnsafeKeyError before a single request is built;
  * every JSON object Stripe returns is checked for "livemode": true, which
    also raises UnsafeKeyError (second line of defence);
  * the key never appears in a message, repr, log line or exception: it is
    sent only in the Authorization header and redacted from any error text.

Transport rules (verified against docs.stripe.com, 2026-10-26):
  * Idempotency-Key header on every POST (v1: keys <= 255 chars, results
    saved once endpoint execution begins, replays return the saved status
    and body, a reused key with different parameters is an error, a
    concurrent duplicate is a 409 that is safe to retry).
  * 429 = rate limited (sandbox: 25 req/s global, 20 req/s for search):
    exponential backoff with jitter, same Idempotency-Key.
  * Connection errors / timeouts: retried with the SAME key, so a request
    that reached Stripe is replayed, never executed twice.
  * 5xx from Stripe: v1 saves a 500 under the key, so a retry cannot help;
    raised as StripeInfraError (the runner aborts the episode, writes no
    row, and the rerun starts a fresh attempt with fresh keys).
  * Stripe-Version is pinned per request for reproducibility.
"""
from __future__ import annotations

import http.client
import json
import os
import random
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

LIVE_BASE = "https://api.stripe.com"
DEFAULT_API_VERSION = "2024-06-20"
TEST_PREFIX = "sk_test_"


class UnsafeKeyError(RuntimeError):
    """Not a test-mode secret key, or Stripe returned a live-mode object."""


class StripeInfraError(RuntimeError):
    """Transport failure or Stripe 5xx: not model behaviour; abort episode."""


class StripeAPIError(RuntimeError):
    """A 4xx answer from Stripe (other than 429/409, which are retried)."""

    def __init__(self, status, err: dict):
        self.status = status
        self.err = err or {}
        self.type = self.err.get("type")
        self.code = self.err.get("code")
        super().__init__(f"HTTP {status} {self.type or ''}: "
                         f"{self.err.get('message', '')}".strip())


def is_param_mismatch(err: dict) -> bool:
    """idempotency_error because the key was reused with DIFFERENT
    parameters (the status code is not documented; 400 or 409). A 409 for a
    concurrent in-flight duplicate is retried instead."""
    return (err.get("type") == "idempotency_error"
            and "parameters" in str(err.get("message", "")).lower())


def assert_test_key(key) -> None:
    if not isinstance(key, str) or not key.startswith(TEST_PREFIX) \
            or len(key) <= len(TEST_PREFIX):
        raise UnsafeKeyError(
            "refusing to run: STRIPE_SECRET_KEY must be a Stripe TEST-mode "
            "secret key (starts with sk_test_). Live keys, restricted keys and "
            "empty keys are rejected.")


def encode_form(params: dict, prefix: str = "") -> list:
    """Stripe form encoding: nested dicts as a[b]=c, bools as true/false."""
    out = []
    for k, v in params.items():
        name = f"{prefix}[{k}]" if prefix else str(k)
        if isinstance(v, dict):
            out += encode_form(v, name)
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                out.append((f"{name}[{i}]", _scalar(x)))
        elif v is not None:
            out.append((name, _scalar(v)))
    return out


def _scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _check_live(obj) -> None:
    if isinstance(obj, dict):
        if obj.get("livemode") is True:
            raise UnsafeKeyError("Stripe returned a LIVE-mode object; aborting")
        for v in obj.values():
            if isinstance(v, (dict, list)):
                _check_live(v)
    elif isinstance(obj, list):
        for v in obj:
            _check_live(v)


class StripeClient:
    def __init__(self, key=None, base=None, api_version=None, max_retries=6,
                 timeout=40.0, min_interval=0.05, sleep=time.sleep):
        key = os.environ.get("STRIPE_SECRET_KEY", "") if key is None else key
        assert_test_key(key)
        self.__key = key
        self.base = (base or os.environ.get("BENCH_STRIPE_BASE") or LIVE_BASE).rstrip("/")
        self.api_version = api_version or DEFAULT_API_VERSION
        self.max_retries = max_retries
        self.timeout = timeout
        self.min_interval = min_interval
        self.sleep = sleep
        self._last = 0.0
        self.stats = {"requests": 0, "retries_429": 0, "retries_409": 0,
                      "retries_transport": 0, "replayed": 0}

    def __repr__(self):
        return f"StripeClient(base={self.base!r}, key=sk_test_***)"

    def redact(self, text) -> str:
        return str(text).replace(self.__key, "sk_test_***")

    # -- transport -------------------------------------------------------

    def _throttle(self):
        dt = time.monotonic() - self._last
        if dt < self.min_interval:
            self.sleep(self.min_interval - dt)
        self._last = time.monotonic()

    def request(self, method: str, path: str, params=None, idem_key=None):
        """-> (status, body dict, headers dict lower-cased)."""
        assert method in ("GET", "POST")
        if method == "POST" and not idem_key:
            raise ValueError("every POST carries an Idempotency-Key")
        if idem_key is not None and len(idem_key) > 255:
            raise ValueError("Idempotency-Key longer than 255 characters")
        form = urllib.parse.urlencode(encode_form(params or {}))
        url = self.base + path
        data = None
        if method == "GET" and form:
            url += "?" + form
        elif method == "POST":
            data = form.encode()
        headers = {"Authorization": f"Bearer {self.__key}",
                   "Stripe-Version": self.api_version,
                   "User-Agent": "Benchmark-real/1"}
        if method == "POST":
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            headers["Idempotency-Key"] = idem_key
        delay = 0.5
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.stats["requests"] += 1
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    status = r.status
                    hdrs = {k.lower(): v for k, v in r.headers.items()}
                    body = json.loads(r.read() or b"{}")
            except urllib.error.HTTPError as e:
                status = e.code
                hdrs = {k.lower(): v for k, v in (e.headers or {}).items()}
                try:
                    body = json.loads(e.read() or b"{}")
                except ValueError:
                    body = {}
            except (urllib.error.URLError, socket.timeout, TimeoutError,
                    http.client.HTTPException, ConnectionError, OSError) as e:
                if attempt < self.max_retries:
                    self.stats["retries_transport"] += 1
                    self.sleep(delay * (1 + random.random())); delay *= 2
                    continue
                raise StripeInfraError(self.redact(f"Stripe unreachable: {e!r}"))
            _check_live(body)
            err = body.get("error") or {}
            if status == 409 and is_param_mismatch(err):
                status = 400   # a reused key with different parameters: never retried
            if status == 429 or status == 409:
                if attempt < self.max_retries:
                    self.stats["retries_429" if status == 429 else "retries_409"] += 1
                    ra = hdrs.get("retry-after")
                    wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else delay
                    self.sleep(wait * (1 + random.random())); delay *= 2
                    continue
                raise StripeInfraError(f"Stripe HTTP {status} after {attempt} retries")
            if status >= 500:
                raise StripeInfraError(self.redact(
                    f"Stripe HTTP {status}: {(body.get('error') or {}).get('message', '')}"))
            if status == 401:
                raise UnsafeKeyError("Stripe rejected the key (HTTP 401)")
            if status >= 400:
                raise StripeAPIError(status, {k: self.redact(v) if isinstance(v, str) else v
                                              for k, v in (body.get("error") or {}).items()})
            if hdrs.get("idempotent-replayed") == "true":
                self.stats["replayed"] += 1
            return status, body, hdrs
        raise StripeInfraError("unreachable")

    # -- endpoints used by the slice ------------------------------------

    def create_customer(self, attempt_id: str, metadata: dict) -> dict:
        _, body, _ = self.request("POST", "/v1/customers", {
            "name": f"Benchmark {attempt_id}",
            "description": "Benchmark real-API validation (test mode)",
            "metadata": metadata}, idem_key=f"ib-{attempt_id}-customer")
        return body

    def create_payment_intent(self, params: dict, idem_key: str):
        """-> (PaymentIntent dict, replayed: bool)."""
        _, body, hdrs = self.request("POST", "/v1/payment_intents", params,
                                     idem_key=idem_key)
        return body, hdrs.get("idempotent-replayed") == "true"

    def list_payment_intents(self, customer: str) -> list:
        """All PaymentIntents of a customer (list API: read-after-write
        consistent, newest first), following has_more."""
        out, after = [], None
        while True:
            p = {"customer": customer, "limit": 100}
            if after:
                p["starting_after"] = after
            _, body, _ = self.request("GET", "/v1/payment_intents", p)
            data = body.get("data") or []
            out += data
            if not body.get("has_more") or not data:
                return out
            after = data[-1]["id"]

    def search_payment_intents(self, query: str) -> list:
        """Search API (eventually consistent, < 1 min normally): used only
        for the delayed cross-check, never for read-after-write."""
        out, page = [], None
        while True:
            p = {"query": query, "limit": 100}
            if page:
                p["page"] = page
            _, body, _ = self.request("GET", "/v1/payment_intents/search", p)
            out += body.get("data") or []
            if not body.get("has_more") or not body.get("next_page"):
                return out
            page = body["next_page"]
