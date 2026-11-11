"""In-process fake of the Stripe v1 endpoints the slice uses (tests only).

POST /v1/customers, POST /v1/payment_intents (confirm=true + pm_card_visa ->
"succeeded"), GET /v1/payment_intents?customer= (newest first, limit,
starting_after, has_more), GET /v1/payment_intents/search (metadata clause,
page/next_page, optional indexing delay), with Stripe v1 Idempotency-Key
semantics: the first response is saved per key; a replay with the same
parameters returns it with "Idempotent-Replayed: true" and executes nothing;
the same key with different parameters is a 400 idempotency_error.
Knobs for failure tests: rate_limit_next (429s), error_next (500s),
drop_after_commit_next (commit, then drop the connection before answering),
livemode (return live-mode objects), search_delay_s.
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _nest(pairs):
    """a[b]=c form pairs -> nested dict."""
    out = {}
    for k, v in pairs:
        parts = re.findall(r"[^\[\]]+", k)
        d = out
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return out


class FakeStripe:
    def __init__(self):
        self.lock = threading.Lock()
        self.customers = {}
        self.pis = []                 # creation order
        self.idem = {}                # key -> (params_json, status, body)
        self.requests = []            # (method, path, idem_key)
        self.rate_limit_next = 0
        self.error_next = 0
        self.drop_after_commit_next = 0
        self.livemode = False
        self.search_delay_s = 0.0
        self.auth_seen = set()
        self._n = 0
        self._t = 1_760_000_000
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status, body, headers=None):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _auth(self):
                a = self.headers.get("Authorization", "")
                fake.auth_seen.add(a)
                return a.startswith("Bearer sk_test_")

            def do_GET(self):
                u = urllib.parse.urlparse(self.path)
                q = dict(urllib.parse.parse_qsl(u.query))
                fake.requests.append(("GET", u.path, None))
                if not self._auth():
                    return self._send(401, {"error": {"type": "invalid_request_error",
                                                      "message": "Invalid API Key provided"}})
                if fake.rate_limit_next:
                    fake.rate_limit_next -= 1
                    return self._send(429, {"error": {"type": "invalid_request_error",
                                                      "message": "rate limited"}},
                                      {"Stripe-Rate-Limited-Reason": "global-rate"})
                if u.path == "/v1/payment_intents":
                    with fake.lock:
                        rows = [p for p in reversed(fake.pis)
                                if not q.get("customer") or p["customer"] == q["customer"]]
                    if q.get("starting_after"):
                        ids = [p["id"] for p in rows]
                        rows = rows[ids.index(q["starting_after"]) + 1:]
                    lim = int(q.get("limit", 10))
                    return self._send(200, {"object": "list", "url": u.path,
                                            "has_more": len(rows) > lim, "data": rows[:lim]})
                if u.path == "/v1/payment_intents/search":
                    m = re.fullmatch(r'metadata\["([^"]+)"\]:"([^"]*)"', q.get("query", ""))
                    if not m:
                        return self._send(400, {"error": {"type": "invalid_request_error",
                                                          "message": "bad query"}})
                    now = time.time()
                    with fake.lock:
                        rows = [p for p in reversed(fake.pis)
                                if p["metadata"].get(m.group(1)) == m.group(2)
                                and now - p["_t"] >= fake.search_delay_s]
                    start = int(q.get("page") or 0)
                    lim = int(q.get("limit", 10))
                    page = rows[start:start + lim]
                    more = start + lim < len(rows)
                    return self._send(200, {"object": "search_result", "data": page,
                                            "has_more": more,
                                            "next_page": str(start + lim) if more else None})
                return self._send(404, {"error": {"type": "invalid_request_error",
                                                  "message": "no such path"}})

            def do_POST(self):
                u = urllib.parse.urlparse(self.path)
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n).decode()
                params = _nest(urllib.parse.parse_qsl(raw, keep_blank_values=True))
                key = self.headers.get("Idempotency-Key")
                fake.requests.append(("POST", u.path, key))
                if not self._auth():
                    return self._send(401, {"error": {"type": "invalid_request_error",
                                                      "message": "Invalid API Key provided"}})
                if fake.rate_limit_next:          # not processed, not saved
                    fake.rate_limit_next -= 1
                    return self._send(429, {"error": {"type": "invalid_request_error",
                                                      "message": "rate limited"}},
                                      {"Stripe-Rate-Limited-Reason": "global-rate"})
                pj = json.dumps([u.path, params], sort_keys=True)
                with fake.lock:
                    if key and key in fake.idem:
                        spj, st, body = fake.idem[key]
                        if spj != pj:
                            return self._send(400, {"error": {
                                "type": "idempotency_error",
                                "message": "Keys for idempotent requests can only be used with "
                                           "the same parameters they were first used with."}})
                        return self._send(st, body, {"Idempotent-Replayed": "true"})
                    if fake.error_next:
                        fake.error_next -= 1
                        st, body = 500, {"error": {"type": "api_error", "message": "boom"}}
                    else:
                        st, body = fake._execute(u.path, params)
                    if key:
                        fake.idem[key] = (pj, st, body)
                    drop = fake.drop_after_commit_next and st == 200
                    if drop:
                        fake.drop_after_commit_next -= 1
                if drop:
                    self.close_connection = True
                    self.connection.close()
                    return
                return self._send(st, body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()
        self.server.server_close()

    def _id(self, prefix):
        self._n += 1
        return f"{prefix}_fake{self._n:06d}"

    def _execute(self, path, p):
        self._t += 1
        if path == "/v1/customers":
            c = {"id": self._id("cus"), "object": "customer", "livemode": self.livemode,
                 "name": p.get("name"), "metadata": p.get("metadata", {}), "created": self._t}
            self.customers[c["id"]] = c
            return 200, c
        if path == "/v1/payment_intents":
            try:
                amount = int(p.get("amount", ""))
            except ValueError:
                return 400, {"error": {"type": "invalid_request_error", "param": "amount",
                                       "message": "Invalid integer"}}
            if amount < 50:
                return 400, {"error": {"type": "invalid_request_error", "code": "amount_too_small",
                                       "message": "Amount must be at least $0.50 usd"}}
            if p.get("currency") != "usd":
                return 400, {"error": {"type": "invalid_request_error", "message": "bad currency"}}
            cus = p.get("customer")
            if cus and cus not in self.customers:
                return 400, {"error": {"type": "invalid_request_error", "message": "No such customer"}}
            apm = p.get("automatic_payment_methods") or {}
            if p.get("confirm") == "true" and apm.get("allow_redirects") != "never" \
                    and not p.get("return_url"):
                return 400, {"error": {"type": "invalid_request_error",
                                       "message": "return_url required"}}
            ok = p.get("confirm") == "true" and p.get("payment_method") == "pm_card_visa"
            pi = {"id": self._id("pi"), "object": "payment_intent", "amount": amount,
                  "currency": "usd", "customer": cus, "livemode": self.livemode,
                  "status": "succeeded" if ok else "requires_payment_method",
                  "metadata": p.get("metadata", {}), "description": p.get("description"),
                  "created": self._t, "_t": time.time()}
            self.pis.append(pi)
            return 200, pi
        return 404, {"error": {"type": "invalid_request_error", "message": "no such path"}}

    def committed(self, attempt_id=None):
        return [p for p in self.pis if p["status"] == "succeeded"
                and (attempt_id is None or p["metadata"].get("attempt_id") == attempt_id)]
