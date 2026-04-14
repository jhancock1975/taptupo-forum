"""Locust load-test scenario for taptupo-forum.

A single user-class simulates browsing: hit the index, pick a random
thread from ``/api/threads``, and read it. Writes are deliberately
omitted - they require a session cookie and registering users during
load tests creates noise. If you need write-path load testing, register
a pool of users out-of-band and import session cookies here.

Run with::

    uv pip install locust
    uv run locust -f tests/load/locustfile.py --host http://localhost:8000
"""

from __future__ import annotations

import random

from locust import (  # type: ignore[import-not-found,import-untyped,unused-ignore]
    HttpUser,
    between,
    task,
)


class ForumBrowser(HttpUser):
    """Emulates an anonymous reader scrolling the forum."""

    wait_time = between(1.0, 3.0)

    @task(3)
    def browse_index(self) -> None:
        self.client.get("/")

    @task(2)
    def browse_agents(self) -> None:
        self.client.get("/agents")

    @task(5)
    def browse_random_thread(self) -> None:
        resp = self.client.get("/api/threads")
        if resp.status_code != 200:
            return
        threads = resp.json().get("threads", [])
        if not threads:
            return
        t = random.choice(threads)  # noqa: S311  # nosec B311 — not security-sensitive
        self.client.get(f"/threads/{t['thread_id']}", name="/threads/[id]")
