"""
Pairing a phone, in the thread.

WHY THE QR CODE LIVES IN THE CONVERSATION

Forge's web UI is one continuous conversation and nothing else -- no
tabs, no settings page, no second screen to put a pairing dialog on.
So the QR code goes where the interface already is, the same argument
delegation.py makes for listing jobs in the thread rather than behind
GET /jobs.

That choice has a consequence this module is mostly about: anything
printed in the thread is, by default, WRITTEN DOWN. orchestrator._finish
persists the turn to memory.json and indexes it in the vector store, so
a long-lived bearer token rendered as a QR code would be permanently
readable by anyone who opens the UI, and retrievable by a recall months
later. The turn is therefore not remembered (see orchestrator.run), and
what the QR carries is not the bearer token.

WHAT THE QR CARRIES

A single-use pairing token, valid for PAIRING_TTL_SECONDS, which the
client exchanges ONCE against the real bearer token at POST /pair/claim.
Three things follow from "single-use, short-lived":

  - a QR that leaks (a photo, a shoulder, a screenshot) is worth
    nothing after the phone has scanned it, and nothing at all after
    five minutes;
  - a QR left on screen is not a standing credential, so there is no
    "remember to clear the chat" step for the user to forget;
  - the bearer token itself never appears in a rendered message, which
    is what made reusing API_TOKEN directly unshippable.

Held in memory, not on disk. A restart between `!pair` and the scan
costs one retyped command, which is cheaper than a secret written to a
file that outlives the sixty seconds it was needed for. Single-process
is already assumed here -- ratelimit.py makes the same assumption for
the same reason (uvicorn runs without --workers, see Containerfile).

WHAT THIS DELIBERATELY DOES NOT DO

No per-device tokens, no device registry, no revocation endpoint. One
user, one bearer token: a per-device token would only earn its keep the
day losing one device must not invalidate the others, and today there
is exactly one device. Revoking everything is `API_TOKEN=<new value>`
and a restart. The seam is `claim()`, which already returns "the token
this device should use" rather than a constant -- that is where a
per-device token would be minted, and nothing above it would change.
"""

from __future__ import annotations

import base64
import hmac
import io
import json
import secrets
import threading
import time
from urllib.parse import urlparse

from forge.config import API_TOKEN, FORGE_PUBLIC_URL, PAIRING_TTL_SECONDS
from forge.logger import log

#: The only spelling that triggers this. Case-insensitive, surrounding
#: whitespace ignored, nothing else -- an enumerable choice belongs in
#: code, where the wrong token cannot be sampled.
_COMMAND = "!pair"

#: Length gate before any string work, same shape and same reason as
#: delegation._MAX_KEYWORD_CHARS: None is the common case and has to
#: stay cheap. A pasted file is rejected on its length, not lowercased
#: first.
_MAX_COMMAND_CHARS = 16

#: Hosts a phone cannot reach. FORGE_PUBLIC_URL goes into the QR
#: verbatim and becomes the Android client's base URL, so a loopback
#: address here produces a client that talks to the phone itself and
#: fails with a connection error that names neither Forge nor this
#: setting.
_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"})

_lock = threading.Lock()
#: token -> expiry (monotonic seconds). Monotonic, not wall clock: a
#: clock step must not extend or void a pairing window.
_pending: dict[str, float] = {}


class PairingNotConfigured(Exception):
    """FORGE_PUBLIC_URL or API_TOKEN is missing or unusable."""


def _check_config() -> str:
    """The URL to advertise, or raise saying exactly what to set."""
    url = (FORGE_PUBLIC_URL or "").strip()
    if not url:
        raise PairingNotConfigured(
            "FORGE_PUBLIC_URL n'est pas configuré. C'est l'adresse à "
            "laquelle le téléphone joint Forge (via WireGuard), par "
            "exemple FORGE_PUBLIC_URL=http://10.8.0.1:8000"
        )

    host = (urlparse(url).hostname or "").lower()
    if host in _LOOPBACK:
        raise PairingNotConfigured(
            f"FORGE_PUBLIC_URL vaut {url!r}, que le téléphone ne peut pas "
            "joindre : cette valeur part telle quelle dans le QR et devient "
            "l'URL de base du client. Mets l'adresse WireGuard de cette "
            "machine, pas une adresse de bouclage."
        )

    if not API_TOKEN:
        raise PairingNotConfigured(
            "API_TOKEN n'est pas configuré : il n'y a pas d'accès à "
            "déléguer. Appairer un téléphone sur une instance ouverte "
            "reviendrait à publier un accès complet à l'API sur le réseau "
            "WireGuard."
        )

    return url


def _purge(now: float) -> None:
    """Drop expired tokens. Called under _lock."""
    for token in [t for t, expiry in _pending.items() if expiry <= now]:
        del _pending[token]


def issue() -> str:
    """
    Mint a single-use pairing token valid for PAIRING_TTL_SECONDS.

    32 bytes from secrets, not a short human-typed code: this is
    scanned, never read aloud, so there is no usability argument for
    making it guessable and claim() is reachable without a token.
    """
    token = secrets.token_urlsafe(32)
    now = time.monotonic()
    with _lock:
        _purge(now)
        _pending[token] = now + PAIRING_TTL_SECONDS
    log.event("pairing.issued", ttl_seconds=PAIRING_TTL_SECONDS)
    return token


def claim(token: str) -> str | None:
    """
    Exchange a pairing token for the durable bearer token, once.

    Returns None for anything unknown, already used or expired -- the
    caller cannot tell those three apart on purpose, since telling a
    prober "expired" confirms the token existed.

    Compared with compare_digest rather than looked up: a dict hit is
    not constant time, and while 256 bits of entropy makes a timing
    oracle useless in practice, the repo already holds this line for
    API_TOKEN (api.require_token) and holding it in one place only is
    how it gets forgotten in the other.
    """
    if not token:
        return None

    now = time.monotonic()
    with _lock:
        _purge(now)
        matched = next(
            (t for t in _pending if hmac.compare_digest(t, token)),
            None,
        )
        if matched is None:
            log.event("pairing.claim_rejected")
            return None
        # Single use: consumed here, inside the lock, so two clients
        # racing on the same QR cannot both come away with a token.
        del _pending[matched]

    log.event("pairing.claimed")
    return API_TOKEN


def pending_count() -> int:
    """Live, unexpired tokens. For tests and logs, never for a caller."""
    with _lock:
        _purge(time.monotonic())
        return len(_pending)


def reset() -> None:
    """Forget every pending token. For tests, and for nothing else."""
    with _lock:
        _pending.clear()


def _qr_png_data_uri(payload: dict) -> str:
    """The payload as a PNG data URI, ready to inline in markdown."""
    import qrcode

    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=2,
    )
    # Compact separators: every byte saved is a smaller QR, and this
    # JSON is parsed by a machine that does not care about spaces.
    qr.add_data(json.dumps(payload, separators=(",", ":")))
    qr.make(fit=True)

    buf = io.BytesIO()
    qr.make_image(fill_color="black", back_color="white").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def qr_ascii(payload: dict) -> str:
    """
    The same payload drawn with text, for callers with no browser.

    Scannable from a terminal, which is the only rendering of a QR
    code a REPL can offer -- printing the token as text instead would
    put a credential in the scrollback for no gain.
    """
    import qrcode

    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, border=2)
    qr.add_data(json.dumps(payload, separators=(",", ":")))
    qr.make(fit=True)

    out = io.StringIO()
    qr.print_ascii(out=out)
    return out.getvalue()


def payload() -> dict:
    """
    What the phone needs to reach this Forge, with a fresh token.

    Raises PairingNotConfigured if it cannot be built.
    """
    url = _check_config()
    return {"url": url, "token": issue()}


def _minutes(seconds: int) -> str:
    minutes = max(1, round(seconds / 60))
    return f"{minutes} minute" + ("s" if minutes > 1 else "")


def build_reply() -> str:
    """The `!pair` answer, as markdown the web UI can render."""
    try:
        data = payload()
    except PairingNotConfigured as e:
        log.warning("pairing refused: %s", e)
        return f"Appairage impossible : {e}"

    return (
        "**Appairage**\n\n"
        f"Scanne ce QR code avec l'app Forge pour connecter ce téléphone à "
        f"`{data['url']}`.\n\n"
        f"![QR code d'appairage]({_qr_png_data_uri(data)})\n\n"
        f"Le code vaut pour **un seul appareil** et expire dans "
        f"**{_minutes(PAIRING_TTL_SECONDS)}**. Il n'est pas conservé dans "
        "l'historique : recharger la page le fait disparaître, `!pair` en "
        "affiche un nouveau."
    )


def intercept(user_input: str) -> str | None:
    """
    Handle *user_input* as `!pair`, or return None to let the rest of
    the turn happen.

    Runs at the same point as delegation.intercept and for the same
    reason: an intercepted turn makes no LLM call at all. Placed BEFORE
    it, because a job waiting on a field would otherwise swallow
    `!pair` as the answer to its question -- the user asking to pair a
    phone is not naming a folder.
    """
    if len(user_input) > _MAX_COMMAND_CHARS:
        return None
    if user_input.strip().lower() != _COMMAND:
        return None
    return build_reply()
