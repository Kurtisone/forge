"""
Tests for forge.pairing: the QR code that pairs the phone.

The property under test throughout is that the durable bearer token
never reaches the conversation. Everything else here -- single use,
expiry, the refusal to advertise a loopback address -- exists to keep
that true when the QR is photographed, left on screen, or scanned
twice.
"""

import base64
import io
import threading
import time

import pytest

from forge import pairing

_URL = "http://10.8.0.1:8000"
_BEARER = "durable-bearer-token-that-must-not-leak"


@pytest.fixture(autouse=True)
def _configured(monkeypatch):
    """A working pairing configuration, and an empty token store."""
    monkeypatch.setattr(pairing, "FORGE_PUBLIC_URL", _URL)
    monkeypatch.setattr(pairing, "API_TOKEN", _BEARER)
    pairing.reset()
    yield
    pairing.reset()


class TestWhatReachesTheConversation:
    def test_the_bearer_token_is_never_drawn_on_screen(self):
        """
        The reason this module exists. The reply is rendered into a
        thread that the web UI redisplays and that a recall can reach,
        so a bearer token inside it would be permanent and public to
        anyone who opens the UI.
        """
        reply = pairing.intercept("!pair")
        assert _BEARER not in reply

    def test_the_reply_carries_an_inline_png(self):
        reply = pairing.intercept("!pair")
        assert "![" in reply, "the QR has to be an image, not a link"
        assert "](data:image/png;base64," in reply

    def test_the_png_is_a_real_image(self):
        """
        Decoded rather than pattern-matched: a truncated or empty
        buffer still matches the data-URI prefix.
        """
        from PIL import Image

        reply = pairing.intercept("!pair")
        encoded = reply.split("base64,", 1)[1].split(")", 1)[0]
        img = Image.open(io.BytesIO(base64.b64decode(encoded)))

        assert img.format == "PNG"
        assert img.width == img.height, "a QR code is square"
        assert img.width > 64
        colours = {c for _, c in img.convert("L").getcolors()}
        assert len(colours) > 1, "an all-white image is not a QR code"

    def test_the_advertised_address_is_shown_so_it_can_be_checked(self):
        """The one part of the payload a human can verify by reading."""
        assert _URL in pairing.intercept("!pair")

    def test_the_payload_is_the_urls_and_a_fresh_token(self):
        first, second = pairing.payload(), pairing.payload()

        assert first["urls"] == [_URL]
        assert first["token"] and first["token"] != second["token"]
        assert first["token"] != _BEARER


class TestOneQrCodeForEveryWayIn:
    """
    The phone reaches Forge over WireGuard from outside and over the
    LAN when it is in the room, and the QR is drawn before anyone
    knows which. The server cannot pick for it: the request that draws
    the QR comes from the browser on the machine itself, never from
    the phone that will scan it. So all of them are encoded and the
    client tries them in order.
    """

    _WG = "http://10.8.0.1:8000"
    _LAN = "http://192.168.1.20:8000"

    @pytest.fixture(autouse=True)
    def _two_addresses(self, monkeypatch):
        monkeypatch.setattr(pairing, "FORGE_PUBLIC_URL", f"{self._WG},{self._LAN}")

    def test_every_address_is_encoded(self):
        assert pairing.payload()["urls"] == [self._WG, self._LAN]

    def test_the_order_written_is_the_order_kept(self, monkeypatch):
        """
        Order is the client's try order, so it is preserved as written
        rather than sorted into something tidier: the first entry is
        the one meant to work from anywhere.
        """
        monkeypatch.setattr(pairing, "FORGE_PUBLIC_URL", f"{self._LAN},{self._WG}")
        assert pairing.payload()["urls"] == [self._LAN, self._WG]

    def test_spacing_around_the_commas_is_forgiven(self, monkeypatch):
        monkeypatch.setattr(
            pairing, "FORGE_PUBLIC_URL", f"  {self._WG} ,  {self._LAN}  "
        )
        assert pairing.payload()["urls"] == [self._WG, self._LAN]

    def test_a_trailing_comma_does_not_add_an_empty_address(self, monkeypatch):
        """
        An empty entry reaches the client as an address to try, which
        it cannot, so it costs a timeout before the real one.
        """
        monkeypatch.setattr(pairing, "FORGE_PUBLIC_URL", f"{self._WG},")
        assert pairing.payload()["urls"] == [self._WG]

    def test_a_loopback_anywhere_in_the_list_is_refused(self, monkeypatch):
        """
        Checked on every entry, not just the first. An unreachable
        address later in the list is a client hanging on a timeout
        before it falls through to one that works.
        """
        monkeypatch.setattr(
            pairing, "FORGE_PUBLIC_URL", f"{self._WG},http://127.0.0.1:8000"
        )
        reply = pairing.intercept("!pair")

        assert "FORGE_PUBLIC_URL" in reply
        assert "data:image" not in reply

    def test_the_addresses_are_readable_in_the_reply(self):
        """
        The one part of the payload a human can verify by reading. A
        QR pointing somewhere unexpected is otherwise
        indistinguishable from one that does not.
        """
        reply = pairing.intercept("!pair")

        assert self._WG in reply and self._LAN in reply

    def test_a_single_address_still_works(self, monkeypatch):
        monkeypatch.setattr(pairing, "FORGE_PUBLIC_URL", self._WG)
        assert pairing.payload()["urls"] == [self._WG]


class TestTheTokenIsWorthNothingForLong:
    def test_one_claim_returns_the_bearer_token(self):
        assert pairing.claim(pairing.issue()) == _BEARER

    def test_a_second_claim_gets_nothing(self):
        """
        Single use is what makes a photographed QR harmless once the
        phone has scanned it.
        """
        token = pairing.issue()
        pairing.claim(token)
        assert pairing.claim(token) is None

    def test_an_expired_token_gets_nothing(self, monkeypatch):
        monkeypatch.setattr(pairing, "PAIRING_TTL_SECONDS", 0.05)
        token = pairing.issue()
        time.sleep(0.1)
        assert pairing.claim(token) is None

    def test_expired_tokens_stop_being_held(self, monkeypatch):
        monkeypatch.setattr(pairing, "PAIRING_TTL_SECONDS", 0.05)
        pairing.issue()
        time.sleep(0.1)
        assert pairing.pending_count() == 0

    def test_an_unknown_token_gets_nothing(self):
        assert pairing.claim("not-a-token") is None
        assert pairing.claim("") is None

    def test_two_clients_racing_on_one_qr_produce_one_winner(self):
        """
        The consume happens inside the lock. Without that, the losing
        phone also walks away with a bearer token -- which is the
        single-use guarantee quietly not holding.
        """
        token = pairing.issue()
        results, barrier = [], threading.Barrier(8)

        def claim():
            barrier.wait()
            results.append(pairing.claim(token))

        threads = [threading.Thread(target=claim) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results.count(_BEARER) == 1
        assert results.count(None) == 7


class TestItRefusesRatherThanHandOutAQrThatCannotWork:
    def test_no_public_url(self, monkeypatch):
        monkeypatch.setattr(pairing, "FORGE_PUBLIC_URL", "")
        reply = pairing.intercept("!pair")

        assert "FORGE_PUBLIC_URL" in reply
        assert "data:image" not in reply

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://[::1]:8000",
            "http://0.0.0.0:8000",
        ],
    )
    def test_a_loopback_address_is_not_advertised(self, monkeypatch, url):
        """
        This value becomes the Android client's base URL verbatim, so
        a loopback here builds a client that calls the phone itself --
        and fails with a connection error naming neither Forge nor
        this setting.
        """
        monkeypatch.setattr(pairing, "FORGE_PUBLIC_URL", url)
        reply = pairing.intercept("!pair")

        assert "FORGE_PUBLIC_URL" in reply
        assert "data:image" not in reply

    def test_an_open_instance_cannot_be_paired(self, monkeypatch):
        """
        With no API_TOKEN there is no access to delegate, and pairing
        would publish an open instance onto the WireGuard network.
        """
        monkeypatch.setattr(pairing, "API_TOKEN", "")
        reply = pairing.intercept("!pair")

        assert "API_TOKEN" in reply
        assert "data:image" not in reply

    def test_a_refusal_mints_no_token(self, monkeypatch):
        monkeypatch.setattr(pairing, "FORGE_PUBLIC_URL", "")
        pairing.intercept("!pair")
        assert pairing.pending_count() == 0


class TestTheCommonCaseIsNotPair:
    @pytest.mark.parametrize(
        "message",
        [
            "bonjour",
            "!pairing",
            "!pair le téléphone",
            "comment fonctionne !pair ?",
            "",
        ],
    )
    def test_anything_else_goes_to_the_router(self, message):
        assert pairing.intercept(message) is None

    def test_a_long_paste_is_rejected_on_its_length_alone(self):
        """
        None is the common case and has to stay cheap: a pasted file
        is turned down before any string work, the same gate
        delegation.py puts in front of its keywords.
        """
        assert pairing.intercept("x" * 100_000) is None

    def test_spelling_and_spacing_are_forgiven(self):
        for spelling in ("!pair", "  !pair  ", "!PAIR", "!Pair\n"):
            assert pairing.intercept(spelling) is not None

    def test_no_token_is_minted_by_a_message_that_is_not_pair(self):
        pairing.intercept("écris-moi une fonction de tri")
        assert pairing.pending_count() == 0


def test_the_ascii_rendering_is_scannable_from_a_terminal():
    """
    The REPL has no way to show an image. Printing the token as text
    instead would put a credential in the scrollback, so the QR is
    drawn with characters.
    """
    art = pairing.qr_ascii({"url": _URL, "token": "x" * 43})
    lines = [line for line in art.splitlines() if line.strip()]

    assert len(lines) > 10
    assert _BEARER not in art
