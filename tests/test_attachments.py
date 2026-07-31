"""Caller attachments: A2A parts in, agent-readable prompt out.

The executor turns the non-text parts of a message into Attachments, and each
backend decides how to deliver them. Both halves are pure, so neither needs a
live agent.
"""

from __future__ import annotations

from a2a.types import Message, Part, Role

from a2acode.backends.attach import append_to_prompt, render
from a2acode.backends.base import Attachment, RunRequest
from a2acode.executor import _build_input


class _Context:
    """The slice of RequestContext that _build_input reads."""

    def __init__(self, message) -> None:
        self.message = message

    def get_user_input(self) -> str:  # pragma: no cover - only for empty messages
        return ""


def _context(*parts) -> _Context:
    return _Context(Message(message_id="m1", role=Role.ROLE_USER, parts=list(parts)))


def test_text_parts_become_the_prompt_and_leave_no_attachments():
    prompt, attachments = _build_input(_context(Part(text="fix"), Part(text="the bug")))

    assert prompt == "fix\nthe bug"
    assert attachments == []


def test_a_text_file_part_is_decoded_and_attached():
    prompt, attachments = _build_input(
        _context(
            Part(text="review this"),
            Part(raw=b"x = 1\n", media_type="text/x-python", filename="a.py"),
        )
    )

    assert prompt == "review this"
    assert len(attachments) == 1
    assert attachments[0].name == "a.py"
    assert attachments[0].text == "x = 1\n"
    assert attachments[0].data is None


def test_binary_part_is_kept_as_bytes():
    png = b"\x89PNG\r\n\x1a\n"
    _, attachments = _build_input(
        _context(Part(raw=png, media_type="image/png", filename="shot.png"))
    )

    assert attachments[0].data == png
    assert attachments[0].text is None


def test_part_declared_text_but_not_decodable_falls_back_to_bytes():
    _, attachments = _build_input(
        _context(Part(raw=b"\xff\xfe\x00", media_type="text/plain", filename="odd.txt"))
    )

    assert attachments[0].text is None
    assert attachments[0].data == b"\xff\xfe\x00"


def test_url_part_becomes_a_link():
    _, attachments = _build_input(
        _context(Part(url="https://example.test/log.txt", media_type="text/plain"))
    )

    assert attachments[0].uri == "https://example.test/log.txt"
    assert attachments[0].text is None


def test_data_part_is_rendered_as_json():
    part = Part()
    part.data.struct_value["failing"] = "test_auth"
    _, attachments = _build_input(_context(part))

    assert "test_auth" in (attachments[0].text or "")


def test_an_oversized_text_attachment_is_truncated_and_says_so(monkeypatch):
    monkeypatch.setattr("a2acode.executor._MAX_ATTACHED", 10)
    _, attachments = _build_input(
        _context(Part(raw=b"y" * 50, media_type="text/plain", filename="big.txt"))
    )

    assert attachments[0].text == "y" * 10
    assert attachments[0].truncated


def test_the_total_budget_bounds_a_message_of_many_attachments(monkeypatch):
    monkeypatch.setattr("a2acode.executor._MAX_ATTACHED_TOTAL", 12)
    _, attachments = _build_input(
        _context(
            Part(raw=b"a" * 10, media_type="text/plain", filename="1.txt"),
            Part(raw=b"b" * 10, media_type="text/plain", filename="2.txt"),
            Part(raw=b"c" * 10, media_type="text/plain", filename="3.txt"),
        )
    )

    assert [len(a.text or "") for a in attachments] == [10, 2, 0]
    assert [a.truncated for a in attachments] == [False, True, True]


def test_an_oversized_binary_attachment_is_dropped_but_still_announced(monkeypatch):
    monkeypatch.setattr("a2acode.executor._MAX_ATTACHED", 4)
    _, attachments = _build_input(
        _context(Part(raw=b"\x89PNG\r\n\x1a\n", media_type="image/png"))
    )

    assert attachments[0].data is None
    assert attachments[0].truncated


def test_render_inlines_text_in_a_fence():
    out = render(Attachment(name="a.py", media_type="text/x-python", text="x = 1\n"))

    assert out == "[attached a.py (text/x-python)]\n```\nx = 1\n\n```"


def test_render_escapes_content_that_contains_a_fence():
    out = render(Attachment(name="d.md", text="```\nnot the end\n```"))

    # A longer fence, so the attachment cannot close the block early and spill
    # its tail into the prompt as instructions.
    assert out.count("````") == 2
    assert out.endswith("````")


def test_render_marks_a_truncated_attachment():
    out = render(Attachment(name="big.log", text="head", truncated=True))
    assert ", truncated]" in out


def test_render_describes_binary_without_inlining_it():
    out = render(Attachment(name="s.png", media_type="image/png", data=b"1234"))
    assert out == "[attached s.png (image/png), 4 bytes: binary, not inlined]"


def test_render_links_a_uri():
    out = render(Attachment(name="log", uri="https://example.test/l"))
    assert out == "[attached log (file): https://example.test/l]"


def test_append_to_prompt_leaves_a_bare_prompt_alone():
    assert append_to_prompt("fix it", []) == "fix it"


def test_append_to_prompt_puts_attachments_after_the_prompt():
    out = append_to_prompt("fix it", [Attachment(name="l", uri="u")])
    assert out == "fix it\n\n[attached l (file): u]"


def test_acp_sends_an_image_block_when_the_agent_reads_images():
    from acp import schema as s

    from a2acode.backends.acp import prompt_blocks

    request = RunRequest(
        prompt="what is this",
        attachments=[Attachment(name="s.png", media_type="image/png", data=b"1234")],
    )
    caps = s.AgentCapabilities(prompt_capabilities=s.PromptCapabilities(image=True))
    blocks = prompt_blocks(request, caps)

    assert len(blocks) == 2
    assert blocks[0].text == "what is this"
    assert blocks[1].type == "image"
    assert blocks[1].mime_type == "image/png"


def test_acp_falls_back_to_a_label_when_the_agent_cannot_read_images():
    from acp import schema as s

    from a2acode.backends.acp import prompt_blocks

    request = RunRequest(
        prompt="what is this",
        attachments=[Attachment(name="s.png", media_type="image/png", data=b"1234")],
    )
    caps = s.AgentCapabilities(prompt_capabilities=s.PromptCapabilities(image=False))
    blocks = prompt_blocks(request, caps)

    assert len(blocks) == 1
    assert "s.png" in blocks[0].text
    assert "binary, not inlined" in blocks[0].text


async def test_echo_renders_attachments_so_the_offline_path_covers_them():
    from a2acode.backends import BackendSession, TextDelta, make_backend

    session = BackendSession()
    request = RunRequest(
        prompt="review",
        attachments=[Attachment(name="a.py", media_type="text/x-python", text="x = 1")],
    )
    session.start(lambda s: make_backend("echo").drive(s, request))
    events = [e async for e in session.drain()]
    await session.close()

    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert "a.py" in text
    assert "x = 1" in text


def test_acp_inlines_text_attachments_regardless_of_capabilities():
    from a2acode.backends.acp import prompt_blocks

    request = RunRequest(
        prompt="review",
        attachments=[Attachment(name="a.py", text="x = 1\n")],
    )
    blocks = prompt_blocks(request, None)

    assert len(blocks) == 1
    assert "x = 1" in blocks[0].text
