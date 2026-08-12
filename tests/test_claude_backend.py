"""Unit tests for the claude backend's mapping and plan tracking, no live Claude."""

from __future__ import annotations

from claude_agent_sdk import (
    AssistantMessage,
    PermissionResultAllow,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from a2acode.backends import claude as claude_mod
from a2acode.backends.base import (
    FileChange,
    PermissionDecision,
    PermissionRequest,
    Plan,
    Result,
    RunRequest,
    TextDelta,
    Thought,
    ToolResult,
    ToolUse,
)
from a2acode.backends.claude import (
    ClaudeBackend,
    PlanTracker,
    allowed_input,
    events_from_message,
)
from a2acode.backends.session import BackendSession


def test_events_from_assistant_message_with_write():
    message = AssistantMessage(
        content=[
            TextBlock(text="creating the file"),
            ToolUseBlock(
                id="t1",
                name="Write",
                input={"file_path": "a.py", "content": "x = 1\n"},
            ),
        ],
        model="claude-test",
    )
    events = list(events_from_message(message))

    assert len(events) == 3
    assert isinstance(events[0], TextDelta)
    assert events[0].text == "creating the file"
    assert isinstance(events[1], ToolUse)
    assert events[1].name == "Write"
    assert isinstance(events[2], FileChange)
    assert events[2].path == "a.py"
    assert "+x = 1" in events[2].diff


def test_events_from_result_message():
    message = ResultMessage(
        subtype="success",
        duration_ms=10,
        duration_api_ms=8,
        is_error=False,
        num_turns=2,
        session_id="s1",
        total_cost_usd=0.0123,
        usage={"input_tokens": 5},
    )
    events = list(events_from_message(message))

    assert len(events) == 1
    result = events[0]
    assert isinstance(result, Result)
    assert result.session_id == "s1"
    assert result.cost_usd == 0.0123
    assert result.num_turns == 2


def test_tool_result_block_maps_to_tool_result():
    message = UserMessage(
        content=[ToolResultBlock(tool_use_id="t1", content="a.py\nb.py\n")]
    )
    events = list(events_from_message(message))

    assert len(events) == 1
    result = events[0]
    assert isinstance(result, ToolResult)
    assert result.tool_use_id == "t1"
    assert not result.failed
    assert result.output == "a.py\nb.py\n"


def test_errored_tool_result_is_flagged_and_block_content_is_joined():
    message = UserMessage(
        content=[
            ToolResultBlock(
                tool_use_id="t1",
                content=[{"type": "text", "text": "boom"}, {"type": "image"}],
                is_error=True,
            )
        ]
    )
    result = next(iter(events_from_message(message)))

    assert result.failed
    # The image block carries no text, so it is skipped rather than stringified.
    assert result.output == "boom"


def _steps(plan: Plan) -> list[tuple[str, str]]:
    return [(step.content, step.status) for step in plan.steps]


def _call(tracker: PlanTracker, name, tool_input, tool_use_id, output="", failed=False):
    """One plan tool call and the result that decides whether it stuck."""
    assert tracker.absorb(ToolUse(name, tool_input, tool_use_id)) is None
    return tracker.absorb(
        ToolResult(tool_use_id=tool_use_id, failed=failed, output=output)
    )


def _create(tracker: PlanTracker, subject, tool_use_id, task_id, **kwargs):
    return _call(
        tracker,
        "TaskCreate",
        {"subject": subject},
        tool_use_id,
        output=f"Task #{task_id} created successfully: {subject}",
        **kwargs,
    )


def test_the_task_tools_build_a_plan_one_entry_at_a_time():
    tracker = PlanTracker()

    first = _create(tracker, "read the code", "t1", "1")
    assert _steps(first) == [("read the code", "pending")]

    second = _create(tracker, "write the fix", "t2", "2")
    assert len(second.steps) == 2

    done = _call(tracker, "TaskUpdate", {"taskId": "1", "status": "completed"}, "t3")
    assert _steps(done) == [
        ("read the code", "completed"),
        ("write the fix", "pending"),
    ]


def test_a_task_whose_call_failed_never_enters_the_plan():
    # Most likely because the A2A caller denied it, and a plan claiming work the
    # agent never took on is worse than no plan.
    tracker = PlanTracker()
    assert _create(tracker, "do the risky thing", "t1", "1", failed=True) is None

    plan = _create(tracker, "do the safe thing", "t2", "2")
    assert _steps(plan) == [("do the safe thing", "pending")]


def test_an_id_from_another_task_in_the_result_is_not_bound():
    tracker = PlanTracker()
    assert (
        _call(
            tracker,
            "TaskCreate",
            {"subject": "do it"},
            "t1",
            output="Blocked by task #7. Task #8 created successfully: do it",
        )
        is not None
    )
    # Binding to #7 would make an update for that unrelated task mutate this
    # step, and leave every update for the real id going nowhere.
    assert (
        _call(tracker, "TaskUpdate", {"taskId": "7", "status": "completed"}, "t2")
        is None
    )
    done = _call(tracker, "TaskUpdate", {"taskId": "8", "status": "completed"}, "t3")
    assert _steps(done) == [("do it", "completed")]


def test_a_whole_list_write_over_the_cap_keeps_its_head(monkeypatch):
    monkeypatch.setattr(claude_mod, "_MAX_STEPS", 2)
    plan = _call(
        PlanTracker(),
        "TodoWrite",
        {"todos": [{"content": f"step {i}"} for i in range(4)]},
        "t1",
    )
    # One call carries the whole list, so the entries to lose are the last.
    assert _steps(plan) == [("step 0", "pending"), ("step 1", "pending")]


def test_a_deleted_task_leaves_the_plan():
    tracker = PlanTracker()
    _create(tracker, "drop me", "t1", "1")

    plan = _call(tracker, "TaskUpdate", {"taskId": "1", "status": "deleted"}, "t2")
    assert plan is not None
    assert plan.steps == []
    # The id stops naming anything, so a later task reusing it is not confused
    # for the one that was deleted.
    assert tracker._keys == {}


def test_an_update_to_a_task_the_tracker_never_saw_is_ignored():
    tracker = PlanTracker()
    assert _call(tracker, "TaskUpdate", {"taskId": "9", "status": "done"}, "t1") is None


def test_todowrite_still_carries_a_whole_plan():
    # Older CLIs write the list in one call, so one call is a whole plan.
    plan = _call(
        PlanTracker(),
        "TodoWrite",
        {
            "todos": [
                {"content": "read the code", "status": "completed"},
                {"content": "write the fix", "status": "in_progress"},
            ]
        },
        "t1",
    )
    assert _steps(plan) == [
        ("read the code", "completed"),
        ("write the fix", "in_progress"),
    ]


def test_a_malformed_task_list_does_not_raise():
    tracker = PlanTracker()
    for todos in ("oops", [1, "x"], []):
        assert _call(tracker, "TodoWrite", {"todos": todos}, "t1") is None
    assert _call(tracker, "TaskCreate", {"description": "no subject"}, "t2") is None


def test_a_result_for_a_tool_that_is_not_a_plan_call_is_ignored():
    tracker = PlanTracker()
    assert (
        tracker.absorb(ToolResult(tool_use_id="t1", output="Task #1 whatever")) is None
    )


def test_a_plan_tool_call_is_only_a_tool_use_to_the_mapper():
    # The plan needs state across calls, so it is the tracker's, not the pure
    # mapper's, which is why keying off one tool name went stale unnoticed.
    message = AssistantMessage(
        content=[ToolUseBlock(id="t1", name="TaskCreate", input={"subject": "x"})],
        model="claude-test",
    )
    assert [type(e) for e in events_from_message(message)] == [ToolUse]


def test_thinking_block_maps_to_a_thought():
    message = AssistantMessage(
        content=[
            ThinkingBlock(thinking="weighing two options", signature="sig"),
            TextBlock(text="going with the second"),
        ],
        model="claude-test",
    )
    events = list(events_from_message(message))

    assert [type(e) for e in events] == [Thought, TextDelta]
    assert events[0].text == "weighing two options"


def test_plain_text_user_message_yields_nothing():
    assert list(events_from_message(UserMessage(content="just text"))) == []


def test_tool_result_output_is_capped():
    message = UserMessage(
        content=[ToolResultBlock(tool_use_id="t1", content="x" * 5000)]
    )
    output = next(iter(events_from_message(message))).output
    assert output.endswith(" …")
    assert len(output) == 2002


def test_empty_text_block_is_skipped():
    message = AssistantMessage(content=[TextBlock(text="")], model="claude-test")
    assert list(events_from_message(message)) == []


def test_a_resumed_turn_keeps_working_the_same_task_list():
    backend = ClaudeBackend()
    first = RunRequest(prompt="start", context_id="ctx-1")
    tracker = backend._plan_for(first)
    _create(tracker, "read the code", "t1", "1")

    # A resumed turn addresses a task by an id only the turn that created it saw.
    resumed = backend._plan_for(
        RunRequest(prompt="carry on", context_id="ctx-1", resume="sess-1")
    )
    plan = _call(resumed, "TaskUpdate", {"taskId": "1", "status": "completed"}, "t9")
    assert _steps(plan) == [("read the code", "completed")]


def test_a_turn_without_a_resume_starts_the_list_over():
    backend = ClaudeBackend()
    tracker = backend._plan_for(RunRequest(prompt="start", context_id="ctx-1"))
    _create(tracker, "read the code", "t1", "1")

    # No resume means a fresh Claude conversation, so an earlier list is not what
    # the agent is working from.
    fresh = backend._plan_for(RunRequest(prompt="again", context_id="ctx-1"))
    assert (
        _call(fresh, "TaskUpdate", {"taskId": "1", "status": "completed"}, "t9") is None
    )


def test_one_lists_steps_are_bounded(monkeypatch):
    # A context can be resumed for as long as the server runs.
    monkeypatch.setattr(claude_mod, "_MAX_STEPS", 2)
    tracker = PlanTracker()
    for i in range(3):
        _create(tracker, f"step {i}", f"t{i}", str(i))

    assert _steps(tracker._plan()) == [("step 1", "pending"), ("step 2", "pending")]
    # The id map is bounded with them, or it outlives every task it named.
    assert len(tracker._keys) == 2


def test_calls_still_waiting_on_a_result_are_bounded(monkeypatch):
    monkeypatch.setattr(claude_mod, "_MAX_PENDING", 1)
    tracker = PlanTracker()
    for i in range(3):
        tracker.absorb(ToolUse("TaskCreate", {"subject": f"step {i}"}, f"t{i}"))

    assert len(tracker._calls) == 1


def test_kept_task_lists_are_bounded(monkeypatch):
    monkeypatch.setattr(claude_mod, "_MAX_PLANS", 2)
    backend = ClaudeBackend()
    for context in ("a", "b", "c"):
        backend._plan_for(RunRequest(prompt="hi", context_id=context))

    assert set(backend._plans) == {"b", "c"}


def test_a_reused_context_moves_off_the_eviction_front(monkeypatch):
    monkeypatch.setattr(claude_mod, "_MAX_PLANS", 2)
    backend = ClaudeBackend()
    backend._plan_for(RunRequest(prompt="hi", context_id="a"))
    backend._plan_for(RunRequest(prompt="hi", context_id="b"))
    # Touching "a" again without a resume still has to refresh its position, or
    # the context in active use is the one evicted.
    backend._plan_for(RunRequest(prompt="hi", context_id="a"))
    backend._plan_for(RunRequest(prompt="hi", context_id="c"))

    assert set(backend._plans) == {"a", "c"}


def test_options_applies_settings():
    backend = ClaudeBackend(
        cwd="/tmp/project",
        permission_mode="acceptEdits",
        max_budget_usd=0.5,
        model="claude-test",
    )
    options = backend._options(
        RunRequest(prompt="hi", resume="sess-1"), can_use_tool=lambda *a: None
    )

    assert options.cwd == "/tmp/project"
    assert options.resume == "sess-1"
    assert options.permission_mode == "acceptEdits"
    assert options.max_budget_usd == 0.5
    # A server must not inherit the developer's personal allowlist.
    assert options.setting_sources == []


_QUESTIONS = {
    "questions": [
        {
            "question": "Which test runner?",
            "header": "Runner",
            "options": [{"label": "pytest"}, {"label": "unittest"}],
            "multiSelect": False,
        }
    ]
}


def test_allowed_input_echoes_an_ordinary_tools_input():
    decision = PermissionDecision(request_id="r1", allow=True)
    given = {"command": "pytest -x"}

    # An allow that omits the input is a malformed result to an older CLI, which
    # denies the call instead of running it.
    assert allowed_input("Bash", given, decision) == given
    # Answers belong to the tool that asked; nothing else runs with an extra
    # argument the agent never wrote.
    answered = PermissionDecision("r1", True, answers={"Which test runner?": "pytest"})
    assert allowed_input("Bash", given, answered) == given


def test_allowed_input_folds_the_answers_into_the_question():
    decision = PermissionDecision(
        request_id="r1", allow=True, answers={"Which test runner?": "pytest"}
    )

    updated = allowed_input("AskUserQuestion", _QUESTIONS, decision)

    assert updated["answers"] == {"Which test runner?": "pytest"}
    # The tool pairs each answer with the question it answers.
    assert updated["questions"] == _QUESTIONS["questions"]


def test_allowed_input_leaves_an_unanswered_question_alone():
    decision = PermissionDecision(request_id="r1", allow=True)
    assert "answers" not in allowed_input("AskUserQuestion", _QUESTIONS, decision)


class _FakeClient:
    """Stands in for ClaudeSDKClient, driving the permission callback once."""

    results: list = []

    def __init__(self, options):
        self._options = options

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, prompt):
        return None

    async def receive_response(self):
        _FakeClient.results.append(
            await self._options.can_use_tool("AskUserQuestion", _QUESTIONS, None)
        )
        return
        yield


async def test_an_answered_question_reaches_the_sdk(monkeypatch):
    monkeypatch.setattr(claude_mod, "ClaudeSDKClient", _FakeClient)
    _FakeClient.results.clear()
    session = BackendSession()
    backend = ClaudeBackend()
    session.start(lambda s: backend.drive(s, RunRequest(prompt="hi")))

    asked = [event async for event in session.drain()][-1]
    assert isinstance(asked, PermissionRequest)
    assert asked.tool_input == _QUESTIONS
    session.resolve(
        PermissionDecision(
            request_id=asked.request_id,
            allow=True,
            answers={"Which test runner?": ["pytest", "unittest"]},
        )
    )
    async for _ in session.drain():
        pass

    allowed = _FakeClient.results[-1]
    assert isinstance(allowed, PermissionResultAllow)
    assert allowed.updated_input["answers"] == {
        "Which test runner?": ["pytest", "unittest"]
    }


async def test_a_denied_question_carries_the_callers_words(monkeypatch):
    monkeypatch.setattr(claude_mod, "ClaudeSDKClient", _FakeClient)
    _FakeClient.results.clear()
    session = BackendSession()
    backend = ClaudeBackend()
    session.start(lambda s: backend.drive(s, RunRequest(prompt="hi")))

    asked = [event async for event in session.drain()][-1]
    session.resolve(
        PermissionDecision(
            request_id=asked.request_id, allow=False, message="stop asking"
        )
    )
    async for _ in session.drain():
        pass

    assert _FakeClient.results[-1].message == "stop asking"
