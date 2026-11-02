"""Regression tests: bounded empty-length retry never alters legacy behavior."""
from types import SimpleNamespace
import sys
import pytest

from ragtree.experiments.kbs import gateway


def _response(text, finish, output_tokens=4096, cost=0.03):
    obj = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
        _hidden_params={'response_cost': cost},
    )
    obj.model_dump = lambda: {
        'choices': [{'message': {'content': text}, 'finish_reason': finish}],
        'usage': {'prompt_tokens': 900, 'completion_tokens': output_tokens,
                  'completion_tokens_details': {'reasoning_tokens': output_tokens if text is None else 2}},
    }
    return obj


@pytest.fixture
def mock_gateway(monkeypatch):
    calls, traces, events = [], [], []
    pending = []
    def completion(**kwargs):
        calls.append(kwargs)
        item = pending.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setitem(sys.modules, 'litellm', SimpleNamespace(completion=completion))
    monkeypatch.setattr(gateway, 'trace_call', lambda request, response, elapsed, error=None:
                        traces.append((request, response, error)))
    monkeypatch.setattr(gateway, 'trace_event', lambda name, data: events.append((name, data)))
    monkeypatch.setenv('KBS_PROTOCOL_MODE','controlled_revision_v2')
    monkeypatch.setenv('KBS_EMPTY_LENGTH_FALLBACK','1')
    monkeypatch.setenv('KBS_FALLBACK_MAX_TOKENS','8192')
    monkeypatch.setenv('KBS_REASONING_EFFORT','low')
    return gateway.LiteLLMClient('openai/openai/gpt-oss-20b'), pending, calls, traces, events


def _call(client):
    return client.chat([{'role': 'user', 'content': 'Return JSON'}], temperature=0.0, max_tokens=4096)


def test_fallback_one_additional_attempt_and_separate_billed_traces(mock_gateway):
    client, pending, calls, traces, events = mock_gateway
    pending.extend([_response(None, 'length'), _response('{"CAUSE": []}', 'stop', 99)])
    assert _call(client) == '{"CAUSE": []}'
    assert [c['max_tokens'] for c in calls] == [4096, 8192]
    assert all(c['reasoning_effort'] == 'low' for c in calls)
    assert len(traces) == 2
    assert traces[0][2] and traces[1][2] is None
    assert traces[0][1]['usage']['completion_tokens'] == 4096
    assert traces[1][1]['usage']['completion_tokens'] == 99
    assert traces[0][1]['kbs_litellm_reported_cost_usd'] == 0.03
    assert len(events) == 1 and events[0][0] == 'generation_fallback'


def test_success_does_not_retry_even_if_finish_length(mock_gateway):
    client, pending, calls, traces, events = mock_gateway
    pending.append(_response('{"CAUSE": []}', 'length'))
    assert _call(client) == '{"CAUSE": []}'
    assert len(calls) == len(traces) == 1
    assert not events


def test_stop_with_empty_answer_fails_without_fallback(mock_gateway):
    client, pending, calls, traces, events = mock_gateway
    pending.append(_response(None, 'stop'))
    with pytest.raises(ValueError, match='no usable final-answer'):_call(client)
    assert len(calls) == len(traces) == 1
    assert not events


def test_two_length_empty_responses_terminate(mock_gateway):
    client, pending, calls, traces, events = mock_gateway
    pending.extend([_response(None, 'length'), _response(None, 'length', 8192)])
    with pytest.raises(ValueError, match='configured_max_tokens=8192'):_call(client)
    assert [c['max_tokens'] for c in calls] == [4096,8192]
    assert len(traces) == 2 and all(t[2] for t in traces)
    assert len(events) == 1


def test_transport_error_no_fallback(mock_gateway):
    client, pending, calls, traces, events = mock_gateway
    pending.append(ConnectionError('simulated provider failure'))
    with pytest.raises(ConnectionError):_call(client)
    assert len(calls) == len(traces) == 1
    assert traces[0][1] is None
    assert not events


def test_second_attempt_transport_error_stays_bounded(mock_gateway):
    client, pending, calls, traces, events = mock_gateway
    pending.extend([_response(None,'length'),TimeoutError('simulated timeout')])
    with pytest.raises(TimeoutError):_call(client)
    assert len(calls) == len(traces) == 2
    assert traces[0][1] is not None and traces[1][1] is None


def test_legacy_protocol_never_uses_fallback(mock_gateway,monkeypatch):
    client, pending, calls, traces, events = mock_gateway
    monkeypatch.setenv('KBS_PROTOCOL_MODE','legacy_xquality')
    pending.append(_response(None,'length'))
    with pytest.raises(ValueError):_call(client)
    assert len(calls) == 1


def test_invalid_fallback_cap_fails_before_api(mock_gateway,monkeypatch):
    client, pending, calls, traces, events = mock_gateway
    monkeypatch.setenv('KBS_FALLBACK_MAX_TOKENS','4096')
    with pytest.raises(ValueError, match='fallback max_tokens'): _call(client)
    assert not calls
