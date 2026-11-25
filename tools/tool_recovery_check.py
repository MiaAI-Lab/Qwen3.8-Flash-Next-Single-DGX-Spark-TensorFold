#!/usr/bin/env python3
"""Separate TensorFold XML transport and simulated client-side recovery tests.
No real tool is ever executed. Python standard library only.
"""
import argparse
import json
import os
import sys
import urllib.request

UNKNOWN = 'tf_recovery_probe_nonexistent'
OFFERED = 'lookup'
RAW = ('<tool_call>\n<function=' + UNKNOWN + '>\n'
       '<parameter=query>\nprobe\n</parameter>\n</function>\n</tool_call>')
TOOLS = [{'type': 'function', 'function': {'name': OFFERED,
    'description': 'Harmless simulated lookup. Returns a test value for the supplied query.',
    'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}},
                   'required': ['query']}}}]
# Preserve the entire original v1 request, including the original tool schema.
TRANSPORT_TOOLS = [{'type':'function','function':{'name':'lookup','description':'Harmless offered test tool','parameters':{'type':'object','properties':{'query':{'type':'string'}}}}}]
TRANSPORT_MESSAGES = [{'role': 'user', 'content':
    'This is a harmless protocol formatting test. Do not invoke any real tool. Produce precisely the following explicit XML tool-call envelope, without code fences or explanation:\n' + RAW}]


def request(a, messages, tools):
    payload = {'model': a.model, 'messages': messages, 'tools': tools,
               'stream': a.stream, 'temperature': 0, 'seed': a.seed, 'max_tokens': 2048}
    headers = {'Content-Type': 'application/json'}
    if os.getenv('API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['API_KEY']
    with urllib.request.urlopen(urllib.request.Request(a.url, json.dumps(payload).encode(), headers), timeout=300) as r:
        if not a.stream:
            c = json.load(r)['choices'][0]
            return c['message'], c.get('finish_reason')
        calls = {}; content = ''; reasoning = ''; finish = None
        for line in r:
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                break
            for c in json.loads(data).get('choices', []):
                d = c.get('delta', {}); finish = c.get('finish_reason') or finish
                content += d.get('content') or ''; reasoning += d.get('reasoning_content') or ''
                for t in d.get('tool_calls') or []:
                    x = calls.setdefault(t['index'], {'type':'function','id':'','function':{'name':'','arguments':''}})
                    x['id'] += t.get('id') or ''
                    for k in ('name','arguments'):
                        x['function'][k] += (t.get('function') or {}).get(k) or ''
        return {'role':'assistant','content':content,'reasoning_content':reasoning,
                'tool_calls':[calls[i] for i in sorted(calls)]}, finish


def report(label, msg, finish):
    print(json.dumps({'phase': label, 'finish_reason': finish, 'message': msg}, indent=2, ensure_ascii=False), flush=True)


def exact_call(msg, finish, name):
    calls = msg.get('tool_calls') or []
    if finish != 'tool_calls' or len(calls) != 1:
        return None
    call = calls[0]
    try:
        if (call['type'] == 'function' and call['id'] and call['function']['name'] == name
                and json.loads(call['function']['arguments']) == {'query':'probe'}):
            return call
    except (KeyError, TypeError, ValueError):
        pass
    return None


def transport(a):
    msg, finish = request(a, TRANSPORT_MESSAGES, TRANSPORT_TOOLS)
    report('original-v1-transport', msg, finish)
    call = exact_call(msg, finish, UNKNOWN)
    if call:
        result = 'PASS' if a.expect == 'forward' else 'FAIL'
        print(result + ': unknown call transported as structured function call')
        return result, call
    if msg.get('tool_calls'):
        print('FAIL: unexpected structured call or invalid arguments/finish')
        return 'FAIL', None
    if RAW in (msg.get('content') or '') and finish == 'stop':
        result = 'PASS' if a.expect == 'keep' else 'FAIL'
        print(result + ': original envelope retained as text')
        return result, None
    print('INCONCLUSIVE: complete intended envelope not observed; inspect raw parser diagnostics')
    return 'INCONCLUSIVE', None


def recovery(a, observed_call=None):
    # Deliberately independent history: a natural task, not an instruction to
    # reproduce XML. The previous failed call is inserted by the test client.
    # Reuse the observed call when available; otherwise construct a fixture.
    # This tests model correction and acceptance of tool history, not a live
    # uninterrupted agent loop or OWUI itself.
    failed = observed_call or {'id':'call_tf_recovery_fixture','type':'function',
        'function':{'name':UNKNOWN,'arguments':'{"query":"probe"}'}}
    messages = [
        {'role':'user','content':'Use the available lookup tool to find the test value for query probe. '
         'If a tool fails or is unavailable, recover using an offered tool rather than repeating the failed call. '
         'When the lookup succeeds, answer with the returned test value only.'},
        {'role':'assistant','content':None,'tool_calls':[failed]},
        {'role':'tool','tool_call_id':failed['id'],
         'content':f'Error: Tool "{UNKNOWN}" not found. No tool was executed.'},
    ]
    print('Recovery fixture: injected prior assistant call + simulated tool error (independent history).', flush=True)
    msg, finish = request(a, messages, TOOLS)
    report('recovery-after-error', msg, finish)
    call = exact_call(msg, finish, OFFERED)
    if not call:
        print('FAIL: model did not call the offered lookup tool with query=probe')
        return 'FAIL'
    # The returned value is not present in the initial prompt.
    messages += [{'role':'assistant','content':msg.get('content'), 'tool_calls':[call]},
                 {'role':'tool','tool_call_id':call['id'],'content':'RECOVERY_OK'}]
    final, finish = request(a, messages, TOOLS)
    report('recovery-final', final, finish)
    if finish != 'stop' or final.get('tool_calls') or (final.get('content') or '').strip() != 'RECOVERY_OK':
        print('FAIL: model did not finish with the simulated lookup result')
        return 'FAIL'
    print('PASS: model recovered from synthetic unknown-tool error and completed the lookup')
    return 'PASS'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8000/v1/chat/completions')
    p.add_argument('--model', default='nvidia/Qwen3.8-Flash-Next-FP4-Local')
    p.add_argument('--stream', action='store_true')
    p.add_argument('--expect', choices=['forward','keep'], default='forward')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--phase', choices=['both','transport','recovery'], default='both')
    a = p.parse_args()
    results = {}; call = None
    if a.phase in ('both','transport'):
        results['transport'], call = transport(a)
    if a.phase in ('both','recovery'):
        results['recovery'] = recovery(a, call)
    print('SUMMARY: ' + ', '.join(k+'='+v for k,v in results.items()))
    return 1 if 'FAIL' in results.values() else 2 if 'INCONCLUSIVE' in results.values() else 0

if __name__ == '__main__':
    sys.exit(main())
