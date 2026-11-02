"""Test-only stand-in for LiteLLM. NEVER use on real inference."""
import os
class Response:
    _hidden_params={'response_cost':0.000012}
    def __init__(self):
        self.choices=[type('Choice',(),{'message':type('Message',(),{'content':'{}'})()})()]
    def model_dump(self):
        return {'usage':{'prompt_tokens':20,'completion_tokens':4},'choices':[{'message':{'content':'{}'}}]}
def completion(*,model,messages,**kwargs):
    if os.environ.get('KBS_TEST_FORCE_ERROR')=='1':raise RuntimeError('forced mock API exception')
    return Response()
def token_counter(model,messages):
    return sum(len(str(m.get('content','')))//4 for m in messages)
