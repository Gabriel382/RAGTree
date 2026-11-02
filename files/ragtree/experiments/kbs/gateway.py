"""LiteLLM adapter with actual provider usage and per-call tracing."""
from __future__ import annotations
import os
import time
from typing import Any, Sequence, Dict
from ragtree.services.llm.mock_client import BaseLLMClient
from .runtime import trace_call, trace_event

class LiteLLMClient(BaseLLMClient):
    def __init__(self, chat_model=None, embed_model=None):
        self.chat_model = chat_model or os.environ.get('KBS_MODEL','openai/gpt-oss-20b')
        self.embed_model = embed_model

    def chat(self, messages: Sequence[Dict[str,str]], **kwargs: Any) -> str:
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError('Install KBS dependencies: pip install -r requirements-kbs.txt') from exc
        model = self.chat_model
        data = dict(kwargs)
        # Preserve journal generation settings through configuration, not hardcoded API overrides.
        data.setdefault('temperature',float(os.environ.get('KBS_TEMPERATURE','0')))
        data.setdefault('max_tokens',int(os.environ.get('KBS_MAX_TOKENS','1024')))
        data.setdefault('timeout',int(os.environ.get('KBS_LLM_TIMEOUT','240')))
        data.setdefault('num_retries',int(os.environ.get('KBS_API_RETRIES','2')))
        if os.environ.get('KBS_LITELLM_API_BASE'):
            data['api_base'] = os.environ['KBS_LITELLM_API_BASE']
        if os.environ.get('KBS_LITELLM_API_KEY'):
            data['api_key'] = os.environ['KBS_LITELLM_API_KEY']
        evidence_budget = os.environ.get('KBS_EVIDENCE_BUDGET_TOKENS')
        if evidence_budget:
            from .context_budget import enforce_external_evidence_budget
            method = os.environ.get('KBS_METHOD','')
            def token_counter(text):
                # Provider/model tokenizer if LiteLLM supports it. Never apply
                # a naive character cutoff masquerading as a token budget.
                return max(0,litellm.token_counter(model=model, messages=[{'role':'user','content':text}]) - litellm.token_counter(model=model,messages=[{'role':'user','content':''}]))
            messages,measure = enforce_external_evidence_budget(list(messages),method,int(evidence_budget),token_counter)
            trace_event('matched_external_evidence',measure)
        start=time.monotonic()
        request={'model':model,'messages':list(messages),'parameters': {k:v for k,v in data.items() if k!='api_key'}}
        try:
            result = litellm.completion(model=model,messages=list(messages),**data)
            payload = result.model_dump() if hasattr(result,'model_dump') else dict(result)
            price = getattr(result,'_hidden_params',{}) or {}
            price = price.get('response_cost')
            if price is not None:
                payload['kbs_litellm_reported_cost_usd']=price
            trace_call(request,payload,time.monotonic()-start)
            choice = result.choices[0].message
            content = getattr(choice,'content',None)
            if isinstance(content,str):
                if not content.strip():raise ValueError('LLM returned empty content (possibly consumed by reasoning tokens)')
                return content
            if isinstance(content,list):
                text='\n'.join(str(x.get('text','')) for x in content if isinstance(x,dict))
                if text.strip():return text
            raise ValueError('LLM returned no usable completion text')
        except Exception as exc:
            trace_call(request,None,time.monotonic()-start,repr(exc))
            raise

    def embed(self,text: str):
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError('pip install -r requirements-kbs.txt') from exc
        if not self.embed_model:
            raise ValueError('No embedding model configured')
        result=litellm.embedding(model=self.embed_model,input=[text])
        return list(result.data[0]['embedding'])
