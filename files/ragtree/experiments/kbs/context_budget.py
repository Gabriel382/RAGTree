"""Matched maximum external-evidence token budget for three audited prompt formats.

Only the explicit retrieval/tool evidence section is shortened. Document, entity
IDs, relation schema, demonstrations, system messages and output instructions stay
untouched. Unknown prompt formats are rejected (not silently budgeted).
"""
from __future__ import annotations
from typing import Callable

EVIDENCE_HEADERS={
 'growlrag': [('## Ontology-guided retrieval context','## Output format')],
 'community_kgrag': [('## Tool: CommunityKG-RAG evidence sentences (retrieved from corpus via KG communities)','## Output format')],
 'langgraph_agentic_simple': [('## Tool: Web context','## Tool: Wikidata context'),
                               ('## Tool: Wikidata context','## Output format')]
}

def _extract(text:str,start:str,end:str):
    s=text.find(start)
    if s<0:raise ValueError('Missing evidence boundary: '+start)
    begin=s+len(start)
    stop=text.find(end,begin)
    if stop<0:raise ValueError('Missing evidence terminator: '+end)
    return begin,stop,text[begin:stop]

def enforce_external_evidence_budget(messages:list[dict],method:str,budget:int,count_tokens:Callable[[str],int]):
    if method not in EVIDENCE_HEADERS:raise ValueError(f'Unsupported evidence budget method: {method}')
    if budget<=0:raise ValueError('Budget must be positive')
    copied=[dict(m) for m in messages]
    user_idx=next((i for i in range(len(copied)-1,-1,-1) if copied[i].get('role')=='user'),None)
    if user_idx is None:raise ValueError('No user message for retrieval context budget')
    content=copied[user_idx].get('content')
    if not isinstance(content,str):raise ValueError('Expected string user prompt')
    ranges=[_extract(content,start,end) for start,end in EVIDENCE_HEADERS[method]]
    original=[count_tokens(v.strip()) for _,_,v in ranges]
    total=sum(original)
    if total<=budget:
        return copied,{'before_tokens':total,'after_tokens':total,'budget_tokens':budget,'truncated':False}
    # Proportional allocation avoids giving the web tool all the tokens at
    # the expense of Wikidata simply because it appears first in the prompt.
    limits=[int(budget*(v/total)) for v in original]
    for i in range(budget-sum(limits)):
        limits[i%len(limits)]+=1
    # Work backwards so offsets for earlier ranges remain valid.
    updated=content
    for (start,end,value),limit in sorted(zip(ranges,limits),key=lambda item:item[0][0],reverse=True):
        if count_tokens(value.strip())<=limit:continue
        low,high=0,len(value)
        while low<high:
            mid=(low+high+1)//2
            if count_tokens(value[:mid].strip())<=limit:low=mid
            else:high=mid-1
        updated=updated[:start]+value[:low]+updated[end:]
    after=[_extract(updated,start,end)[2] for start,end in EVIDENCE_HEADERS[method]]
    final_tokens=sum(count_tokens(v.strip()) for v in after)
    if final_tokens>budget:
        raise AssertionError(f'Budget enforcement failed: {final_tokens}>{budget}')
    copied[user_idx]['content']=updated
    return copied,{'before_tokens':total,'after_tokens':final_tokens,
                   'budget_tokens':budget,'truncated':True}
