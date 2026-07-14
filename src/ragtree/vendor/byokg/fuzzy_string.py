# ragtree/vendor/byokg/fuzzy_string.py
import difflib

def fuzzy_match(query, candidates, top_k=5):
    """
    Perform fuzzy matching using difflib.
    :param query: Query string
    :param candidates: List of candidate strings
    :param top_k: Number of top matches to return
    :return: List of (candidate, score) tuples
    """
    matches = difflib.get_close_matches(query, candidates, n=top_k, cutoff=0.0)
    scored = [(m, difflib.SequenceMatcher(None, query, m).ratio()) for m in matches]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
