import re

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str):
    if text is None:
        return []
    return _TOKEN_RE.findall(str(text).lower())