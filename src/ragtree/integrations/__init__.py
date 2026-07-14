# ragtree/integrations/__init__.py
"""Optional adapters connecting external stacks to the core protocols.

Import rule (design doc, section 7.2): adapter modules import external SDKs
lazily, inside constructors or methods, after ``require_extra`` — never at
module import time.
"""
