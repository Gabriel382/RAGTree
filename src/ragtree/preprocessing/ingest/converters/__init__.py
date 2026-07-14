# ragtree/preprocessing/ingest/converters/__init__.py

# Import all converter modules so their @register decorators run
from . import causalbank       # noqa: F401
from . import eventstoryline   # noqa: F401
from . import docred_causal    # noqa: F401
from . import fincausal        # noqa: F401
from . import maven_ere        # noqa: F401
