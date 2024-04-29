from importlib.metadata import version as _version, PackageNotFoundError
try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

# create the bibcat configuration object
from bibcat.core.config import get_config
config = get_config()


