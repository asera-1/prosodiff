"""Domain-specific errors with user-facing messages."""


class ProsodiffError(Exception):
    """Base class for expected analysis failures."""


class AudioInputError(ProsodiffError):
    """Raised when an input recording cannot be analysed safely."""
