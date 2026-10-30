"""What both CLM backends and the router share."""

HEADS_FILE = "CLM_v0.1-8B.safetensors"     # the name the released heads use, so trained ones load the same way


class ClmError(RuntimeError):
    """CLM could not be asked; the message says what to do about it."""
