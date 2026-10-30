"""What both CLM backends and the router share."""

HEADS_FILE = "CLM_v0.1-8B.safetensors"     # the name the released heads use, so trained ones load the same way


class ClmError(RuntimeError):
    """CLM could not be asked; the message says what to do about it."""


def not_installed(error: ModuleNotFoundError) -> ClmError:
    """The error for a CLM module imported without the clm extra, which the core install leaves out."""
    return ClmError(f"CLM is not installed ({error.name} is missing): install the clm extra, "
                    "pip install 'clm-decision-agent[clm]', or run uv sync in a checkout")
