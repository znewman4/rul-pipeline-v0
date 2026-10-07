"""Forward-library I/O (BristolFE / MATLAB exports)."""

from .bristolfe import (
    CANONICAL_KEYS,
    ForwardLibrary,
    flatten_response,
    load_forward_library,
    pairs_to_fmc_cube,
    save_forward_library_npz,
    validate_forward_library,
)

__all__ = [
    "CANONICAL_KEYS",
    "ForwardLibrary",
    "flatten_response",
    "load_forward_library",
    "pairs_to_fmc_cube",
    "save_forward_library_npz",
    "validate_forward_library",
]
