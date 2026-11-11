"""Import-only shim for the proven dense S2TT stack.

The historical Megatron tree imports ``flash_moe`` even for dense models.
H100-2 does not provide the optional package, while this 4B configuration has
``USE_MOE=false`` and never calls it.  Fail loudly if a future configuration
does try to execute a FlashMoE branch.
"""


def __getattr__(name):
    raise RuntimeError(
        f"flash_moe.{name} was accessed, but the streaming S2TT compatibility "
        "shim is import-only; install the real flash_moe package for MoE runs"
    )
