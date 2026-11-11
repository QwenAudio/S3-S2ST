# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable

import torch

from vllm.triton_utils.custom_cache_manager import (
    CustomCacheManager,
    maybe_set_triton_cache_manager,
)
from vllm.triton_utils.importing import (
    HAS_TRITON,
    TritonLanguagePlaceholder,
    TritonPlaceholder,
)
from vllm.utils.torch_utils import is_torch_equal_or_newer

if TYPE_CHECKING or HAS_TRITON:
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice
else:
    triton = TritonPlaceholder()
    tl = TritonLanguagePlaceholder()
    tldevice = TritonLanguagePlaceholder()


def _patch_triton_knobs():
    try:
        from triton import knobs
    except (ImportError, ModuleNotFoundError):
        return

    from triton.compiler.compiler import CompiledKernel

    setattr(CompiledKernel, "launch_enter_hook", knobs.runtime.launch_enter_hook)  # noqa
    setattr(CompiledKernel, "launch_exit_hook", knobs.runtime.launch_exit_hook)  # noqa


def _patch_triton_key():
    try:
        from triton.runtime.cache import triton_key
    except (ImportError, ModuleNotFoundError):
        return

    import triton.compiler.compiler
    triton.compiler.compiler.triton_key = triton_key


def _patch_triton_autotuner_config():
    try:
        from triton.runtime.autotuner import Config
    except (ImportError, ModuleNotFoundError):
        return

    config_parameters = inspect.signature(Config.__init__).parameters
    if ('num_consumer_groups' in config_parameters
        or "num_buffers_warp_spec" in config_parameters
    ):
        return

    def __patched_init__(
        self,
        *args: Any,
        num_consumer_groups: int | None = None,
        num_buffers_warp_spec: int | None = None,
        **kwargs: Any,
    ) -> None:
        return Config.__init_orig__(self, *args, **kwargs)

    Config.__init_orig__ = Config.__init__
    Config.__init__ = __patched_init__


def _patch_triton_compile_result():
    try:
        from torch._inductor.runtime.triton_heuristics import (
            TritonCompileResult,
            config_to_dict,
            get_first_attr,
            triton_version_uses_attrs_dict,
        )
    except (ImportError, ModuleNotFoundError):
        return

    # For torch 2.8.0dev in NGC 25.06 docker container.
    if not is_torch_equal_or_newer("2.8.0dev") or is_torch_equal_or_newer("2.10.0dev"):
        return

    # Only needed for triton==3.6.0
    if triton.__version__ != "3.6.0":
        return

    def make_launcher(self) -> Any:
        """
        Launching triton kernels is performance sensitive, we compile
        a custom Python function get the grid() and reorder the args to
        the underlying wrapper.
        """
        cfg = self.config
        compile_meta = self.compile_meta
        binary = self.kernel
        fn = binary.src.fn
        binary._init_handles()
        (call_args, def_args, none_args) = self._get_arg_lists(
            fn.arg_names, fn.constexprs
        )
        binary_shared = (
            binary.shared if hasattr(binary, "shared") else binary.metadata.shared
        )

        scope = {
            "grid_meta": cfg.kwargs,
            "bin": binary,
            "launch_enter_hook": binary.__class__.launch_enter_hook,
            "launch_exit_hook": binary.__class__.launch_exit_hook,
            "metadata": (
                binary.packed_metadata
                if hasattr(binary, "packed_metadata")
                else binary.metadata
            ),
            "shared": binary_shared,
            "num_warps": (
                binary.num_warps
                if hasattr(binary, "num_warps")
                else binary.metadata.num_warps
            ),
            "cta_args": (
                (
                    binary.num_ctas,
                    *get_first_attr(binary, "cluster_dims", "clusterDims"),
                )
                if hasattr(binary, "num_ctas")
                else (
                    (binary.metadata.num_ctas, *binary.metadata.cluster_dims)
                    if hasattr(binary, "metadata")
                    and hasattr(binary.metadata, "num_ctas")
                    and hasattr(binary.metadata, "cluster_dims")
                    else ()
                )
            ),
            "function": get_first_attr(binary, "function", "cu_function"),
            "runner": get_first_attr(binary, "run", "c_wrapper"),
        }

        if not hasattr(binary, "launch_metadata"):
            # launch args before CompiledKernel.launch_metadata is added.
            # TODO(jansel): delete this branch in mid-2025
            runner_args = [
                "grid_0",
                "grid_1",
                "grid_2",
                "num_warps",
                "*cta_args",
                "shared",
                "stream",
                "function",
                "launch_enter_hook",
                "launch_exit_hook",
                "metadata",
                *call_args,
            ]
        else:  # args after CompiledKernel.launch_metadata: https://github.com/openai/triton/pull/3492
            # Getting the kernel launch args is extremely perf-sensitive.  Evaluating
            # `bin.launch_metadata` is relatively expensive, and returns None unless a
            # `launch_enter_hook` is installed.  So if we don't have that hook
            # installed, we want to burn None in to the launch args with zero overhead.
            # See https://github.com/pytorch/pytorch/issues/123597
            if binary.__class__.launch_enter_hook:
                launch_metadata = f"bin.launch_metadata((grid_0, grid_1, grid_2), stream, {', '.join(call_args)})"  # noqa
            else:
                launch_metadata = "None"
            runner_args = [
                "grid_0",
                "grid_1",
                "grid_2",
                "stream",
                "function",
                "metadata",
                launch_metadata,
                "launch_enter_hook",
                "launch_exit_hook",
                *call_args,
            ]

        launcher = self._gen_launcher_code(scope, def_args, runner_args)

        launcher = scope["launcher"]
        launcher.config = cfg
        launcher.n_regs = getattr(binary, "n_regs", None)
        launcher.n_spills = getattr(binary, "n_spills", None)
        launcher.shared = binary_shared
        launcher.store_cubin = self.inductor_meta.get("store_cubin", False)
        # store this global variable to avoid the high overhead of reading it
        # when calling run
        if launcher.store_cubin:
            launcher.fn = fn
            launcher.bin = binary
            if triton_version_uses_attrs_dict():
                # arg filtering wasn't done above
                cfg_dict = config_to_dict(cfg)
                def_args = [x for x in def_args if x not in cfg_dict]
                call_args = [
                    x
                    for x in call_args
                    if compile_meta["signature"].get(x, "constexpr") != "constexpr"
                    and x not in none_args
                ]
            launcher.def_args = def_args
            launcher.call_args = call_args
        return launcher

    TritonCompileResult.make_launcher = make_launcher


def _patch_torch280_ngc2506_generate_ttir():

    # For torch 2.8.0dev in NGC 25.06 docker container.
    if not is_torch_equal_or_newer("2.8.0dev") or is_torch_equal_or_newer("2.10.0dev"):
        return

    # Only needed for triton==3.6.0
    if triton.__version__ != "3.6.0":
        return

    def patched_generate_ttir(
        kernel, kwargs: dict[str, Any]
    ) -> tuple[Any, list[str]]:
        """
        Uses Triton's internal code generation to create TTIR
        """
        import sympy
        import triton
        import triton.runtime.jit
        from torch._inductor.utils import (
            TritonAttrsDescriptorVersion,
            get_triton_attrs_descriptor_version,
            triton_version_uses_attrs_dict,
        )
        from triton.compiler.compiler import ASTSource
        from triton.runtime.autotuner import Autotuner
        from triton.runtime.jit import JITFunction

        triton_version = get_triton_attrs_descriptor_version()

        import torch._inductor.ir
        from torch._subclasses.fake_tensor import FakeTensor

        if isinstance(kernel, Autotuner):
            if len(kernel.configs) > 0:
                # If we are autotuning, then it doesn't matter which version gets
                # picked for tracing purposes, so lets pick the first one
                kwargs = {**kwargs, **kernel.configs[0].kwargs}
            kernel = kernel.fn

        assert isinstance(kernel, JITFunction)

        context = triton._C.libtriton.ir.context()
        target = triton.runtime.driver.active.get_current_target()
        backend = triton.compiler.compiler.make_backend(target)
        options = backend.parse_options({})

        # ignore backend-specific kwargs same way as in the native Triton code
        # https://github.com/triton-lang/triton/blob/a6bb57d6285e723c58e87dd7cba263db6efff789/python/triton/runtime/jit.py#L594-L596
        # why this is important for user-defined Triton kernels on AMD: https://github.com/pytorch/pytorch/issues/140800
        for name in list(kwargs):
            if name not in kernel.arg_names and name in options.__dict__:
                kwargs.pop(name)

        if len(kwargs) != len(kernel.arg_names):
            raise ValueError(
                "Incorrect number of arguments passed to kernel: "
                f"passed {list(kwargs.keys())}, expected {kernel.arg_names}."
            )

        # Replace all SymExprs with a regular value for TTIR generation
        # Replace all FakeTensor/TensorBox with real tensors
        # These replacements are needed for triton's type, key and config functions
        ordered_args: dict[str, Any] = {}
        for name in kernel.arg_names:
            a = kwargs[name]
            if isinstance(a, (torch.SymInt, torch.SymFloat, torch.SymBool, sympy.Expr)):
                ordered_args[name] = 2
            elif isinstance(a, (FakeTensor, torch._inductor.ir.TensorBox)):
                with torch._C._DisableTorchDispatch():
                    ordered_args[name] = torch.empty(2, dtype=a.dtype)
            else:
                ordered_args[name] = a

        ordered_tensor_names = [
            name for name, arg in ordered_args.items() if isinstance(arg, torch.Tensor)
        ]

        def _get_specialization(args):  # type: ignore[no-untyped-def]
            # Support multiple triton versions.
            # This code basically copies JITFunction.run() logic to get the attrs to
            # construct an ASTSource.
            if triton_version == TritonAttrsDescriptorVersion.V1_COMPILER:
                return kernel._get_config(*args)
            elif triton_version in {
                TritonAttrsDescriptorVersion.V2_BACKENDS,
                TritonAttrsDescriptorVersion.V3_BACKENDS_TUPLE,
            }:
                from triton.backends.compiler import AttrsDescriptor  # noqa: F401

                target = triton.runtime.driver.active.get_current_target()
                backend_ = triton.compiler.compiler.make_backend(target)
                return backend_.get_attrs_descriptor(args, kernel.params)
            else:
                assert (
                    get_triton_attrs_descriptor_version()
                    == TritonAttrsDescriptorVersion.V4_DICT
                )
                # specialize_impl switched to create_specialize_impl in https://github.com/triton-lang/triton/pull/6099
                if hasattr(triton.runtime.jit, "create_specialize_impl"):
                    try:
                        # Latest versions of Triton take specialize_extra as an arg to
                        # create_specialize_impl
                        specialize_impl = triton.runtime.jit.create_specialize_impl(
                            specialize_extra=backend.get_arg_specialization
                        )
                    except TypeError:  # Unknown arg `specialize_extra`
                        # Older versions of Triton take specialize_extra as an arg to
                        # specialize_impl
                        specialize_impl = functools.partial(
                            triton.runtime.jit.create_specialize_impl(),
                            specialize_extra=backend.get_arg_specialization,
                        )
                # create_specialize_impl is removed in https://github.com/triton-lang/triton/pull/7771
                # switch to native_specialize_impl instead
                elif hasattr(triton.runtime.jit, "native_specialize_impl"):
                    from triton.backends import BaseBackend
                    from triton.runtime.jit import native_specialize_impl

                    def _native_specialize_impl(
                        arg: Any,
                        is_const: bool = False,
                        specialize_value: bool = True,
                        align: bool = True,
                    ) -> Callable:
                        return native_specialize_impl(
                            BaseBackend, arg, is_const, specialize_value, align
                        )

                    specialize_impl = _native_specialize_impl
                else:
                    from triton.runtime.jit import (
                        specialize_impl as specialize_impl_orig,
                    )

                    specialize_impl = functools.partial(
                        specialize_impl_orig,
                        specialize_extra=backend.get_arg_specialization,
                    )

                from triton._utils import find_paths_if, get_iterable_path

                # logic is copied from: binder = create_function_from_signature(
                # self.signature, self.params, backend)
                attrvals = []
                for arg, kp in zip(args, kernel.params):
                    if kp.is_constexpr:
                        attrvals.append(arg)
                    else:
                        spec = specialize_impl(
                            arg,
                            is_const=kp.is_const,
                            specialize_value=not kp.do_not_specialize,
                            align=not kp.do_not_specialize_on_alignment,
                        )
                        attrvals.append(spec[1])

                attrs = find_paths_if(attrvals, lambda _, x: isinstance(x, str))
                attrs = {
                    k: backend.parse_attr(get_iterable_path(attrvals, k))
                    for k in attrs
                }
                return attrs

        specialization = _get_specialization(ordered_args.values())
        constants = {
            name: arg
            for name, arg in ordered_args.items() if not isinstance(arg, torch.Tensor)
        }

        if (
            (mangle_type := getattr(triton.runtime.jit, "mangle_type", None))
            is not None
        ):

            def get_signature_value(idx: int, arg: Any) -> str:
                if kernel.params[idx].is_constexpr:
                    return "constexpr"
                return mangle_type(arg)

        else:

            def get_signature_value(idx: int, arg: Any) -> str:
                return kernel._type_of(kernel.key_of(arg))

        if triton_version_uses_attrs_dict():
            # In newer versions of Triton, the signature includes constexpr args
            signature = {
                name: get_signature_value(i, arg)
                for i, (name, arg) in enumerate(ordered_args.items())
            }
        else:
            # In older versions of Triton, the signature does not include constexpr args
            signature = {
                name: get_signature_value(i, arg)
                for i, (name, arg) in enumerate(ordered_args.items())
                if i not in kernel.constexprs
            }

        triton._C.libtriton.ir.load_dialects(context)
        backend.load_dialects(context)

        src = ASTSource(kernel, signature, constants, specialization)

        # Triton changes ASTSource.make_ir to take 3/4 arguments. Handle
        # backward compatibility here.
        make_ir_sig_params = len(inspect.signature(src.make_ir).parameters)
        get_codegen_implementation_sig_params = len(
            inspect.signature(backend.get_codegen_implementation).parameters
        )
        if make_ir_sig_params == 2:
            ttir_module = src.make_ir(options, context)
        elif make_ir_sig_params == 3:
            codegen_fns = backend.get_codegen_implementation()
            ttir_module = src.make_ir(options, codegen_fns, context)
        elif make_ir_sig_params == 4:
            codegen_args = (
                [options] if get_codegen_implementation_sig_params == 1 else []
            )
            codegen_fns = backend.get_codegen_implementation(*codegen_args)
            module_map = backend.get_module_map()
            ttir_module = src.make_ir(options, codegen_fns, module_map, context)
        else:
            target = triton.runtime.driver.active.get_current_target()
            codegen_args = (
                [options] if get_codegen_implementation_sig_params == 1 else []
            )
            codegen_fns = backend.get_codegen_implementation(*codegen_args)
            module_map = backend.get_module_map()
            ttir_module = src.make_ir(target, options, codegen_fns, module_map, context)
        if not ttir_module.verify():
            raise RuntimeError("Verification for TTIR module has failed")

        return ttir_module, ordered_tensor_names

    import torch._higher_order_ops.triton_kernel_wrap
    torch._higher_order_ops.triton_kernel_wrap.generate_ttir = patched_generate_ttir


if HAS_TRITON:
    _patch_triton_knobs()
    _patch_triton_key()
    _patch_triton_autotuner_config()
    _patch_triton_compile_result()
    _patch_torch280_ngc2506_generate_ttir()


__all__ = [
    "HAS_TRITON",
    "CustomCacheManager",
    "maybe_set_triton_cache_manager",
    "triton",
    "tl",
    "tldevice",
]
