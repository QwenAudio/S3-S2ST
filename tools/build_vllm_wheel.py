#!/usr/bin/env python3
"""Package the released Python snapshot with matching prebuilt CUDA extensions."""

import argparse
import base64
import csv
import hashlib
import io
from pathlib import Path
import zipfile


VERSION = "0.11.1+s3.1.cu128"
NATIVE_FILES = ("_C.abi3.so", "_moe_C.abi3.so", "cumem_allocator.abi3.so")
NATIVE_SHA256 = (
    "c6c15e55d6c02b793ac293b87b54ff2fa712530bf77334091e7a536f4927d6d3",
    "d1769ff4a33399e2bd450a14a0bb0485253ba4e3cf86cfac76ef1ac912e35f01",
    "6a472922bef5df322a355ab678500296156a419ef6280e49972961b8ee286166",
)


def build(source, native, output):
    for name, expected in zip(NATIVE_FILES, NATIVE_SHA256):
        if not (native / name).is_file():
            raise FileNotFoundError(native / name)
        digest = hashlib.sha256()
        with (native / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Native extension does not match the reference runtime: {name}")
    output.mkdir(parents=True, exist_ok=True)
    wheel = output / f"vllm-{VERSION}-cp312-cp312-linux_x86_64.whl"
    info = f"vllm-{VERSION}.dist-info"
    records = []
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        def add(name, data):
            entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            records.append((name, "sha256=" + digest, str(len(data))))

        for path in sorted(source.rglob("*")):
            rel = path.relative_to(source)
            if not path.is_file() or any(p.startswith(".") or p == "__pycache__" for p in rel.parts):
                continue
            if path.suffix in (".pyc", ".pyo", ".so"):
                continue
            add("vllm/" + rel.as_posix(), path.read_bytes())
        for name in NATIVE_FILES:
            add("vllm/" + name, (native / name).read_bytes())
        add(info + "/METADATA", (
            "Metadata-Version: 2.1\nName: vllm\nVersion: " + VERSION + "\n"
            "Summary: S3-S2ST vLLM runtime for NVIDIA PyTorch 25.03\n"
            "Requires-Python: >=3.12,<3.13\nLicense: Apache-2.0\n"
            "\nInstall with the accompanying inference-runtime.txt in the documented base container.\n"
        ).encode())
        add(info + "/WHEEL", b"Wheel-Version: 1.0\nGenerator: S3-S2ST\nRoot-Is-Purelib: false\nTag: cp312-cp312-linux_x86_64\n")
        add(info + "/licenses/LICENSE", (source / "LICENSE").read_bytes())
        add(info + "/licenses/NOTICE.md", (source / "NOTICE.md").read_bytes())
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerows(records)
        writer.writerow((info + "/RECORD", "", ""))
        entry = zipfile.ZipInfo(info + "/RECORD", date_time=(2026, 1, 1, 0, 0, 0))
        entry.compress_type = zipfile.ZIP_DEFLATED
        entry.external_attr = 0o100644 << 16
        archive.writestr(entry, buffer.getvalue())
    print(wheel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[1] / "vendor/vllm")
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.source_dir, args.native_dir, args.output_dir)
