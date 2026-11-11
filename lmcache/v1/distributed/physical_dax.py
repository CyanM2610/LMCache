# SPDX-License-Identifier: Apache-2.0
"""Map an explicitly assigned Device-DAX region without an anonymous fallback."""

# Standard
from dataclasses import dataclass
import ctypes
import mmap
import os
from pathlib import Path
import re
import stat
import sys


@dataclass(frozen=True)
class DaxDescriptor:
    """Observed device geometry and the bounded mapping owned by this process."""

    device: str
    capacity: int
    offset_bytes: int
    device_capacity: int
    alignment: int
    target_node: int
    assigned_region_bytes: int


def validate_region(
    capacity: int, alignment: int, offset: int, assigned: int, mapped: int
) -> None:
    """Reject out-of-bounds or unaligned mappings before opening memory.

    Args:
        capacity: Device capacity from sysfs, in bytes.
        alignment: Device mmap alignment from sysfs, in bytes.
        offset: Start of the assigned region, in bytes.
        assigned: Length explicitly assigned to this experiment, in bytes.
        mapped: Requested CXL pool size, in bytes.
    Raises:
        ValueError: If the mapping exceeds either bound or violates alignment.
    """
    if any(type(v) is not int for v in (capacity, alignment, offset, assigned, mapped)):
        raise ValueError("DAX geometry must use integer bytes")
    if min(capacity, alignment, assigned, mapped) <= 0 or offset < 0:
        raise ValueError("invalid DAX geometry")
    if alignment & (alignment - 1):
        raise ValueError("DAX alignment must be a power of two")
    if any(v % alignment for v in (offset, assigned, mapped)):
        raise ValueError("DAX region and mapped quota must match device alignment")
    if mapped > assigned or offset + assigned > capacity:
        raise ValueError("DAX mapping exceeds assigned region or physical device")


class PhysicalDaxMapping:
    """Own a physical mmap; callers pin it and retain it through CUDA completion."""

    def __init__(self, device: str, offset: int, assigned: int, mapped: int) -> None:
        """Validate the device and map only its explicitly assigned byte range.

        Args:
            device: Absolute /dev/daxN.M character-device path.
            offset: Assigned region start, aligned to the device mmap size.
            assigned: Authorized region length in bytes.
            mapped: Portion of the assigned region used as this pool.
        Raises:
            ValueError: If the path or observed geometry is unsuitable.
            OSError: If sysfs, device access, exclusive advisory lock or mmap fails.
            RuntimeError: On a non-Linux host.
        Notes:
            The advisory lock coordinates cooperating processes. Assignment of
            the physical range must be established outside this allocator.
            No data is initialized and no device mode is reconfigured here.
        """
        if sys.platform != "linux":
            raise RuntimeError("Device-DAX requires Linux")
        if not re.fullmatch(r"/dev/dax\d+\.\d+", device):
            raise ValueError("expected an explicit /dev/daxN.M device")
        # Import only on Linux so region validation remains usable in CPU tools.
        import fcntl

        metadata = os.stat(device, follow_symlinks=False)
        if not stat.S_ISCHR(metadata.st_mode):
            raise ValueError("physical DAX requires a character device")
        sysfs = Path("/sys/bus/dax/devices") / Path(device).name
        observed_dev = (sysfs / "dev").read_text().strip()
        if observed_dev != f"{os.major(metadata.st_rdev)}:{os.minor(metadata.st_rdev)}":
            raise ValueError("device does not match DAX sysfs identity")
        capacity = int((sysfs / "size").read_text().strip(), 0)
        alignment = int((sysfs / "align").read_text().strip(), 0)
        target = int((sysfs / "target_node").read_text().strip())
        validate_region(capacity, alignment, offset, assigned, mapped)
        self.descriptor = DaxDescriptor(
            device, mapped, offset, capacity, alignment, target, assigned
        )
        self.mapping: mmap.mmap | None = None
        self.address = 0
        self.fd = os.open(device, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            opened = os.fstat(self.fd)
            if opened.st_rdev != metadata.st_rdev or not stat.S_ISCHR(opened.st_mode):
                raise ValueError("DAX identity changed while opening device")
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.mapping = mmap.mmap(
                self.fd,
                mapped,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
                offset=offset,
            )
            self.address = ctypes.addressof(ctypes.c_char.from_buffer(self.mapping))
            if self.address % alignment:
                raise ValueError("kernel returned an unaligned Device-DAX mapping")
        except BaseException:
            if self.mapping is not None:
                self.mapping.close()
                self.mapping = None
            os.close(self.fd)
            self.fd = -1
            raise

    def close(self) -> None:
        """Unmap after CUDA unregistration; reject outstanding Python views.

        Raises:
            BufferError: If external views retain the mmap. The descriptor and
                advisory lock remain owned until a later successful close.
        """
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
        self.address = 0
