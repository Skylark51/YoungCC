"""Parse explicit CPU reservations for the SSH worker launchers."""

import re


def worker_hosts(specifications):
    hosts = {}
    names = set()
    for specification in specifications:
        host, separator, value = specification.partition("=")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", host):
            raise ValueError(f"Invalid worker hostname: {host}")
        name = host.split(".")[0].lower()
        if name in names:
            raise ValueError(f"Duplicate worker host: {host}")
        names.add(name)
        cpus = None
        if separator:
            values = value.split(",")
            if not all(cpu.isascii() and cpu.isdecimal() for cpu in values):
                raise ValueError(f"Use a comma-separated CPU list for {host}.")
            cpus = sorted({int(cpu) for cpu in values})
            if len(cpus) != len(values):
                raise ValueError(f"The CPU assignment for {host} contains duplicates.")
        hosts[host] = cpus
    if not hosts:
        raise ValueError("Supply at least one worker host.")
    return hosts


def reserved_cpus(host, available, requested):
    if requested is None:
        return available
    if not set(requested) <= set(available):
        raise ValueError(f"The requested CPUs are outside the available affinity on {host}.")
    return requested
