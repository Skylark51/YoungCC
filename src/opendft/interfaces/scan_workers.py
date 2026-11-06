"""SSH worker discovery and CPU allocation shared by the scan launcher."""
import json
import re
import shlex
import subprocess
import sys
import numpy as np

def _ssh(host, *command):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", host):
        raise ValueError(f"Invalid worker hostname: {host}")
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
            shlex.join(command)]

def _available(host):
    code = "import json,os; print(json.dumps(sorted(os.sched_getaffinity(0))))"
    output = subprocess.check_output(
        _ssh(host, sys.executable, "-c", code), stdin=subprocess.DEVNULL, text=True, timeout=15,
    )
    return json.loads(output)

def _available_memory_mb(host):
    output = subprocess.check_output(
        _ssh(host, "cat", "/proc/meminfo"), stdin=subprocess.DEVNULL, text=True, timeout=15,
    )
    return next(int(line.split()[1]) / 1024 for line in output.splitlines()
                if line.startswith("MemAvailable:"))

def _allocate_all_cores(hosts, affinities, task_count, threads, teams_per_host, *, host_limits=None):
    if host_limits is None:
        host_limits = [teams_per_host] * len(hosts)
    limits = [min(limit, len(cpus) // threads)
              for cpus, limit in zip(affinities, host_limits, strict=True)]
    if min(limits) < 1 or task_count < len(hosts):
        raise ValueError("Each participating host needs at least one team and one task.")
    counts = [1] * len(hosts)
    for _ in range(min(task_count, sum(limits)) - len(hosts)):
        eligible = (i for i, count in enumerate(counts) if count < limits[i])
        index = max(eligible, key=lambda i: len(affinities[i]) / (counts[i] + 1))
        counts[index] += 1
    return [
        (host, partition.tolist())
        for host, cpus, count in zip(hosts, affinities, counts, strict=True)
        for partition in np.array_split(cpus, count)
    ]
