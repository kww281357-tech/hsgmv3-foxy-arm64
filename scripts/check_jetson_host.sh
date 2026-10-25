#!/bin/bash
# Read-only preflight. Run on the intended Jetson host, not the cloud workstation.
set -euo pipefail
[[ "$(uname -m)" == aarch64 ]] || { echo "Expected ARM64 Jetson host" >&2; exit 1; }
[[ -r /etc/nv_tegra_release ]] || { echo "Missing NVIDIA L4T release metadata" >&2; exit 1; }
case "${L4T_TAG:-}" in
  r35.2.1) revision="2\.1" ;;
  r35.3.1) revision="3\.1" ;;
  *) echo "Set L4T_TAG to r35.2.1 or r35.3.1 after checking host release" >&2; exit 1 ;;
esac
grep -Eq "^# R35 .*REVISION: ${revision}," /etc/nv_tegra_release || {
  echo "L4T_TAG does not match the host release; do not run this GPU image." >&2
  head -n 1 /etc/nv_tegra_release
  exit 1
}
echo "${L4T_TAG} confirmed. Board model:"
tr -d '\0' < /proc/device-tree/model || true
echo
docker info --format '{{json .Runtimes}}'
echo "Verify nvidia runtime is listed. This script changes no host configuration."
