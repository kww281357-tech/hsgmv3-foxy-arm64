# Source this file inside the isolated Habitat environment; no installation.
export LIBGL_ALWAYS_SOFTWARE=true
export GALLIUM_DRIVER=llvmpipe
export MESA_SHADER_CACHE_DISABLE=true
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/blue-help-matplotlib}"
