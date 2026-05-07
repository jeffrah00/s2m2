#!/usr/bin/env bash
# Install s2m2 + s2m2_ros2 on the current host.
#
# Auto-detects the Ubuntu codename and picks a sensible ROS2 distro:
#   Ubuntu 22.04 (jammy) -> humble
#   Ubuntu 24.04 (noble) -> jazzy   (override with --distro kilted)
#
# Override the auto-detect with --distro humble|jazzy|kilted, or by exporting
# ROS_DISTRO before running. Set --skip-pip or --skip-colcon to do only one
# half of the install.

set -euo pipefail

SUPPORTED_DISTROS=("humble" "jazzy" "kilted")
DISTRO=""
SKIP_PIP=0
SKIP_COLCON=0

usage() {
    cat <<'EOF'
Usage: scripts/install.sh [--distro humble|jazzy|kilted] [--skip-pip] [--skip-colcon] [-h|--help]

Auto-detects Ubuntu version and installs s2m2 (pip) + s2m2_ros2 (colcon) for
the matching ROS2 distro.

Distro resolution priority:
  1. --distro flag
  2. $ROS_DISTRO environment variable
  3. /etc/os-release VERSION_CODENAME (jammy -> humble, noble -> jazzy)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --distro)
            DISTRO="${2:-}"
            shift 2
            ;;
        --skip-pip)
            SKIP_PIP=1
            shift
            ;;
        --skip-colcon)
            SKIP_COLCON=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument '$1'" >&2
            usage
            exit 1
            ;;
    esac
done

contains() {
    local needle="$1"
    shift
    for x in "$@"; do
        [[ "$x" == "$needle" ]] && return 0
    done
    return 1
}

detect_distro_from_ubuntu() {
    if [[ ! -r /etc/os-release ]]; then
        echo "ERROR: /etc/os-release not readable; cannot auto-detect Ubuntu version." >&2
        echo "       Pass --distro humble|jazzy|kilted explicitly." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    . /etc/os-release
    case "${VERSION_CODENAME:-}" in
        jammy)  echo "humble" ;;
        noble)  echo "jazzy"  ;;
        *)
            cat >&2 <<EOF
ERROR: unsupported Ubuntu version: ${PRETTY_NAME:-unknown} (codename: ${VERSION_CODENAME:-unknown}).
       Supported: 22.04 (jammy) -> humble, 24.04 (noble) -> jazzy|kilted.
       Pass --distro humble|jazzy|kilted to override.
EOF
            exit 1
            ;;
    esac
}

# 1. Resolve distro
if [[ -z "$DISTRO" ]]; then
    if [[ -n "${ROS_DISTRO:-}" ]]; then
        DISTRO="$ROS_DISTRO"
        echo "==> Using ROS_DISTRO from environment: $DISTRO"
    else
        DISTRO="$(detect_distro_from_ubuntu)"
        echo "==> Auto-detected ROS2 distro: $DISTRO"
    fi
fi

if ! contains "$DISTRO" "${SUPPORTED_DISTROS[@]}"; then
    echo "ERROR: distro '$DISTRO' is not supported by this script." >&2
    echo "       Supported: ${SUPPORTED_DISTROS[*]}." >&2
    exit 1
fi

# 2. Source ROS2
ROS_SETUP="/opt/ros/${DISTRO}/setup.bash"
if [[ ! -f "$ROS_SETUP" ]]; then
    cat >&2 <<EOF
ERROR: $ROS_SETUP not found.
       Install ROS2 ${DISTRO} first, e.g.:
           sudo apt install ros-${DISTRO}-ros-base ros-${DISTRO}-cv-bridge \\
                            python3-colcon-common-extensions
EOF
    exit 1
fi
# shellcheck disable=SC1090
# ROS setup files reference optional vars (AMENT_TRACE_SETUP_FILES, etc.)
# without ${VAR:-} defaults, so relax `set -u` for the duration of the source.
set +u
source "$ROS_SETUP"
set -u
echo "==> Sourced $ROS_SETUP (ROS_DISTRO=$ROS_DISTRO)"

# 3. Repo root (the parent of this script)
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
echo "==> Repo root: $REPO_ROOT"
echo "==> Python: $(python3 --version) ($(command -v python3))"
if [[ -z "${CONDA_DEFAULT_ENV:-}" && -z "${VIRTUAL_ENV:-}" ]]; then
    echo "WARN: no active conda/venv detected; pip + colcon will use system Python." >&2
fi

# 4. pip install s2m2
if [[ "$SKIP_PIP" -eq 0 ]]; then
    echo "==> Installing s2m2 Python package (pip install -e . --break-system-packages --no-deps)"
    pip install --break-system-packages --no-deps -e "$REPO_ROOT"
else
    echo "==> Skipping pip install (--skip-pip)"
fi

# 5. colcon build s2m2_ros2
if [[ "$SKIP_COLCON" -eq 0 ]]; then
    echo "==> Building s2m2_ros2 with colcon"
    (
        cd "$REPO_ROOT/ros2_ws"
        colcon build --packages-select s2m2_ros2 --symlink-install
    )
else
    echo "==> Skipping colcon build (--skip-colcon)"
fi

cat <<EOF

Done.

Next steps:
  source $REPO_ROOT/ros2_ws/install/setup.bash
  ros2 launch s2m2_ros2 s2m2_depth.launch.py
EOF
