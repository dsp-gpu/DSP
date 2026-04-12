#!/usr/bin/env bash
# build_all_debian.sh — Сборка всех 9 репо DSP-GPU на Debian/ROCm
#
# Запуск: bash ~/dsp-gpu/DSP/scripts/build_all_debian.sh
# Предполагает: все репо склонированы в ~/dsp-gpu/
# Требует: ROCm 7.2+, cmake >= 3.25, ninja-build

set -euo pipefail

PRESET="${1:-debian-local-dev}"
JOBS="${2:-$(nproc)}"
BASE_DIR="${HOME}/dsp-gpu"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'

build_repo() {
    local repo="$1"
    local dir="${BASE_DIR}/${repo}"
    echo -e "\n${YELLOW}━━━ Building: ${repo} ━━━${NC}"
    if [ ! -d "$dir" ]; then
        echo -e "${RED}ERROR: ${dir} not found — run git clone first${NC}"
        return 1
    fi
    cd "$dir"
    cmake -S . -B build --preset "${PRESET}"
    cmake --build build -j"${JOBS}"
    echo -e "${GREEN}✓ ${repo} — OK${NC}"
}

echo -e "${YELLOW}DSP-GPU build (preset=${PRESET}, jobs=${JOBS})${NC}"
echo -e "${YELLOW}Base: ${BASE_DIR}${NC}"

# Порядок важен: зависимости раньше зависимых
build_repo core
build_repo spectrum
build_repo stats
build_repo linalg
build_repo signal_generators
build_repo heterodyne
build_repo radar
build_repo strategies

echo -e "\n${GREEN}━━━ All repos built successfully! ━━━${NC}"
echo -e "Следующий шаг: запустить тесты на GPU:"
echo -e "  ${BASE_DIR}/core/build/test_core_main"
echo -e "  ${BASE_DIR}/spectrum/build/test_spectrum_main"
