cmake_minimum_required(VERSION 3.25)
project(DSP VERSION 0.1.0
  DESCRIPTION "DSP-GPU meta-repo: unified build of all modules"
  LANGUAGES NONE)

include(FetchContent)
include(cmake/fetch_deps.cmake)
dsp_check_dependency_guards()

# ── Условная сборка — управляется через пресеты ────────────────────
option(DSP_BUILD_CORE              "Build core"              ON)
option(DSP_BUILD_SPECTRUM          "Build spectrum"          ON)
option(DSP_BUILD_STATS             "Build stats"             ON)
option(DSP_BUILD_SIGNAL_GENERATORS "Build signal_generators" ON)
option(DSP_BUILD_HETERODYNE        "Build heterodyne"        ON)
option(DSP_BUILD_LINALG            "Build linalg"            ON)
option(DSP_BUILD_RADAR             "Build radar"             ON)
option(DSP_BUILD_STRATEGIES        "Build strategies"        ON)

# Централизованный FetchContent_Declare ПЕРЕД всеми зависимыми!
# Решает diamond dependency — все модули получают одну версию.
if(DSP_BUILD_CORE)
  fetch_dsp_core()
endif()
if(DSP_BUILD_SPECTRUM)
  fetch_dsp_spectrum()
endif()
if(DSP_BUILD_STATS)
  fetch_dsp_stats()
endif()
if(DSP_BUILD_SIGNAL_GENERATORS)
  fetch_dsp_signal_generators()
endif()
if(DSP_BUILD_HETERODYNE)
  fetch_dsp_heterodyne()
endif()
if(DSP_BUILD_LINALG)
  fetch_dsp_linalg()
endif()
if(DSP_BUILD_RADAR)
  fetch_dsp_radar()
endif()
if(DSP_BUILD_STRATEGIES)
  fetch_dsp_strategies()
endif()

# ── Слой 2: автоматический reconfigure при git pull в субрепо ──────
# Работает только в local-dev / debian-local-dev (где заданы FETCHCONTENT_SOURCE_DIR_*).
# В CI-режиме (GIT_TAG) — if(DEFINED ...) не срабатывает, блок молча пропускается.
set(_watch_modules core spectrum stats signal_generators
                   heterodyne linalg radar strategies)

foreach(_mod ${_watch_modules})
    string(TOUPPER "${_mod}" _up)
    string(REPLACE "_" "" _up "${_up}")
    set(_key "FETCHCONTENT_SOURCE_DIR_DSP${_up}")

    if(DEFINED ${_key})
        set(_dir "${${_key}}")

        # Поддержка git worktrees: .git может быть файлом
        if(IS_DIRECTORY "${_dir}/.git")
            set(_git_dir "${_dir}/.git")
        elseif(EXISTS "${_dir}/.git")
            file(READ "${_dir}/.git" _git_link)
            string(REGEX REPLACE "^gitdir: (.*)$" "\\1" _git_dir "${_git_link}")
            string(STRIP "${_git_dir}" _git_dir)
        else()
            continue()
        endif()

        # .git/index     — меняется при git pull, commit, merge
        # .git/FETCH_HEAD — меняется при git fetch/pull с remote
        foreach(_gf index FETCH_HEAD)
            if(EXISTS "${_git_dir}/${_gf}")
                set_property(DIRECTORY APPEND PROPERTY
                    CMAKE_CONFIGURE_DEPENDS "${_git_dir}/${_gf}")
            endif()
        endforeach()
    endif()
endforeach()
unset(_watch_modules)
unset(_mod)
unset(_up)
unset(_key)
unset(_dir)
unset(_git_dir)
