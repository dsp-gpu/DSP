# DSP-GPU Architecture

> **Источник данных**: автоматический анализ GPUWorkLib (880 файлов, 110K LOC)
> **Дата**: 2026-04-12
> **Статус**: Фаза 0 — базовая архитектура для обсуждения

## Содержание

| Документ | Описание |
|----------|----------|
| [dependencies.md](dependencies.md) | Граф зависимостей (Mermaid) + таблица + внешние SDK |
| [repo_map.md](repo_map.md) | Карта репозиториев: что куда переносим |
| [repo_structure.md](repo_structure.md) | Шаблон структуры каждого репо |

## Быстрый обзор

```
9 репозиториев в github.com/dsp-gpu/
  core → spectrum → signal_generators → heterodyne → radar → strategies
                                                       ↑
  core → stats ─────────────────────────────────────────┘
  core → linalg → capon ─────────────────────────────→ strategies
```

Исходный проект: `GPUWorkLib` (E:\C++\GPUWorkLib на Windows, .../C++/GPUWorkLib на Debian)
