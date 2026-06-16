# RAGTree architecture and testability notes

This document summarizes the repository structure behind the README.

## Key decision

RAGTree should use a `src/` layout and a separated test hierarchy.

## Main separation

```text
src/ragtree/integrations/    # adapter implementation code
tests/integration/           # tests for real optional integrations
```

## Core import rule

Core modules do not import external optional SDKs. Optional dependencies stay behind adapters.

## Test layers

- `tests/unit/`: pure logic.
- `tests/contract/`: protocol conformance.
- `tests/integration/`: optional real stacks.
- `tests/e2e/`: tiny full pipelines.
- `tests/regression/`: protect existing experiment outputs.

## Migration rule

Current scripts remain runnable until they are wrapped by stable package APIs.
