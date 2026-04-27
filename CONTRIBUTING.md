# Contributing to profi

Thanks for taking the time to contribute. Changes are easiest to review when
they are small, focused, and include the documentation or tests needed to make
the behavior clear.

## Development Setup

Build the project locally with:

```bash
make build
```

Install the repository hooks once:

```bash
make prek-install
```

Before opening a pull request, run the same local check suite used by CI:

```bash
make prek-run
```

## Pull Requests

Open pull requests against `main`. Include a concise description of the
problem, the approach, and any operational impact for Kubernetes, eBPF, metrics,
or Helm users. Keep unrelated refactors out of feature and bug-fix PRs.

If a change touches user-facing behavior, update the relevant documentation in
`README.md`, `docs/`, or `deploy/profi/README.md`. For benchmark or performance
claims, update `docs/BENCHMARK.md` with the workload, mode, and measurement
methodology.
