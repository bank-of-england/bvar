# Contributor Guide
## Initial Setup

1. **Fork and clone the repository**
```bash
git clone https://github.com/bank-of-england/bvar.git
cd bvar
```

2. **Set up the development environment**
```bash
conda create --name bvar
conda activate bvar
conda install pip
pip install -e ".[dev, docs, notebooks]"  # Install development and notebook dependencies.
```

3. **Install pre-commit hooks**
```bash
pre-commit install
```

Pre-commit runs the following checks when you commit changes:

- Ruff's linter, with automatic fixes where possible
- Ruff's formatter
- API documentation generation from the public exports in `src/bvar`
- NumPy-style docstring validation with `pydoclint`
- The full pytest suite

The generated API manifest is written to `docs/api.md`. If a source module's
public API changes, the hook updates that file so it can be included in the
same commit. To run every hook across the repository without creating a
commit, use:

```bash
pre-commit run --all-files
```

Pre-commit does not build the documentation site or distribution package; run
the commands in the [Documentation](#documentation) and [Code Style](#code-style)
sections when you need to check those artefacts.

4. **Verify the installation**
```bash
pytest
```

### Running the ecosystem tests

The `dev` extra installs `opera-eco[test]`, so the ecosystem gate can be run
locally. Build the wheel, overlay it on the pinned ecosystem, and run
opera-eco's shared contract and pipeline tests:

```bash
pytest --pyargs opera.tests -m contract -vv
pytest --pyargs opera.tests -m pipeline -vv
```

## Development Workflow

### Branch Strategy

- **`main`** — protected, release-only. No direct pushes; changes arrive by pull request from `dev`.
- **`dev`** — integration branch. Open feature and fix branches from `dev` and merge them back into `dev`.
- **Feature branches**: `feature/<issue>`
- **Bug fixes**: `fix/<issue>`
- **Documentation**: `docs/<topic>`

## Protected Branches and Pull Requests

All contributions must be submitted through a pull request. The `main` is protected, so contributors cannot push changes directly to it.

Branch out from `dev`, commit and push your changes there, then open a pull request targeting `dev`.

### PR checks

- **`package-quality`** — builds and inspects the distribution, runs Ruff lint and format checks, verifies the generated API docs, builds the documentation site in strict mode, and runs the test suite. This workflow must pass.
- **`ecosystem`** ("Ecosystem gate") — builds this module's wheel, installs it into the ecosystem pinned by `opera-eco[test]`, and runs opera-eco's shared contract and pipeline tests.

### Creating a Feature Branch

```bash
git checkout dev
git pull origin dev
git checkout -b feature/xyz
```

### Commit your changes

Use [Conventional Commit](https://www.conventionalcommits.org/) subjects
(`fix:`, `feat:`, `deps:`, `docs:`, `chore:`, ...); Release Please builds the
changelog and the next version from them.

```bash
git add .
git commit -m "fix: describe your change"
git push
```

## Code Standards

### Code Style

We use **Ruff** for formatting and linting:

```bash
# Format the code.
ruff format .

# Check for lint issues.
ruff check .

# Fix issues that Ruff can resolve.
ruff check . --fix

# Check formatting without changing files.
ruff format --check .
```

## Documentation

The documentation site is built with Zensical. API pages are rendered by
`mkdocstrings` from the public objects listed in each package `__all__`
declaration. `docs/api.md` is the generated manifest that connects those
objects to the API reference; edit the source docstrings and public exports,
not the generated directives in that file.

Install the documentation dependencies in an existing development
environment with:

```bash
pip install -e ".[docs]"
```

Regenerate the API manifest explicitly when needed:

```bash
python scripts/generate_api_docs.py
```

Build the complete documentation site locally, including strict validation:

```bash
zensical build --clean --strict
```

The continuous integration checks regenerate `docs/api.md` and fail if that
produces a diff, so generated API documentation cannot become stale silently.

### Naming Conventions

- **Variables**: `snake_case`
- **Functions/methods**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private functions/methods**: `_leading_underscore`

## Submitting Changes

### Before Submitting

1. **Open an issue** to discuss the bug or feature.

2. **Use the issue number in the branch name; for example, `fix/1-prior`.**

3. **Make your changes**

4. **Add a test for the change.**


5. **Format, document, and test the code.**
```bash
ruff format
ruff check .
pytest
```

6. **Commit and push the changes** with a Conventional Commit subject.
```bash
git add .
git commit -m "fix: describe your change (#1)"
git push origin fix/1-prior
```

7. **Submit a pull request.**

## Creating a Release (for maintainers)

Releases are automated with [Release Please](https://github.com/googleapis/release-please).
`release-please.yml` watches `main` for [Conventional Commits](https://www.conventionalcommits.org/)
and opens or updates a release pull request carrying the next version and the
generated `CHANGELOG.md` entries. Only recognised subject types
(`fix:`, `feat:`, `deps:`, ...) are picked up; an untyped subject is ignored.

`release-please-config.json` sets an `always-bump-patch` strategy, so every
release is a patch bump: `fix:`, `feat:`, `deps:`, and breaking commits all
take `0.3.1` to `0.3.2`. Commit types still organise the changelog but do not
change the version bump. A `Release-As: 0.4.0` footer on a typed commit is an
exact one-time override; it is not needed for normal changes.

`release-please.yml` enables auto-merge on the release pull request, so GitHub
merges it once the required `package-quality` and `ecosystem` checks and branch
protection pass. Keep work on `dev` until it is ready for the automatic release
path through `main`.

When the release pull request merges, Release Please creates the `v<version>`
tag and the GitHub Release. The published release then starts:

- `publish-pypi.yml` — builds the distribution and publishes it to PyPI, then
  triggers `update-ecosystem.yml` to re-pin `bvar` in `opera-eco`.
- `deploy-docs.yml` — builds the documentation site and deploys it to GitHub Pages.

Both also support manual dispatch against an existing tag.

### One-time setup

- Add a `RELEASE_PLEASE_TOKEN` repository secret: a token that can write
  contents, issues, pull requests, tags, and releases. A plain `GITHUB_TOKEN`
  will not do, because the release it creates must be able to trigger the
  downstream publication and documentation workflows.
- Enable **Allow auto-merge** in the repository settings.
- Require the `package-quality` (and `ecosystem`) checks on `main`, and make
  sure required human reviews do not block the automation pull requests.
