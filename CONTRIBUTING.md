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

## Development Workflow

### Branch Strategy

- **`main`**: Production-ready code
- **Feature branches**: `feature/your-feature-name`
- **Bug fixes**: `fix/issue-description`
- **Documentation**: `docs/topic-name`

### Creating a Feature Branch

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

### Keeping Your Branch Updated

```bash
git checkout main
git pull origin main
git checkout feature/your-feature-name
   git rebase main  # Merge main instead when necessary.
```

### Commit your changes

```bash
git add .
git commit -m "describe your changes"
   git push  # Or specify the branch explicitly.
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

# Build the package.
python -m build

# Build the documentation.
zensical build
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

6. **Commit and push the changes.**
```bash
git add .
git commit -m "Fixes #1: Describe your changes"
git push origin fix/#1-prior
```

7. **Submit a pull request.**

## Creating a Release (for maintainers)

The release automation starts when you push a version tag. The tag must use
the `v<version>` format and match the version in `pyproject.toml`.

For example, to release version `0.1.1`:

1. Update `version` in `pyproject.toml` to `0.1.1`.
2. Move the relevant notes from the `Unreleased` section of `CHANGELOG.md` to
   a `0.1.1` section.
3. Run the quality checks locally:

```bash
ruff check .
ruff format --check .
python scripts/generate_api_docs.py
git diff --exit-code -- docs/api.md
pydoclint --style=numpy .
zensical build --clean --strict
pytest
```

4. Commit the version and changelog updates, then create the matching tag:

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "Prepare release 0.1.1"
git tag v0.1.1
git push origin main v0.1.1
```

Pushing the tag starts `.github/workflows/create-release.yml`. That workflow calls the reusable quality workflow and creates the GitHub Release only when Ruff, API documentation checks, pydoclint, the strict Zensical build, and the full test suite pass. A failed check prevents release creation.

When GitHub publishes the release, two workflows start automatically:

- `publish-pypi.yml` builds the distribution and publishes it to PyPI.
- `deploy-docs.yml` builds the documentation site and deploys it to GitHub Pages.

The release workflow generates GitHub release notes from the commit history.
Keep `CHANGELOG.md` up to date as the project record; the workflow does not edit that file automatically. The `workflow_dispatch` options in the individual workflows provide manual operations and do not replace the normal tagged-release sequence.
