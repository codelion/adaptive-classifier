# Contributing to Adaptive Classifier

We love your input! We want to make contributing to Adaptive Classifier as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Pull Request Process

1. Update the README.md with details of changes to the interface, if applicable.
2. Update the docs/ with any new API changes or features.
3. The PR will be merged once you have the sign-off of two other developers.

## Any contributions you make will be under the Apache-2.0 license

In short, when you submit code changes, your submissions are understood to be under the same [Apache-2.0 license](https://choosealicense.com/licenses/apache-2.0/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issue tracker]

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/codelion/adaptive-classifier/issues/new); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Local Development Setup

1. Clone the repository:
```bash
git clone https://github.com/codelion/adaptive-classifier.git
cd adaptive-classifier
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

4. Run tests:
```bash
pytest tests/
```

## Code Style

We use:
- [Black](https://github.com/psf/black) for code formatting
- [isort](https://github.com/PyCQA/isort) for import sorting
- [flake8](https://github.com/PyCQA/flake8) for linting

Before submitting a PR, please run:
```bash
black .
isort .
flake8 .
```

## Documentation

We use Google-style docstrings. Please ensure all new code is properly documented:

```python
def function_name(param1: str, param2: int) -> bool:
    """Short description.

    Longer description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: Description of when this is raised.
    """
```

## Testing

- Write tests for all new features
- Maintain or increase code coverage
- Run the full test suite before submitting a PR
