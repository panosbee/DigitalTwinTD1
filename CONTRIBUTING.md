# Contributing to Digital Twin T1D SDK

First off, thank you for considering contributing to the Digital Twin T1D SDK! 💙 It's people like you who make this project a reality that can help 1 billion people with diabetes worldwide.

## Our Mission

"Technology powered by love can change the world" - this isn't just a tagline, it's our core belief. Every contribution, no matter how small, brings us closer to keeping our Christmas Promise: "Kids will be able to enjoy Christmas sweets again!"

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

**When reporting a bug, include:**
- A clear and descriptive title
- Steps to reproduce the behavior
- Expected behavior
- Screenshots (if applicable)
- Your environment (OS, Python version, etc.)

### 💡 Suggesting Enhancements

We love new ideas! When suggesting enhancements:
- Use a clear and descriptive title
- Provide a step-by-step description of the suggested enhancement
- Explain why this enhancement would be useful
- Include mockups or examples if possible

### 🔧 Pull Requests

1. Fork the repo and create your branch from `main`:
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. Make your changes following our coding standards:
   - Follow PEP 8 for Python code
   - Add type hints to all functions
   - Include docstrings (Google style)
   - Write/update tests as needed

3. Ensure the test suite passes:
   ```bash
   pytest
   ```

4. Update documentation if needed:
   ```bash
   cd docs
   make html
   ```

5. Commit your changes:
   ```bash
   git commit -m "Add amazing feature that helps diabetes management"
   ```

6. Push to your fork and submit a pull request

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/digital-twin-t1d.git
   cd digital-twin-t1d
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # For development tools
   ```

4. Install in development mode:
   ```bash
   pip install -e .
   ```

## Code Style Guidelines

### Python
- Use PEP 8
- Maximum line length: 100 characters
- Use type hints for all function arguments and returns
- Docstrings for all public functions/classes (Google style)

### Example:
```python
def predict_glucose(
    history: List[float], 
    horizon_minutes: int = 30
) -> Tuple[float, float]:
    """Predicts future glucose levels.
    
    Args:
        history: List of recent glucose values in mg/dL.
        horizon_minutes: Prediction horizon in minutes.
        
    Returns:
        Tuple of (predicted_value, confidence_interval).
    """
    # Implementation
    pass
```

### Git Commit Messages
- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues and pull requests liberally

## Testing

- Write tests for any new functionality
- Ensure all tests pass before submitting PR
- Aim for >90% code coverage
- Include both unit and integration tests

### Running Tests:
```bash
# All tests
pytest

# With coverage
pytest --cov=sdk --cov=models

# Specific module
pytest tests/test_models.py
```

## Documentation

- Update docstrings for any changed functionality
- Update README.md if adding new features
- Add examples to the docs/examples folder
- Update the Sphinx documentation if needed

## Community

- Be respectful and inclusive
- Help others in issues and discussions
- Share your success stories!
- Remember: we're building this with ❤️

## Areas We Need Help

### 🎯 High Priority:
- Clinical validation studies
- Real-world device integrations
- Performance optimizations
- Security audits

### 🔍 Always Welcome:
- Documentation improvements
- Bug fixes
- Test coverage expansion
- New model implementations
- UI/UX for dashboard

## Recognition

All contributors will be added to our CONTRIBUTORS.md file and will receive our eternal gratitude! 🌟

## Questions?

Feel free to:
- Open an issue with the label 'question'
- Contact us at: contribute@digitaltwin-t1d.org
- Join our Discord community

---

Remember: Every line of code you write could help save a life. That's the power of technology created with love.

