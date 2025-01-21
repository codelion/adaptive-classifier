from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

test_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-randomly>=3.12.0",
    "psutil",
]

setup(
    name="adaptive-classifier",
    version="0.0.4",
    author="codelion",
    author_email="codelion@okyasoft.com",
    description="A flexible, adaptive classification system for dynamic text classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/codelion/adaptive-classifier",
    project_urls={
        "Bug Tracker": "https://github.com/codelion/adaptive-classifier/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "test": test_requirements,
    },
    include_package_data=True,
)
