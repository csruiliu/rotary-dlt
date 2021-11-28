from setuptools import setup, find_packages

if __name__ == "__main__":
    with open('README.md', 'r') as fh:
        long_description = fh.read()

    setup(
        name='rotary',
        version='1.0',
        author="ruiliu",
        description="Resource Arbitration System",
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=find_packages(
            exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
        ),
        setup_requires=["setuptools-pipfile"],
        use_pipfile=True,
        classifiers=[
            "Programming Language :: Python :: 3"
        ],
        python_requires=">=3.7",
    )
