from setuptools import setup, find_packages

setup(
    name="macecho",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # 在这里添加项目依赖
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A voice assistant that runs completely on your Mac",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/macecho",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
    ],
    python_requires=">=3.12",
)
