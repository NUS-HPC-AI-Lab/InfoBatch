import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="infobatch",
    license='MIT',
    version="0.1.0",
    author="Ziheng Qin",
    author_email="zihengq@comp.nus.edu.sg",
    description="InfoBatch code implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/henryqin1997/InfoBatch",
    packages=setuptools.find_packages(),
    keywords=[
        'artificial intelligence',
        'deep learning',
        'dataloader',
        'acceleration'
    ],
    install_requires=[
        'torch>=1.11'
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
)