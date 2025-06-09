import setuptools

setuptools.setup(
    name="TTI",
    version='0.1.0',
    author=("Junhong Shen & Hao Bai"),
    description="Code for Improving Agent Reasoning by Scaling Test-Time Interaction",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    license='MIT',
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires='>=3.7',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)

