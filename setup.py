from setuptools import setup, find_packages

setup(
    name="dcec",
    version="0.1",
    author="Sohil Atul Shah and Vladlen KoltunXifeng Guo, Xinwang Liu, En Zhu, Jianping Yin",
    url="https://github.com/lbasora/DCEC/",
    description="Deep Clustering with Convolutional Autoencoders",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "keras",
        # "tensorflow-gpu",
        "numpy",
        "pandas",
        "traffic",
        "altair",
    ],
    python_requires=">=3.6",
)
