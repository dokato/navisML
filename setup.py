from setuptools import setup, find_packages
import sys, os
from collections import defaultdict

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.md')).read()
NEWS = open(os.path.join(here, 'NEWS.md')).read()

version = '0.0.1'

extras_require   = defaultdict(list)
reqs = install_requires = []

with open("requirements.txt") as f:
    for line in f:
        if line.startswith("#extra: "):
            extra = line[8:].split("#")[0].strip()
            reqs = extras_require[extra]
        elif not line.startswith("#") and line.strip():
            reqs.append(line.strip())

dev_only = ["dev"]
specialized = ['r']
all_dev_deps = []
all_deps = []
for k, v in extras_require.items():
    if k in specialized:
        continue
    all_dev_deps.extend(v)
    if k not in dev_only:
        all_deps.extend(v)

extras_require["all"] = all_deps
extras_require["all-dev"] = all_dev_deps

setup(name='navisML',
    version=version,
    description="machine learning with neurons made easy",
    long_description=README + '\n' + NEWS,
    long_description_content_type='text/markdown',
    classifiers=[
      'Programming Language :: Python :: 3',
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering :: Bio-Informatics',
      'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords='navis neurons ml machine-learning neuroscience',
    author='dokato',
    author_email='',
    url='',
    license='MIT',
    packages=find_packages(include=["navisML", "navisML.*"]),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    extras_require=dict(extras_require),
    python_requires='>=3.7'
)
