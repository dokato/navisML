from setuptools import setup, find_packages
import sys, os

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.md')).read()
NEWS = open(os.path.join(here, 'NEWS.md')).read()


version = '0.0.1'

extras_require: DefaultDict[str, List[str]] = defaultdict(list)
install_requires: List[str] = []
reqs = install_requires

with open("requirements.txt") as f:
    for line in f:
        if line.startswith("#extra: "):
            extra = line[8:].split("#")[0].strip()
            reqs = extras_require[extra]
        elif not line.startswith("#") and line.strip():
            reqs.append(line.strip())

dev_only = ["test-notebook", "dev"]
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
    long_description=README + '\n\n' + NEWS,
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
    package_dir = {'': 'src'},include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    extras_require=dict(extras_require),
    python_requires='>=3.7',
    #entry_points={
    #    'console_scripts':
    #        ['navisML=navisml:main']
    #}
)
