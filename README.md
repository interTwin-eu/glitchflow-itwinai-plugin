# Writing plugins for itwinai

[![GitHub Super-Linter](https://github.com/interTwin-eu/itwinai-plugin-template/actions/workflows/lint.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![GitHub Super-Linter](https://github.com/interTwin-eu/itwinai-plugin-template/actions/workflows/check-links.yml/badge.svg)](https://github.com/marketplace/actions/markdown-link-check)
 [![SQAaaS source code](https://github.com/EOSC-synergy/itwinai-plugin-template.assess.sqaaas/raw/main/.badge/status_shields.svg)](https://sqaaas.eosc-synergy.eu/#/full-assessment/report/https://raw.githubusercontent.com/eosc-synergy/itwinai-plugin-template.assess.sqaaas/main/.report/assessment_output.json)

It would be beneficial to integrate the interTwin use case code
directly into itwinai, providing an easy way to distribute and
compare AI methods used by various researchers.

However, requiring all researchers to maintain their code within
the itwinai repository can be complicated and may create an
unnecessary barrier for newcomers interested in joining this
ecosystem. Researchers and use case developers often prefer
the flexibility of developing their code in a separate repository.
This allows them to independently manage the packaging and
publishing of their software, for instance, by releasing it
to PyPI when it best suits their needs.

To address this challenge, itwinai implements a *plugin*
architecture, allowing the community to independently develop
sub-packages for itwinai. These sub-packages can later be installed
from PyPI and imported into Python code as if they were part of
itwinai. This is made possible by
[namespace packages](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/).

Itwinai plugins are developed by assuming that the plugin logic,
once installed alongside itwinai, will be accessible under
the `itwinai.plugins.*` namespace. Example:

```python
from itwinai.plugins.my_awesome_plugin import awesome_module

# Call a function implemented by the plugin
awesome_module.awesome_function(some=1, argument='foo')

# Instantiate a class implemented by the plugin
my_obj = awesome_module.AwesomeClass(another='bar', arg=False)
```

Similarly, you can import modules from plugin's subfolders. In this case:

```python
from itwinai.plugins.my_awesome_plugin.plugin_subfolder import another_module
# Call some function
another_module.another_function()

from itwinai.plugins.my_awesome_plugin.another_plugin_subfolder import yet_another_module

# Call some function
yet_another_module.yet_another_function()
```

## How to write a new plugin

To write a new plugin, you can take inspiration from
[this repository](https://github.com/interTwin-eu/itwinai-plugin-template),
and you can also use it as a template
when creating a new repository in GitHub.

> [!NOTE]
> Do NOT contribute directly to the plugin template repository,
> rather use it as a GitHub repository template, or fork it,
> or copy it.

The plugin template repository describes a python project using
a `pyproject.toml`, where the following configuration declares
that the Python package implemented by the plugin belongs under
the `itwinai.plugins.*` namespace:

```toml
[tool.setuptools.packages.find]

# Declare this package as part of the `itwinai.plugins` namespace
where = ["src"]

# List plugin folders and subfolders
include = [
    "itwinai.plugins.my_awesome_plugin",
    "itwinai.plugins.my_awesome_plugin.plugin_subfolder",
    "itwinai.plugins.my_awesome_plugin.another_plugin_subfolder",
]
```

> [!CAUTION]
> Make sure to list all the plugin subfolders in the `include` field
> of the `pyproject.toml` file,
> otherwise the plugin may not be installed correctly and plugin
> sub-packages may not be visble!

As a plugin developer, all you have to do is maintaining your Python
packages and modules under `src/itwinai/plugins`.

> [!WARNING]
> It is important to use an original name for the root package of your
> plugin (`my_awesome_plugin` in the example above). Avoid reusing existing
> plugin names, otherwise your plugin may be overridden by another one
> when installed at the same time!

Also, be minful of the Python
[packaging rules](https://packaging.python.org/en/latest/tutorials/packaging-projects/)!

> [!IMPORTANT]  
> Since your plugin is a Python library, remember that **each package
> and sub-package (i.e., folder and sub-folder) must contain at
> least one `__init__.py` file**, even
> if empty. Otherwise it will not be possible to correctly import
> the modules from the plugin.

Alternatively, instead of a `pyproject.toml`, you can as well use the
legacy `setup.py` to describe your plugin package. In that case, remember
to add the following arguments to the `setup` function, to ensure the correct
installation of the plugin under the `itwinai.plugins` namespace:

```python
from setuptools import setup, find_namespace_packages

setup(
    name='itwinai-awesome-plugin',
    ...
    packages=find_namespace_packages(where='src/', include=['itwinai.plugins.my_awesome_plugin']),
    package_dir={'': 'src'},
    install_requires=['itwinai'],  # Ensure `itwinai` is installed
    ...
)
```

### Docker containers

In this template repository you can already find two sample Dockerfiles as
example to build your containers using the itwinai container images as a base.

- `Dockerfile` provides an example of recipe that you would like to use to define
a general purpose container, to be executed on cloud or on HPC.
- `jupyter.Dockerfile` is an example of JupyterLab container based on the itwinai
one.

In this repository you can also find a `.dockerignore` file, which lists all the
files and directories to be ignored by docker build. Make sure to customize it
to your needs!

### Writing unit and integration tests

It is good practice to write tests for your code, and the itwinai plugins are not
an exception! Find some examples of unit tests under `tests/`, which can be
launched with `pytest` from the root of the repository with:

```bash
pytest -v tests/
```

Learn more about `pytest` [here](https://docs.pytest.org/en/stable/).

Notice that the tests will be executed automatically in the GitHub Actions space
every time you push to the main branch. You can change this behavior editing the
file at `.github/workflows/pytest.yaml`.

## Installing a plugin

An end user can install a plugin directly from its repository or from
PyPI (if available).

To use a plugin in your code, install it alongside itwinai:

```bash
# Install itwinai
pip install itwinai

# Now, install one or more plugins
pip install itwinai-awesome-plugin
# Or directly from the plugin's repository
pip install itwinai-awesome-plugin@git+https://github.com/USER/PROJECT@BRANCH
```

Conversely, a plugin developer may be interested in installing the plugin in
*editable mode* from the project directory:

```bash
git clone git@github.com:USER/REPOSITORY.git && cd REPOSITORY

# After having installed itwinai with its dependencies,
# install the plugin from source in editable mode.
pip install --editable . 
```

### Beware of installations in editable mode

When a package is installed in editable mode, it creates a symlink to the
source directory, which works differently from how non-editable packages are
installed. Namespace packages can behave inconsistently when some parts are
installed in editable mode and others are not.

To check whether a package was installed in editable mode you can use
`pip show PACKAGE_NAME`. Editable packages show an *"Editable"* path
in the output of the command.

> [!WARNING]
> When itwinai, the core library, is installed in editable mode **also**
> **the plugin must be installed in editable mode**. Otherwise, the plugin
> may not be visible when imported.

From empirical experience, the following installation configurations
*seem* to be working:

- itwinai: editable, plugin: editable
- itwinai: non-editable, plugin: non-editable
- itwinai: non-editable, plugin: editable
