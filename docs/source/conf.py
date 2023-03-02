import os
import sys
import sphinx_rtd_theme
from recommonmark.parser import CommonMarkParser

sys.path.insert(0, os.path.abspath(__file__+'../../../..'))

project = 'scButterfly'
copyright = '2023, BioX-NKU'
author = 'BioX-NKU'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'nbsphinx'
]

autosummary_generate = True
autodoc_member_order = 'bysource'

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_custom_sections = [('Params', 'Parameters')]
todo_include_todos = False

templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints']


html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = dict(navigation_depth=4, logo_only=True)
html_context = dict(
    display_github=True,
    github_user='BioX-NKU',
    github_repo='scButterfly',
    github_version='main',
    conf_py_path='/docs/',
)
html_static_path = ['_static']
html_show_sphinx = False