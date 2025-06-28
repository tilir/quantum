import os
import sys
sys.path.insert(0, os.path.abspath('..'))

extensions = [
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'autoapi.extension'
]

# AutoAPI setting
autoapi_type = 'python'
autoapi_dirs = ['../experiments']
autoapi_add_toctree_entry = True
autoapi_keep_files = True
autoapi_generate_api_docs = True
templates_path = ['_templates']
exclude_patterns = []

autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
]

# Setting for Google Style
napoleon_google_docstring = True
napoleon_include_init_with_doc = True

# Theme
html_theme = 'sphinx_rtd_theme'
