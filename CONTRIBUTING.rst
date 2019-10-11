Contributing to SPORCO
======================

Contributions to SPORCO are welcome. Before starting work, please contact the maintainers, either via email or the GitHub issue system, to discuss the relevance of your contribution and the most appropriate location within the existing package structure.

Please follow these guidelines in developing code intended for SPORCO:

• Set up your editor to remove trailing whitespace.
• Follow `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`__ coding
  conventions, with minor deviations as present in the existing code in
  SPORCO. `Flake8 <http://flake8.pycqa.org/en/latest/>`__ is a useful tool
  for checking coding style; to check ``module.py`` do::

    flake8 --ignore=D202,E303,D413,D402 module.py
• All code must be thoroughly documented, adequately commented, and have
  corresponding `pytest <https://docs.pytest.org/>`__ unit tests.


Guides for Other Projects
-------------------------

Some other open source projects have detailed contributing guides that are worth consulting for recommended good practices, technical details of git usage, etc.:

• `scikit-image <http://scikit-image.org/docs/dev/contribute.html>`__
• `OpenStack <https://docs.openstack.org/hacking/latest/user/hacking.html>`__
