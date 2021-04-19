
dicoFolding
------------

The project aims to study cortical folding patterns thanks to deep learning tools.

Development
-----------

.. code-block:: shell

    git clone https://github.com/JoelChavas/dicoFolding.git

    # Install for development
    bv bash
    cd dicoFolding
    virtualenv -p /casa/install/bin/python --system-site-packages venv
    . bin/activate/venv
    pip install -e .

    # Tests
    python -m pytest  # run tests



If you want to install the package:

.. code-block:: shell

    python setup.py install

Notebooks are in the repertory notebooks, access using:

.. code-block:: shell

    bv bash # to enter brainvisa environnment
    jupyter notebook # then click on file to open a notebook

