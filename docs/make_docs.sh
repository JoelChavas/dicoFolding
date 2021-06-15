#!/bin/bash

BUILDDIR=../../dicoFolding_docs

if [ ! -d $BUILDDIR/html ]; then
    cd source
    make firstbuild
    cd ..
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/host/usr/lib
export PYTHONPATH=$PYTHONPATH:/host/usr/lib/python3.6/lib-dynload:/usr/local/lib/python3.6

cd source
make buildandpush
