#!/usr/bin/env bash

export PIP_REQUIRE_VIRTUALENV=true
PYTHON_INTERPRETER=$(which python)
DESTINATION="./python-virtualenv"

virtualenv --distribute --no-site-packages -p $PYTHON_INTERPRETER $DESTINATION

function requirements {
    grep -E '^[^ #]+' ../requirements.txt
}

if [ ! -d "$DESTINATION" ]; then
            mkdir $DESTINATION
    fi
    pushd $DESTINATION
    source bin/activate
    popd

    pushd $DESTINATION
    IFS=$'\n'
    for line in $(requirements); do
        cmd="pip install $line"
        echo "Running: $cmd"

        # Do the install for realz
        eval $cmd
        if [ "$?" != 0 ]; then
            echo "Incomplete installation of $line"
            exit $PIP_INSTALL_ERROR
            break
        fi
    done
    popd

    deactivate
