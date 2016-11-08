#!/bin/bash

GIT_ROOT=`git rev-parse --show-toplevel`
GIT_LAST_COMMIT=`git log master -1 --oneline`

if cd $GIT_ROOT && (git checkout gh-pages || git checkout --orphan gh-pages) ;
then
    rm -rf *
    git checkout master Documentation Software
    if cd Documentation/Sphinx && make html ;
    then
        cd $GIT_ROOT
        ls -R Documentation/Sphinx/build/html/_static
        mv -vf Documentation/Sphinx/build/html/* ./
        rm -rf Documentation/ Software/
    fi
    git add -A && git commit -am "Pushing to gh-pages: $GIT_LAST_COMMIT" && git push origin gh-pages
    git checkout master
fi
