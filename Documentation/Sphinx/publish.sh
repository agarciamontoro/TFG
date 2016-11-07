#!/bin/bash

GIT_ROOT=`git rev-parse --show-toplevel`
GIT_LAST_COMMIT=`git log master -1 --oneline`

# make html
# git add -f build/html/*
# git stash
#
# if git checkout gh-pages;
# then
#     rm -rf _sources _static _modules
#     git stash pop
#
#     cd $GIT_ROOT
#     mv -fv Documentation/Sphinx/build/html/* ./
#     rm -rf Documentation/ Software/
#
#     echo $GIT_LAST_COMMIT
#
#     # git add -A
#     # git commit -m "Generated gh-pages for $GIT_LAST_COMMIT"
#     # git push origin gh-pages
#     # git checkout master
# else
#     git stash drop
# fi

if git checkout gh-pages || git checkout --orphan gh-pages ;
then
    rm -rf *
    git checkout master *
    if cd Documentation/Sphinx && make html ;
    then
        cd $GIT_ROOT
        mv -f Documentation/Sphinx/build/html/* ./
        rm -rf Documentation/ Software/
    fi
    git add -A && git commit -am "Pushing to gh-pages: $GIT_LAST_COMMIT"
    git checkout master
fi
