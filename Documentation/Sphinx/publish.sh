#!/bin/bash
#
# This script automates the task of building the documentation, moving it to
# gh-pages and pushing it to Github. It can be called as a standalone script
# or from Sphinx Makefile using `make gh-pages`

# Root directory of the repository
GIT_ROOT=`git rev-parse --show-toplevel`

# Message of the last commit, used to commit on gh-pages
GIT_LAST_COMMIT=`git log master -1 --oneline`

# Move to the repository root and checkouth gh-pages (create it if it does not
# exist)
if cd $GIT_ROOT && (git checkout gh-pages || git checkout --orphan gh-pages) ;
then
    # Start with a clean directory
    rm -rf *

    # Retrieve everything important from the master branch
    git checkout master Documentation Software .gitignore

    # Move to the Sphinx directory and build the documentation
    if cd Documentation/Sphinx && make html ;
    then
        # Move the Sphinx output back to the root
        cd $GIT_ROOT
        mv -f Documentation/Sphinx/build/html/* ./

        # Keep only the sphinx output
        rm -rf Documentation/ Software/

        # Add everything, commit and push to Github
        git add -A && git commit -am "Pushing to gh-pages: $GIT_LAST_COMMIT" && git push origin gh-pages
    fi

    # Checkout again to master branch
    git checkout master
fi
