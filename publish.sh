#!/bin/sh
# Script to publish to private GH-pages
make cleandoctrees html pdf
ghp-import --no-jekyll -r origin --push --force build/html


