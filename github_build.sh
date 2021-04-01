# checkout master branch
git checkout master

# force clean build
rm -rf doc/_build

# build site
sphinx-build -b html doc doc/_build

# move site to tmp
mv doc/_build ~/Desktop/

# checkout gh-pages and replace content
git checkout -B gh-pages
rm -rf *
mv ~/Desktop/_site/* .
rmdir ~/Desktop/_site

# commit to gh-build
git add .
git commit -m "site updated at $(date)"
git push origin gh-build --force
git checkout master

# remove all untracked files
git clean -f
