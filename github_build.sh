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
setopt rmstarsilent
rm -rf *
mv ~/Desktop/_build/* .
mv ~/Desktop/_build/.* .
rmdir ~/Desktop/_build

# commit to gh-build
touch .nojekyll
git add .
git commit -m "site updated at $(date)"
git push origin gh-pages --force
git checkout master

# remove all untracked files
git clean -f
