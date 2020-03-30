# Source code and website for "Quality Diversity for Multitask Optimization"

# Reproducing the experiments

## Website

This repo contains the source for the article.

### If all you want to do is view the page locally

Run `python -m http.server` to serve on the base directory to view `index.html` in a local browser.

### Article

`draft.md` - main text of the article, in markdown.

`draft_appendix.md` - appendix, in markdown.

`draft_bib.html` - the citations.

`draft_header.html` - start of the document

`index.html` - generated, don't edit this file.

### Instructions to Build and Test
```bash
git clone https://github.com/weightagnostic/2020_mouret_gecco
cd 2020_mouret_gecco
npm install
```

Modify text by editing `index.md` -- this is where all of the content exists.

Appendix content goes in `draft_appendix.md`. Add bib entries to `draft_bib.html`.

Run `./bin/make` to build document into `index.html` (which are identical).
Run `python -m http.server` to serve on the base directory to view `index.html` in a local browser for debugging.

To watch all markdown files for changes and then compile them, you can run the following
```
brew install fswatch
./bin/watch
```
