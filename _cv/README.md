# Public academic CV

This website edition is based on the supplied faculty-application CV. It omits
the application reference section and Interfolio addresses. Bibliographic dates
and publication status were reconciled with public publication records. The
original faculty-application source is maintained separately.

To update the download, edit `public-cv.tex` and `publications.tex`, run
`pdflatex public-cv.tex` twice from this directory, visually inspect every page,
then copy `public-cv.pdf` to `../assets/resume.pdf`. Do not commit TeX build files.
The `_cv` directory is excluded from the generated site by Jekyll's normal
underscore-directory convention.
