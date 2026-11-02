lib <- Sys.getenv("R_LIBS_USER")
if (!nzchar(lib)) lib <- "r-library"
dir.create(lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(lib,.libPaths()))
options(timeout=300)
install.packages(c("mnormt","pbivnorm","numDeriv","quadprog","jsonlite","lavaan","semTools"), repos="https://packagemanager.posit.co/cran/2024-03-15",type="binary",lib=lib)
for (p in c("lavaan","semTools","jsonlite")) cat(p,as.character(packageVersion(p)),"\n")
stopifnot(as.character(packageVersion("lavaan"))=="0.6.17",as.character(packageVersion("semTools"))=="0.5.6",as.character(packageVersion("jsonlite"))=="1.8.8")
