# Stage 2a: survey-weighted multi-group ordinal CFA (WLSMV, theta parameterisation)
# Each disability domain vs adults without disability, NHIS 2019+2022 (training years).
# Sequence (Wu & Estabrook 2016): configural -> thresholds -> thresholds+loadings -> +intercepts
# Saves fit indices and group-specific parameters of the configural and full scalar models.
suppressMessages({library(lavaan); library(semTools); library(jsonlite)})
d <- read.csv("data/proc/nhis_analytic.csv")
d <- d[d$split == "train", ]
scales <- list(
  phq = paste0("phq8", 1:8),
  gad = paste0("gad7", 1:7))
domains <- c("vision","hearing","mobility","cognition","selfcare","communication")
steps <- list(configural = "", thresholds = "thresholds",
              thr_load = c("thresholds","loadings"),
              scalar = c("thresholds","loadings","intercepts"),
              strict = c("thresholds","loadings","intercepts","residuals"))
fits_out <- list(); pars_out <- list(); diag_out <- list()

grab <- function(fit) {
  est <- lavInspect(fit, "est")
  lapply(est, function(g) list(lambda = as.numeric(g$lambda), tau = as.numeric(g$tau),
                               theta = as.numeric(diag(g$theta)), psi = as.numeric(g$psi),
                               alpha = as.numeric(g$alpha), nu = as.numeric(g$nu)))
}

for (sc in names(scales)) {
  items <- scales[[sc]]
  for (dm in domains) {
    dd <- d[(d[[paste0("d_", dm)]] %in% 1) | (d$disab %in% 0), c(items, "wt_pooled", paste0("d_", dm))]
    dd$grp <- ifelse(dd[[paste0("d_", dm)]] %in% 1, "domain", "nodisab")
    dd <- dd[complete.cases(dd[, items]), ]
    dd$w <- dd$wt_pooled / mean(dd$wt_pooled)
    for (it in items) dd[[it]] <- ordered(dd[[it]], levels = 0:3)
    mod <- paste("F =~", paste(items, collapse = " + "))
    for (st in names(steps)) {
      syn <- as.character(measEq.syntax(configural.model = mod, data = dd, ordered = items,
                                        parameterization = "theta", ID.fac = "std.lv",
                                        ID.cat = "Wu.Estabrook.2016", group = "grp",
                                        group.equal = steps[[st]]))
      fit <- try(cfa(syn, data = dd, ordered = items, group = "grp", estimator = "WLSMV",
                     parameterization = "theta", sampling.weights = "w"), silent = TRUE)
      if (inherits(fit, "try-error")) { cat(sc, dm, st, "FAILED\n"); next }
      converged <- isTRUE(lavInspect(fit, "converged"))
      estcheck <- try(lavInspect(fit, "est"), silent=TRUE)
      admissible <- !inherits(estcheck,"try-error") && all(vapply(estcheck, function(g) all(is.finite(g$theta)) && all(diag(g$theta)>0) && all(is.finite(g$psi)) && all(diag(g$psi)>0), logical(1)))
      diag_out[[length(diag_out)+1]] <- data.frame(scale=sc,domain=dm,step=st,converged=converged,admissible=admissible)
      if (!converged || !admissible) { cat(sc,dm,st,"EXCLUDED nonconvergent or inadmissible\n"); next }
      fm <- fitMeasures(fit, c("chisq.scaled","df.scaled","cfi.scaled","tli.scaled","rmsea.scaled","srmr"))
      fits_out[[length(fits_out) + 1]] <- data.frame(scale = sc, domain = dm, step = st,
        n_domain = sum(dd$grp == "domain"), n_ref = sum(dd$grp == "nodisab"), as.list(round(fm, 4)))
      if (st %in% c("configural", "scalar", "strict")) pars_out[[paste(sc, dm, st, sep = "|")]] <- grab(fit)
      cat(sc, dm, st, "CFI", round(fm["cfi.scaled"], 4), "RMSEA", round(fm["rmsea.scaled"], 4), "\n")
    }
  }
}
fits <- do.call(rbind, fits_out)
write.csv(fits, "out/invariance_fit.csv", row.names = FALSE)
write_json(pars_out, "out/invariance_params.json", digits = 8, auto_unbox = TRUE)

write.csv(do.call(rbind,diag_out), "out/invariance_diagnostics.csv", row.names=FALSE)
writeLines(capture.output(sessionInfo()), "out/R_session_info.txt")
