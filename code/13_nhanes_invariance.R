# Stage 6a: NHANES 2021-2023 replication of the survey-weighted multi-group ordinal CFA (WLSMV, theta)
# Each disability domain vs adults without disability, NHANES (top two response categories merged).
# Sequence (Wu & Estabrook 2016): configural -> thresholds -> thresholds+loadings -> +intercepts
# Saves fit indices and group-specific parameters of the configural and full scalar models.
suppressMessages({library(lavaan); library(semTools); library(jsonlite)})
d <- read.csv("data/proc/nhanes_analytic.csv")
d <- d[d$split == "train", ]
scales <- list(
  phq = paste0("phq8", 1:8))
d$d_any <- d$disab
domains <- c("any","mobility","cognition")
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
    for (it in items) dd[[it]] <- ordered(pmin(dd[[it]], 2), levels = 0:2)  # top two categories merged (sparse cells)
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
      vc <- try(vcov(fit),silent=TRUE)
      se_ok <- !inherits(vc,"try-error") && all(is.finite(vc)) && all(diag(vc)>0)
      admissible <- se_ok && !inherits(estcheck,"try-error") && all(vapply(estcheck, function(g) all(is.finite(g$theta)) && all(diag(g$theta)>0) && all(is.finite(g$psi)) && all(diag(g$psi)>0), logical(1)))
      diag_out[[length(diag_out)+1]] <- data.frame(scale=sc,domain=dm,step=st,converged=converged,admissible=admissible,standard_errors_available=se_ok)
      if (!converged || !admissible) { cat(sc,dm,st,"EXCLUDED nonconvergent or inadmissible\n"); next }
      fm <- try(fitMeasures(fit, c("chisq.scaled","df.scaled","cfi.scaled","tli.scaled","rmsea.scaled","srmr")), silent = TRUE)
      if (inherits(fm, "try-error")) { cat(sc, dm, st, "fit measures unavailable\n"); next }
      fits_out[[length(fits_out) + 1]] <- data.frame(scale = sc, domain = dm, step = st,
        n_domain = sum(dd$grp == "domain"), n_ref = sum(dd$grp == "nodisab"), as.list(round(fm, 4)))
      if (st %in% c("configural", "scalar", "strict")) pars_out[[paste(sc, dm, st, sep = "|")]] <- grab(fit)
      cat(sc, dm, st, "CFI", round(fm["cfi.scaled"], 4), "RMSEA", round(fm["rmsea.scaled"], 4), "\n")
    }
  }
}
fits <- do.call(rbind, fits_out)
write.csv(fits, "out/nhanes_invariance_fit.csv", row.names = FALSE)
write_json(pars_out, "out/nhanes_invariance_params.json", digits = 8, auto_unbox = TRUE)

write.csv(do.call(rbind,diag_out), "out/nhanes_invariance_diagnostics.csv", row.names=FALSE)
writeLines(capture.output(sessionInfo()), "out/nhanes_R_session_info.txt")
