def eo_thresholds(s, yy, ww, g, eps):
    rocs = {gg: roc_pts(s[g == gg], yy[g == gg], ww[g == gg]) for gg in GROUPS}
    best, best_thr = -1, None
    for T in np.linspace(0.3, 0.95, 66):
        for F in np.linspace(0.01, 0.45, 89):
            thr, tprs, fprs, wt_pos, wt_neg = {}, {}, {}, {}, {}
            for gg, (ss, tp, fp) in rocs.items():
                i = np.argmin(np.abs(tp - T) + np.abs(fp - F))
                thr[gg], tprs[gg], fprs[gg] = ss[i], tp[i], fp[i]
                m = g == gg
                wt_pos[gg], wt_neg[gg] = (ww * yy)[m].sum(), (ww * (1 - yy))[m].sum()
            aod = max(0.5 * (abs(tprs[gg] - tprs["none"]) + abs(fprs[gg] - fprs["none"])) for gg in GROUPS if gg != "none")
            if aod > eps:
                continue
            tpr = sum(tprs[gg] * wt_pos[gg] for gg in GROUPS) / sum(wt_pos.values())
            fpr = sum(fprs[gg] * wt_neg[gg] for gg in GROUPS) / sum(wt_neg.values())
            if (tpr + 1 - fpr) / 2 > best:
                best, best_thr = (tpr + 1 - fpr) / 2, thr
    return best_thr