"""Paper assets built only from the result files (Prompt 15): number macros, LaTeX tables, figure copies.

Usage (from repo/):
  python -X utf8 scripts/make_paper_assets.py --results results --paper paper
Writes <paper>/generated/numbers.tex (one \\newcommand per number used in the text), <paper>/generated/tab_*.tex
(tabular bodies, booktabs) and copies <results>/figures/*.pdf to <paper>/figures/. Never edit these by hand.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = lambda x: f"{100 * x:.1f}"                    # proportion -> percent, 1 decimal
D3 = lambda x: f"{x:.3f}"


def pfmt(p: float) -> str:
    return f"{p:.2f}" if p >= 0.1 else f"{p:.3f}" if p >= 0.001 else "<0.001"


def minus(s) -> str:
    """A leading minus on a number becomes a typeset minus sign."""
    return re.sub(r"(?<![\w$-])-(?=\d)", "$-$", str(s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--paper", default=str(ROOT / "paper"))
    a = ap.parse_args()
    res, paper = Path(a.results), Path(a.paper)
    r = pd.read_csv(res / "results.csv")
    ex = pd.read_csv(res / "results_v2_exploratory.csv")
    ep = pd.read_csv(res / "error_patterns.csv")
    audit = json.loads((res / "audit_report.json").read_text(encoding="utf-8"))
    t = pd.read_csv(res / "item_level.csv")

    def one(frame, **kw):
        hit = frame
        for k, v in kw.items():
            hit = hit[hit[k] == v]
        if len(hit) != 1:
            raise ValueError(f"expected one row for {kw}, found {len(hit)}")
        return hit.iloc[0]

    R = lambda m, s="all": one(r, metric=m, subset=s)
    X = lambda an, m: one(ex, analysis=an, metric=m)
    E = lambda g, m: one(ep, group=g, metric=m)

    def mcn(note):
        m = re.search(r"p=([0-9.eE-]+); (?:discordant )?b=(\d+).*c=(\d+)", note)
        return float(m.group(1)), int(m.group(2)), int(m.group(3))

    macros = {}

    def put(name, value):
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"bad macro name {name}")
        macros[name] = value

    # Items and audit.
    n = int(R("accuracy_greedy").n)
    put("nTest", n)
    put("nM", int((t["code"] == "M").sum()))
    put("nO", int((t["code"] == "O").sum()))
    put("nObjective", int(t["objective_subset"].astype(bool).sum()))
    put("auditPassed", audit["checks_passed"])
    put("auditTotal", audit["checks_total"])

    # RQ1 and sensitivity.
    for tag, subset in (("Main", "all"), ("Lenient", "all (lenient)"), ("Objective", "objective_subset"),
                        ("M", "subgroup_M"), ("O", "subgroup_O")):
        g, v, d = R("accuracy_greedy", subset), R("accuracy_majority10", subset), R("accuracy_difference_majority10_minus_greedy", subset)
        p, b, c = mcn(d.note)
        put(f"accGreedy{tag}", P(g.value)); put(f"accGreedy{tag}Lo", P(g.ci_low)); put(f"accGreedy{tag}Hi", P(g.ci_high))
        put(f"accVote{tag}", P(v.value)); put(f"accVote{tag}Lo", P(v.ci_low)); put(f"accVote{tag}Hi", P(v.ci_high))
        put(f"diff{tag}", P(d.value)); put(f"diff{tag}Lo", P(d.ci_low)); put(f"diff{tag}Hi", P(d.ci_high))
        put(f"p{tag}", pfmt(p)); put(f"b{tag}", b); put(f"c{tag}", c); put(f"n{tag}Analysis", int(d.n))
    s = R("accuracy_single_sample_mean")
    put("accSingle", P(s.value))
    sd, lo, hi = map(float, re.search(r"= ([0-9.]+); min ([0-9.]+), max ([0-9.]+)", s.note).groups())
    put("accSingleSD", f"{100 * sd:.2f}"); put("accSingleMin", P(lo)); put("accSingleMax", P(hi))
    for k, word in ((1, "One"), (3, "Three"), (5, "Five"), (7, "Seven"), (9, "Nine"), (10, "Ten")):
        put(f"accK{word}", P(R(f"accuracy_majority_of_first_{k}").value))

    # RQ2 and RQ3.
    for name, metric, f in (("unanimity", "unanimity_rate", P), ("modalShare", "mean_modal_share", D3),
                            ("kappaRuns", "fleiss_kappa_across_runs", D3), ("flipRate", "pairwise_flip_rate", P),
                            ("auroc", "auroc_modal_share_vs_majority_correct", D3)):
        x = R(metric)
        put(name, f(x.value)); put(f"{name}Lo", f(x.ci_low)); put(f"{name}Hi", f(x.ci_high))
    put("accGeNine", P(R("accuracy_agreement_ge9").value)); put("nGeNine", int(R("accuracy_agreement_ge9").n))
    put("accLtNine", P(R("accuracy_agreement_lt9").value)); put("nLtNine", int(R("accuracy_agreement_lt9").n))
    cov10 = R("coverage_at_agreement_ge10")
    put("coverageTen", P(cov10.value))
    m10 = re.search(r"= ([0-9.]+)", cov10.note) if isinstance(cov10.note, str) else None   # no note if no item is 10/10
    put("accTen", P(float(m10.group(1))) if m10 else "n/a")

    # Output quality and cost.
    put("unpGreedyPct", P(R("unparseable_rate_greedy").value))
    put("unpSampledPct", P(R("unparseable_rate_sampled").value))
    put("truncGreedyPct", P(R("truncation_rate_greedy").value))
    put("truncSampledPct", P(R("truncation_rate_sampled").value))
    put("ties", int(R("vote_ties").value))
    for word, arm in (("Greedy", "greedy"), ("Sampled", "sampled")):
        a_ = audit["per_arm"][arm]
        put(f"unp{word}N", a_["unparseable"]); put(f"unp{word}TruncN", a_["unparseable_truncated"])
        put(f"runs{word}N", f"{a_['records']:,}".replace(",", "{,}"))
    gt, st = R("output_tokens_total_greedy").value, R("output_tokens_total_sampled").value
    gs, ss = R("gpu_seconds_greedy").value, R("gpu_seconds_sampled").value
    put("tokGreedyItem", f"{gt / n:.0f}"); put("tokVoteItem", f"{st / n:,.0f}".replace(",", "{,}"))
    put("tokRatio", f"{st / gt:.1f}"); put("gpuGreedy", f"{gs:.0f}"); put("gpuVote", f"{ss:,.0f}".replace(",", "{,}"))
    put("gpuRatio", f"{ss / gs:.1f}")
    put("rerunFiftySame", f"{round(R('greedy_rerun_identical_letter', 'rerun_subset').value * 50)}")

    # Exploratory (protocol v2).
    put("rerunSamePct", P(X("greedy_self_agreement", "same_letter_rate").value))
    put("rerunSame", round(X("greedy_self_agreement", "same_letter_rate").value * n))
    rd = X("greedy_self_agreement", "accuracy_difference_rerun_minus_greedy")
    put("rerunDiff", P(rd.value)); put("rerunDiffLo", P(rd.ci_low)); put("rerunDiffHi", P(rd.ci_high)); put("rerunP", pfmt(mcn(rd.note)[0]))
    put("accRerun", P(X("greedy_self_agreement", "accuracy_greedy_rerun_full").value))
    for word, temp in (("Low", "0.3"), ("High", "1.0")):
        an = f"temperature_{temp}"
        d = X(an, "accuracy_difference_majority10_minus_greedy")
        put(f"diffT{word}", P(d.value)); put(f"diffT{word}Lo", P(d.ci_low)); put(f"diffT{word}Hi", P(d.ci_high))
        put(f"pT{word}", pfmt(mcn(d.note)[0]))
        put(f"accVoteT{word}", P(X(an, "accuracy_majority10").value))
        put(f"unanimityT{word}", P(X(an, "unanimity_rate").value))
        put(f"aurocT{word}", D3(X(an, "auroc_modal_share_vs_majority_correct").value))
    pr = X("contamination_probe", "exact_match_rate")
    put("probeHits", round(pr.value * pr.n)); put("probeN", int(pr.n)); put("probeLo", P(pr.ci_low)); put("probeHi", P(pr.ci_high))

    # Error patterns.
    put("bGreedyLetterRuns", f"{E('b: vote right, greedy wrong', 'mean sampled runs giving greedy' + chr(39) + 's letter (of 10)').value:.1f}")
    put("bKeyRuns", f"{E('b: vote right, greedy wrong', 'mean sampled runs giving the key (of 10)').value:.1f}")
    put("cKeyRuns", f"{E('c: greedy right, vote wrong', 'mean sampled runs giving the key (of 10)').value:.1f}")
    put("tiesB", int(E("b: vote right, greedy wrong", "votes decided by the tie rule").value))
    put("tiesC", int(E("c: greedy right, vote wrong", "votes decided by the tie rule").value))
    put("bGreedyUnparseable", int(E("b: vote right, greedy wrong", "greedy unparseable").value))
    put("allWrong", int(E("wrong in all 11", "items").value))
    put("allWrongSame", int(E("wrong in all 11", "same wrong letter in all 11 generations").value))
    put("allWrongMixed", int(E("wrong in all 11", "two or more different wrong letters").value))
    put("allWrongUnparseable", macros["allWrong"] - macros["allWrongSame"] - macros["allWrongMixed"])
    uw, un = E("10/10 unanimous", "of which wrong"), E("10/10 unanimous", "items")
    put("unanimousWrong", int(uw.value)); put("unanimousN", int(un.value)); put("unanimousWrongPct", P(uw.value / un.value))
    for word, temp in (("Low", "0.3"), ("Mid", "0.7"), ("High", "1.0")):
        put(f"stableT{word}", P(E(f"T = {temp}", "share of items with >= 9/10 runs agreeing").value))
    vr = E("vote vs greedy rerun", "accuracy difference")
    lo, hi = map(float, re.search(r"\[([-0-9.]+), ([-0-9.]+)\]", vr.note).groups())
    put("voteVsRerun", P(vr.value)); put("voteVsRerunLo", P(lo)); put("voteVsRerunHi", P(hi))
    put("voteVsRerunP", pfmt(float(re.search(r"p=([0-9.eE-]+)", vr.note).group(1))))

    # Post hoc analyses requested by the peer review (scripts/posthoc_review.py); labelled post hoc in the paper.
    ph = pd.read_csv(res / "posthoc_review.csv") if (res / "posthoc_review.csv").exists() else None
    if ph is not None:
        H = lambda grp, m: one(ph, group=grp, metric=m)
        for name, metric in (("decompSingle", "single_run_mean_minus_greedy"), ("decompVote", "vote_minus_single_run_mean")):
            x = H("decomposition", metric)
            put(name, P(x.value)); put(f"{name}Lo", P(x.ci_low)); put(f"{name}Hi", P(x.ci_high))
        x = H("decomposition", "vote_minus_each_run_min_to_max")
        put("voteEachMin", P(x.ci_low)); put("voteEachMax", P(x.ci_high))
        bc = re.search(r"b (\d+)-(\d+), c (\d+)-(\d+), McNemar p ([0-9.]+)-([0-9.]+)", x.note).groups()
        put("voteEachB", f"{bc[0]}--{bc[1]}"); put("voteEachC", f"{bc[2]}--{bc[3]}")
        put("voteEachP", f"{pfmt(float(bc[4]))}--{pfmt(float(bc[5]))}")
        put("discordancePct", P(H("decomposition", "observed_discordance_rate").value))
        for o in ("train", "dev", "test"):
            w = o.capitalize()
            d = H(f"origin_{o}", "difference")
            pb = re.search(r"b=(\d+), c=(\d+), McNemar p=([0-9.eE-]+)", d.note).groups()
            put(f"nOrigin{w}", int(H(f"origin_{o}", "n").value))
            put(f"accGreedyOrigin{w}", P(H(f"origin_{o}", "accuracy_greedy").value))
            put(f"accVoteOrigin{w}", P(H(f"origin_{o}", "accuracy_vote").value))
            put(f"diffOrigin{w}", P(d.value)); put(f"diffOrigin{w}Lo", P(d.ci_low)); put(f"diffOrigin{w}Hi", P(d.ci_high))
            put(f"bOrigin{w}", pb[0]); put(f"cOrigin{w}", pb[1]); put(f"pOrigin{w}", pfmt(float(pb[2])))
        for name, metric in (("unanDiffLowMid", "unanimity_T0.3_minus_T0.7"), ("unanDiffHighMid", "unanimity_T1.0_minus_T0.7"),
                             ("unanDiffHighLow", "unanimity_T1.0_minus_T0.3")):
            x = H("temperature", metric)
            put(name, P(x.value)); put(f"{name}Lo", P(x.ci_low)); put(f"{name}Hi", P(x.ci_high))
        for name, metric in (("aurocFirstThree", "modal_share_first_3_runs_vs_vote_correct"),
                             ("aurocGreedyAgree", "greedy_equals_vote_vs_vote_correct"),
                             ("aurocGreedyCorrect", "modal_share_10_runs_vs_greedy_correct")):
            x = H("rq3_comparison", metric)
            put(name, D3(x.value)); put(f"{name}Lo", D3(x.ci_low)); put(f"{name}Hi", D3(x.ci_high))
        for arm, w in (("greedy", "Greedy"), ("sampled", "Sampled")):
            x = H("position", f"{arm}_wrong_chose_A")
            put(f"wrongA{w}", int(x.value)); put(f"wrongA{w}Exp", re.search(r"expected ([0-9.]+)", x.note).group(1))
            put(f"wrong{w}N", f"{int(x.n):,}".replace(",", "{,}"))
        put("wrongLettersP", pfmt(H("position", "greedy_wrong_letters_chi2_p").value))
        for arm, w in (("greedy", "Greedy"), ("vote", "Vote")):
            put(f"acc{w}KeyA", P(H("position", f"accuracy_{arm}_key_A").value))
            others = [H("position", f"accuracy_{arm}_key_{x}").value for x in "BCD"]
            put(f"acc{w}KeyOtherMin", P(min(others))); put(f"acc{w}KeyOtherMax", P(max(others)))
        for k, w in ((1, "One"), (3, "Three"), (5, "Five"), (10, "Ten")):
            put(f"accKOrd{w}", P(H("majority_of_k_orders", f"k{k}").value))
        for w, m in (("Short", "hits_short_1_2_words"), ("Long", "hits_long_3plus_words")):
            x = H("probe", m)
            put(f"probe{w}Hits", int(x.value)); put(f"probe{w}N", int(x.n))
        x = H("error_analysis", "b_items_greedy_letter_runs_parsed_only")
        put("bGreedyLetterRunsParsed", f"{x.value:.1f}"); put("bParsedN", int(x.n))
    put("pairAgree", P(1 - R("pairwise_flip_rate").value))
    for word, arm in (("Greedy", "greedy"), ("Sampled", "sampled")):
        put(f"unp{word}NotTruncN", audit["per_arm"][arm]["unparseable_not_truncated"])

    # Data pipeline: MedQA size, keyword pool, screening codes and agreement, split (committed files).
    sys.path.insert(0, str(ROOT / "src"))
    from board_vote.data import load_medqa
    raw_dir = ROOT / "data" / "raw" / "medqa_4opt_hf"
    if not raw_dir.exists():
        raise SystemExit(f"MedQA files missing in {raw_dir}; run scripts/fetch_medqa.py first")
    put("medqaN", f"{len(load_medqa(raw_dir)):,}".replace(",", "{,}"))
    scr = ROOT / "data" / "screening"
    codes = pd.read_csv(scr / "codes_adjudicated.csv")
    put("poolN", f"{len(codes):,}".replace(",", "{,}"))
    for code in "MOX":
        put(f"code{code}", int((codes["code"] == code).sum()))
    man = json.loads((ROOT / "data" / "splits" / "SPLIT_MANIFEST.json").read_text(encoding="utf-8"))
    put("eligibleN", man["eligible_before_exclusions"]); put("excludedN", man["excluded_image_or_bare_letter"])
    put("finalN", man["final_items"]); put("debugN", man["splits"]["debug"]["n"]); put("debugM", man["splits"]["debug"]["M"])
    first = pd.read_csv(scr / "ai_codes_final.csv").set_index("id")["code"]
    second = pd.read_csv(scr / "second_screener_codes.csv").set_index("id")["code"]
    a1 = first.loc[second.index]
    po = (a1 == second).mean()
    pe = sum((a1 == k).mean() * (second == k).mean() for k in set(a1) | set(second))
    put("screenN", len(second)); put("screenAgree", int((a1 == second).sum())); put("screenKappa", D3((po - pe) / (1 - pe)))
    adj = pd.read_csv(scr / "adjudication_log.csv")
    put("adjudicatedN", len(adj)); put("adjudicatedChanged", int((adj["final"] != adj["first"]).sum()))
    put("screenChangedPct", P((adj["final"] != adj["first"]).sum() / len(second)))
    put("unsampledN", len(codes) - len(second))

    # Run times: generation time from the Kaggle run summaries; whole-session time from the Kaggle logs (local only).
    # The Kaggle logs are local only; their session times are kept in results/env/session_hours.json (committed),
    # refreshed here whenever the logs are present.
    env = res / "env"
    hours_file = env / "session_hours.json"
    hours = json.loads(hours_file.read_text(encoding="utf-8")) if hours_file.exists() else {}
    for d in sorted(p.name for p in env.glob("kaggle_*") if p.is_dir()):
        log = next((env / d).glob("*.log"), None)
        if log:
            hours[d] = round(json.loads(log.read_text(encoding="utf-8"))[-1]["time"] / 3600, 3)
    if hours and env.exists():
        hours_file.write_text(json.dumps(dict(sorted(hours.items())), indent=1) + "\n", encoding="utf-8")
    for word, d, f in (("Main", "kaggle_03_test_v1", "run_summary.json"), ("Extras", "kaggle_04_extras_v1", "run_summary_v2.json")):
        summ = env / d / f
        put(f"genHours{word}", f"{json.loads(summ.read_text(encoding='utf-8'))['total_seconds'] / 3600:.2f}"
            if summ.exists() else "n/a")
        put(f"sessionHours{word}", f"{hours[d]:.2f}" if d in hours else "n/a")
    if hours:
        put("sessionHoursTotal", f"{sum(hours.values()):.2f}")

    gen = paper / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated by scripts/make_paper_assets.py from results/*.csv -- do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{minus(v)}}}" for k, v in macros.items()]
    (gen / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Tables (tabular bodies only; captions live in the section files).
    def ci(x, lo, hi):
        return f"{x} ({lo}, {hi})"

    rows = []
    for tag, label in (("Main", "All items (confirmatory)"), ("Lenient", "Lenient parsing"), ("Objective", "Objective subset"),
                       ("M", "Management (M)"), ("O", "Other oncology (O)")):
        m = lambda k: macros[k]
        rows.append(f"{label} & {m(f'n{tag}Analysis')} & {ci(m(f'accGreedy{tag}'), m(f'accGreedy{tag}Lo'), m(f'accGreedy{tag}Hi'))} & "
                    f"{ci(m(f'accVote{tag}'), m(f'accVote{tag}Lo'), m(f'accVote{tag}Hi'))} & "
                    f"{ci(m(f'diff{tag}'), m(f'diff{tag}Lo'), m(f'diff{tag}Hi'))} & {m(f'b{tag}')}/{m(f'c{tag}')} & {m(f'p{tag}')} \\\\")
    rows = [minus(r_) for r_ in rows]
    (gen / "tab_rq1.tex").write_text("\n".join([
        r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
        r"Analysis & $n$ & Greedy \% (95\% CI) & Majority-10 \% (95\% CI) & $\Delta$ points (95\% CI) & $b$/$c$ & $p$ \\",
        r"\midrule", rows[0], r"\midrule", *rows[1:], r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf-8")

    m = macros
    cons = [("Unanimity, 10/10 runs (\\%)", ci(m["unanimity"], m["unanimityLo"], m["unanimityHi"])),
            ("Mean modal share", ci(m["modalShare"], m["modalShareLo"], m["modalShareHi"])),
            ("Fleiss' $\\kappa$ across runs", ci(m["kappaRuns"], m["kappaRunsLo"], m["kappaRunsHi"])),
            ("Pairwise flip rate (\\%)", ci(m["flipRate"], m["flipRateLo"], m["flipRateHi"])),
            ("AUROC, modal share vs.\\ vote correct", ci(m["auroc"], m["aurocLo"], m["aurocHi"])),
            (f"Vote accuracy, $\\geq$9/10 agree ($n$ = {m['nGeNine']}) (\\%)", m["accGeNine"]),
            (f"Vote accuracy, $<$9/10 agree ($n$ = {m['nLtNine']}) (\\%)", m["accLtNine"]),
            ("Coverage / accuracy at 10/10 agreement (\\%)", f"{m['coverageTen']} / {m['accTen']}")]
    post = []
    if ph is not None:
        post = [("Pairwise agreement between two sampled runs (\\%)", m["pairAgree"]),
                ("AUROC, modal share of first 3 runs vs.\\ vote correct", ci(m["aurocFirstThree"], m["aurocFirstThreeLo"], m["aurocFirstThreeHi"])),
                ("AUROC, greedy agrees with vote vs.\\ vote correct", ci(m["aurocGreedyAgree"], m["aurocGreedyAgreeLo"], m["aurocGreedyAgreeHi"])),
                ("AUROC, modal share vs.\\ greedy correct", ci(m["aurocGreedyCorrect"], m["aurocGreedyCorrectLo"], m["aurocGreedyCorrectHi"]))]
    (gen / "tab_consistency.tex").write_text("\n".join(
        [r"\begin{tabular}{@{}lr@{}}", r"\toprule", r"Measure & Estimate (95\% CI) \\", r"\midrule"]
        + [f"{k} & {v} \\\\" for k, v in cons]
        + ([r"\midrule", r"\multicolumn{2}{@{}l}{\emph{Post hoc comparisons}} \\"] + [f"{k} & {v} \\\\" for k, v in post] if post else [])
        + [r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf-8")

    # Post hoc table: where the RQ1 difference comes from, and whether it holds by MedQA split of origin.
    if ph is not None:
        ptab = [("Single sampled run (mean of 10) $-$ greedy", m["nTest"], ci(m["decompSingle"], m["decompSingleLo"], m["decompSingleHi"]), "--", "--"),
                ("Majority of 10 $-$ single sampled run (mean)", m["nTest"], ci(m["decompVote"], m["decompVoteLo"], m["decompVoteHi"]), "--", "--"),
                ("Majority of 10 $-$ each single run (range)", m["nTest"], f"{m['voteEachMin']} to {m['voteEachMax']}", f"{m['voteEachB']}/{m['voteEachC']}", m["voteEachP"]),
                ("Majority of 10 $-$ second greedy run", m["nTest"], ci(m["voteVsRerun"], m["voteVsRerunLo"], m["voteVsRerunHi"]), "--", m["voteVsRerunP"])]
        for w, label in (("Train", "MedQA train split"), ("Dev", "MedQA dev split"), ("Test", "MedQA test split")):
            ptab.append((f"Majority $-$ greedy, items from the {label}", m[f"nOrigin{w}"],
                         ci(m[f"diffOrigin{w}"], m[f"diffOrigin{w}Lo"], m[f"diffOrigin{w}Hi"]), f"{m[f'bOrigin{w}']}/{m[f'cOrigin{w}']}", m[f"pOrigin{w}"]))
        body = [minus(" & ".join(map(str, row)) + r" \\") for row in ptab]
        (gen / "tab_posthoc.tex").write_text("\n".join(
            [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule", r"Comparison & $n$ & $\Delta$ points (95\% CI) & $b$/$c$ & $p$ \\", r"\midrule"]
            + body[:4] + [r"\midrule"] + body[4:] + [r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf-8")

    cost = [("Greedy (1 run, $T$ = 0)", m["accGreedyMain"], m["tokGreedyItem"], "1.0", m["gpuGreedy"], "1.0"),
            ("One sampled run ($T$ = 0.7), mean of 10", m["accSingle"], f"{st / 10 / n:.0f}", f"{st / 10 / gt:.1f}", "--", "--"),
            ("Majority of 10 ($T$ = 0.7)", m["accVoteMain"], m["tokVoteItem"], m["tokRatio"], m["gpuVote"], m["gpuRatio"])]
    (gen / "tab_cost.tex").write_text("\n".join(
        [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r"Method & Acc.\ (\%) & Tokens/item & $\times$ tokens & GPU s & $\times$ GPU \\", r"\midrule"]
        + [" & ".join(map(str, row)) + r" \\" for row in cost] + [r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf-8")

    expl = [("Majority of 10, $T$ = 0.3", m["accVoteTLow"], ci(m["diffTLow"], m["diffTLowLo"], m["diffTLowHi"]), m["pTLow"], m["unanimityTLow"], m["aurocTLow"]),
            ("Majority of 10, $T$ = 0.7 (main)", m["accVoteMain"], ci(m["diffMain"], m["diffMainLo"], m["diffMainHi"]), m["pMain"], m["unanimity"], m["auroc"]),
            ("Majority of 10, $T$ = 1.0", m["accVoteTHigh"], ci(m["diffTHigh"], m["diffTHighLo"], m["diffTHighHi"]), m["pTHigh"], m["unanimityTHigh"], m["aurocTHigh"]),
            ("Greedy rerun of all items", m["accRerun"], ci(m["rerunDiff"], m["rerunDiffLo"], m["rerunDiffHi"]), m["rerunP"], "--", "--")]
    body = [" & ".join(map(str, row)) + r" \\" for row in expl]
    body = [minus(b_) for b_ in body]
    (gen / "tab_exploratory.tex").write_text("\n".join(
        [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r"Setting & Acc.\ (\%) & $\Delta$ vs.\ greedy (95\% CI) & $p$ & Unanimity (\%) & AUROC \\", r"\midrule"]
        + body + [r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf-8")

    figs = paper / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    copied = [shutil.copy2(f, figs / f.name) for f in sorted((res / "figures").glob("*.pdf"))]
    print(f"{len(macros)} macros, {len(list(gen.glob('tab_*.tex')))} tables -> {gen}; {len(copied)} figures -> {figs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
