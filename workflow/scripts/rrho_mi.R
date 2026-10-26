#!/usr/bin/env Rscript
# rrho_mi.R — Rank-Rank Hypergeometric Overlap between two MI runs.
#
# Compares the ranked gene lists of two mi_pseudotime.py merged outputs
# (mi_all.csv each) with RRHO (Plaisier et al. 2010, NAR; Bioconductor).
# Because MI is nonnegative (no sign/direction), we run RRHO in one-sided
# ranking mode: genes ranked by MI (descending). This asks "do the two runs
# concentrate the same genes at the top of their rankings?" — which is the
# right question for cross-run MI comparison, and is invariant to any
# monotone estimator bias (the reason RRHO suits MI specifically).
#
# NOTE on package flavors: Bioconductor 'RRHO' is the original; GitHub
# 'RRHO2' (Cahill et al. 2018) fixes heatmap interpretation for SIGNED
# lists (up/down quadrants). For unsigned MI rankings the original RRHO is
# sufficient; RRHO2's quadrant machinery has nothing to act on.
#
# Library: installed to a REPO-LOCAL library (envs/rlib_rrho), not the user
# or system library — set R_LIBS_USER or rely on the .libPaths below.
#
# Usage:
#   Rscript rrho_mi.R A=<mi_all.csv> B=<mi_all.csv> LABELA=<name> LABELB=<name> OUT=<dir>
#
# Outputs under OUT:
#   rrho_heatmap.png        -log10(hypergeom p) over rank-rank grid
#   rrho_summary.json       max -log10 p, its rank coordinates, step size,
#                           n shared genes, overlap gene count at max
#   rrho_top_overlap.csv    genes in the maximal-overlap corner

repo_lib <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))),
                      "..", "..", "envs", "rlib_rrho")
if (dir.exists(repo_lib)) .libPaths(c(normalizePath(repo_lib), .libPaths()))

suppressPackageStartupMessages({
  ok <- requireNamespace("RRHO", quietly = TRUE)
  if (!ok) stop("RRHO not installed. Install to the repo-local lib with:\n",
                "  R_LIBS_USER=envs/rlib_rrho R -e 'BiocManager::install(\"RRHO\")'")
  library(RRHO)
})

args <- commandArgs(trailingOnly = TRUE)
kv <- strsplit(args, "=", fixed = TRUE)
opt <- setNames(vapply(kv, `[`, "", 2L), vapply(kv, `[`, "", 1L))
for (need in c("A", "B", "LABELA", "LABELB", "OUT"))
  if (is.na(opt[need])) stop("missing arg: ", need)

read_mi <- function(path) {
  d <- read.csv(path, stringsAsFactors = FALSE)
  stopifnot("gene" %in% names(d), "mi" %in% names(d))
  d <- d[is.finite(d$mi), c("gene", "mi")]
  d[!duplicated(d$gene), ]
}
a <- read_mi(opt["A"]); b <- read_mi(opt["B"])
shared <- intersect(a$gene, b$gene)
cat(sprintf("run A: %d genes | run B: %d genes | shared: %d\n",
            nrow(a), nrow(b), length(shared)))
stopifnot(length(shared) >= 500)

# RRHO wants data.frames of (gene, score); score ranks the list.
# MI descending = high scores at top. RRHO ranks ascending by default, so
# pass MI directly (it treats larger as more extreme in one-sided mode).
la <- a[match(shared, a$gene), ]; lb <- b[match(shared, b$gene), ]
dir.create(opt["OUT"], recursive = TRUE, showWarnings = FALSE)
step <- max(1L, as.integer(sqrt(length(shared)) / 2))

res <- RRHO::RRHO(la, lb, stepsize = step, labels = c(opt["LABELA"], opt["LABELB"]),
                  alternative = "enrichment", plots = TRUE,
                  outputdir = opt["OUT"], BY = TRUE)

pmat <- res$hypermat            # -log(p) matrix (natural log)
# exclude the trivial outer band (>= 95% of the list in BOTH directions —
# overlap there is near-total by construction and its p is uninformative)
nb <- nrow(pmat)
band <- max(2L, floor(nb * 0.95))
search <- pmat; search[band:nb, band:nb] <- NA
pmax_nat <- max(search, na.rm = TRUE)
ij <- which(search == pmax_nat, arr.ind = TRUE)[1, ]
sum <- list(
  n_shared          = length(shared),
  stepsize          = step,
  max_neglog10_p    = pmax_nat / log(10),
  max_at_rankA      = min(unname(ij[1]) * step, length(shared)),
  max_at_rankB      = min(unname(ij[2]) * step, length(shared)),
  labelA = unname(opt["LABELA"]), labelB = unname(opt["LABELB"]),
  fileA = unname(opt["A"]), fileB = unname(opt["B"])
)
top_a <- la$gene[rank(-la$mi) <= sum$max_at_rankA]
top_b <- lb$gene[rank(-lb$mi) <= sum$max_at_rankB]
overlap <- intersect(top_a, top_b)
sum$n_overlap_at_max <- length(overlap)
writeLines(jsonlite::toJSON(sum, auto_unbox = TRUE, pretty = TRUE),
           file.path(opt["OUT"], "rrho_summary.json"))
write.csv(data.frame(gene = overlap), file.path(opt["OUT"], "rrho_top_overlap.csv"),
          row.names = FALSE)
cat(sprintf("max -log10 p = %.1f at ranks (%d, %d); %d genes in overlap corner\n",
            sum$max_neglog10_p, sum$max_at_rankA, sum$max_at_rankB, length(overlap)))
