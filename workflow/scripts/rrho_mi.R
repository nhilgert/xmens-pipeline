#!/usr/bin/env Rscript
# Rank-Rank Hypergeometric Overlap (RRHO) between two mutual-information runs.
#
# Reads two mi_pseudotime.py merged outputs (mi_all.csv: gene, mi) and writes,
# under OUT, the RRHO heatmap (rrho_heatmap.png), a summary of the overlap
# maximum (rrho_summary.json) and the genes in the maximal-overlap corner
# (rrho_top_overlap.csv). Standalone tool; not called by a Snakemake rule.
#
#   Rscript rrho_mi.R A=<mi_all.csv> B=<mi_all.csv> LABELA=<name> LABELB=<name> OUT=<dir>
#
# RRHO (Plaisier et al. 2010, NAR; Bioconductor package RRHO) asks whether the
# two runs concentrate the same genes at the top of their rankings. MI is
# nonnegative, so the lists are unsigned and RRHO runs in one-sided mode with
# genes ranked by MI descending. The comparison depends only on ranks, so it is
# invariant to any monotone estimator bias between runs, which is what makes it
# suitable for MI. The GitHub RRHO2 package (Cahill et al. 2018) only changes
# the heatmap for signed lists (up/down quadrants), so it has nothing to add for
# unsigned rankings.

# RRHO lives in a repo-local library (envs/rlib_rrho), not the user or system
# library; the path is resolved relative to this script so Rscript finds it
# from any working directory.
repo_lib <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))),
                      "..", "..", "envs", "rlib_rrho")
if (dir.exists(repo_lib)) .libPaths(c(normalizePath(repo_lib), .libPaths()))

suppressPackageStartupMessages({
  ok <- requireNamespace("RRHO", quietly = TRUE)
  if (!ok) stop("RRHO not installed. Install to the repo-local lib with:\n",
                "  R_LIBS_USER=envs/rlib_rrho R -e 'BiocManager::install(\"RRHO\")'")
  library(RRHO)
})

# Arguments are KEY=value pairs, parsed into a named character vector.
args <- commandArgs(trailingOnly = TRUE)
kv <- strsplit(args, "=", fixed = TRUE)
opt <- setNames(vapply(kv, `[`, "", 2L), vapply(kv, `[`, "", 1L))
for (need in c("A", "B", "LABELA", "LABELB", "OUT"))
  if (is.na(opt[need])) stop("missing arg: ", need)

# Load the two rankings ----
read_mi <- function(path) {
  d <- read.csv(path, stringsAsFactors = FALSE)
  stopifnot("gene" %in% names(d), "mi" %in% names(d))
  d <- d[is.finite(d$mi), c("gene", "mi")]
  d[!duplicated(d$gene), ]
}

# Only genes present in both runs can be ranked against each other.
a <- read_mi(opt["A"]); b <- read_mi(opt["B"])
shared <- intersect(a$gene, b$gene)
cat(sprintf("run A: %d genes | run B: %d genes | shared: %d\n",
            nrow(a), nrow(b), length(shared)))
stopifnot(length(shared) >= 500)  # minimum shared genes for a usable rank grid

# Run RRHO ----
# RRHO wants data.frames of (gene, score); score ranks the list.
# MI is passed directly as the score.
la <- a[match(shared, a$gene), ]; lb <- b[match(shared, b$gene), ]
dir.create(opt["OUT"], recursive = TRUE, showWarnings = FALSE)

# Rank-grid step in genes; the grid is (n / step)^2 cells.
step <- max(1L, as.integer(sqrt(length(shared)) / 2))

# BY = TRUE applies Benjamini-Yekutieli correction across the grid cells.
res <- RRHO::RRHO(la, lb, stepsize = step, labels = c(opt["LABELA"], opt["LABELB"]),
                  alternative = "enrichment", plots = TRUE,
                  outputdir = opt["OUT"], BY = TRUE)

# Locate the overlap maximum ----
pmat <- res$hypermat            # -log(p) matrix (natural log)

# Exclude the trivial outer band (>= 95% of the list in both directions):
# overlap there is near-total by construction and its p is uninformative.
nb <- nrow(pmat)
band <- max(2L, floor(nb * 0.95))
search <- pmat; search[band:nb, band:nb] <- NA
pmax_nat <- max(search, na.rm = TRUE)
ij <- which(search == pmax_nat, arr.ind = TRUE)[1, ]

# Grid indices convert back to list ranks via the step size; the clamp handles
# the last (partial) grid cell.
sum <- list(
  n_shared          = length(shared),
  stepsize          = step,
  max_neglog10_p    = pmax_nat / log(10),
  max_at_rankA      = min(unname(ij[1]) * step, length(shared)),
  max_at_rankB      = min(unname(ij[2]) * step, length(shared)),
  labelA = unname(opt["LABELA"]), labelB = unname(opt["LABELB"]),
  fileA = unname(opt["A"]), fileB = unname(opt["B"])
)
# Write the summary and the genes in the maximal-overlap corner ----
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
