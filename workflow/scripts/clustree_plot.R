# Clustree plot of the Leiden ladder for one (species, lineage).
#
# Reads the per-cell cluster table written by workflow/scripts/atlas/
# export_clustree.py (one column per resolution, named leiden_scVI_<res>; that
# prefix is what clustree keys on) and writes the clustree PNG. Called by rule
# clustree_plot in workflow/rules/98_clustree.smk; also runs from the shell:
#
#   Rscript clustree_plot.R <input_csv> <output_png> [title]

suppressPackageStartupMessages({
  library(clustree)
  library(readr)
  library(ggplot2)
})

# Paths come from the Snakemake object under a rule, otherwise from argv.
if (exists("snakemake")) {
  input_file  <- snakemake@input[["csv"]]
  output_file <- snakemake@output[["png"]]
  plot_title  <- snakemake@params[["title"]]
} else {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 2) stop("usage: Rscript clustree_plot.R <input_csv> <output_png> [title]")
  input_file  <- args[1]
  output_file <- args[2]
  plot_title  <- if (length(args) >= 3) args[3] else ""
}

data <- read_csv(input_file, show_col_types = FALSE)

res_cols <- grep("^leiden_scVI_", colnames(data), value = TRUE)
# A tree needs at least two resolutions.
if (length(res_cols) < 2) {
  stop(sprintf("need >=2 leiden_scVI_* columns for a tree; found %d in %s",
               length(res_cols), input_file))
}

# clustree needs the cluster columns as factors; ord sorts the resolutions
# numerically for the log line (alphabetical order would put 10 before 2).
ord <- order(as.numeric(sub("^leiden_scVI_", "", res_cols)))
for (cc in res_cols) data[[cc]] <- as.factor(data[[cc]])
message(sprintf("clustree over %d resolutions: %s",
                length(res_cols), paste(res_cols[ord], collapse = ", ")))

clustree_plot <- clustree(data, prefix = "leiden_scVI_")
if (nzchar(plot_title)) clustree_plot <- clustree_plot + ggtitle(plot_title)

dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
# 10 x 8 in at 300 dpi (landscape).
ggsave(output_file, clustree_plot, width = 10, height = 8, dpi = 300)
message(sprintf("wrote %s", output_file))
