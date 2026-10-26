# Clustree over the Leiden ladder.
#
# From Cagri's original (thanks) -- fixed the curly quotes, which R can't parse,
# and parameterised the paths so it works from Snakemake or the command line.
#
#   Rscript clustree_plot.R <input_csv> <output_png> [title]
#
# Input: one column per resolution, named leiden_scVI_<res> (that prefix is what
# clustree keys on). Produced by workflow/scripts/export_clustree.py.

suppressPackageStartupMessages({
  library(clustree)
  library(readr)
  library(ggplot2)
})

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
if (length(res_cols) < 2) {
  stop(sprintf("need >=2 leiden_scVI_* columns for a tree; found %d in %s",
               length(res_cols), input_file))
}
# clustree needs these as factors, and ordered by resolution not alphabetically
ord <- order(as.numeric(sub("^leiden_scVI_", "", res_cols)))
for (cc in res_cols) data[[cc]] <- as.factor(data[[cc]])
message(sprintf("clustree over %d resolutions: %s",
                length(res_cols), paste(res_cols[ord], collapse = ", ")))

clustree_plot <- clustree(data, prefix = "leiden_scVI_")
if (nzchar(plot_title)) clustree_plot <- clustree_plot + ggtitle(plot_title)

dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
ggsave(output_file, clustree_plot, width = 10, height = 8, dpi = 300)
message(sprintf("wrote %s", output_file))
