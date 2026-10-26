# workflow/rules/09_homology.smk        (upstream of 10_samap.smk)
# =============================================================================
# STATE 2 — homology maps as a checked-in build that RUNS but stays DORMANT.
# Every input is ancient() and the finished maps already exist on disk, so nothing
# fires unless you delete a map (or --force it). Identical discipline to run_samap:
# the artifact is the source of truth; this rule only documents how to remake it.
#
# Recovered end to end from proteomes/slurm/phmmer_{hs_vs_mm,mm_vs_hs}.sh. The phmmer
# line below is VERBATIM. -E 1e-5 is the one scientific knob.
#
# THREE execution contexts — do not collapse them:
#   * chunking      : FASTA splitter unidentified. NOT esl-ssplit (that miniapp isn't built in
#                     HMMER 3.4 and nothing calls it). chunks/{hs,mm}/*.fa.{1..32} already exist
#                     and are ancient() inputs, so no chunker rule is needed for the frozen build.
#   * phmmer search : ABSOLUTE binary via config[samap][phmmer]
#                     (/n/lab_storage/.../software/hmmer-3.4/src/phmmer, rebuilt this session).
#                     NOT bare `phmmer` on PATH — SLURM jobs may not source .bashrc.
#   * aggregate     : config[samap][saturn_python]  (the real run wrote from .../saturn_uv).
#
# config additions:
#   samap:
#     proteomes_root: /n/lab_storage/eddy_lab/users/nhilgert/data/proteomes
#     saturn_python:  /n/lab_storage/eddy_lab/users/nhilgert/.uv/envs/saturn_uv/bin/python
#     phmmer:         /n/lab_storage/eddy_lab/users/nhilgert/software/hmmer-3.4/src/phmmer
# This file reads its own config values, so include order vs 10_samap.smk does not matter.
#
# 2026-09-10: phmmer_to_samap_symbols.py now takes --prot-dir/--out-dir (was hardcoded to the dead
# /n/eddy_lab mount and wrote one level above hsmm/). The aggregate rules pass both explicitly, so
# declared outputs == where the script writes. Rebuild into scratch verified byte-identical to the
# 2026-04-24 maps (see results/tables/homology_rebuild_check.txt).
# =============================================================================
SAMAP_ROOT    = config["samap"]["root"]            # read here too, so include-order is irrelevant
PROT_ROOT     = config["samap"]["proteomes_root"]
SATURN_PY     = config["samap"]["saturn_python"]
PHMMER        = config["samap"]["phmmer"]          # ABSOLUTE path (rebuilt HMMER 3.4), NOT bare `phmmer`
N_CHUNKS      = 32          # SLURM array 1-32 in both phmmer_*.sh
PHMMER_EVALUE = "1e-5"      # -E from the phmmer scripts
# completeness check from proteomes/check.sh (counts '# CPU time' lines in the phmmer logs = finished
# QUERY proteins, one per FASTA entry): hu_vs_mu -> 123845 = n(human proteins) | mu_vs_hu -> 66558 =
# n(mouse proteins). NOT hit counts: the shards hold ~12.8M hit lines per direction at E<=1e-5
# (2026-09-10 rebuild check: aggregate re-run is byte-identical to the 2026-04-24 maps).

# direction -> query species subdir, query/target FASTA basenames, tbl prefix, output map basename
_HMMER_DIRS = {
    "hu_vs_mu": dict(prefix="hu", q_sp="hs",
                     q_fa="Homo_sapiens.GRCh38.pep.all.fa", t_fa="Mus_musculus.GRCm39.pep.all.fa",
                     out="hs_to_mm.txt"),
    "mu_vs_hu": dict(prefix="mu", q_sp="mm",
                     q_fa="Mus_musculus.GRCm39.pep.all.fa", t_fa="Homo_sapiens.GRCh38.pep.all.fa",
                     out="mm_to_hs.txt"),
}

wildcard_constraints:
    direction = "|".join(_HMMER_DIRS),
    prefix    = "hu|mu",
    i         = r"[0-9]+",

def _chunk(wc):   return ancient(f"{PROT_ROOT}/chunks/{_HMMER_DIRS[wc.direction]['q_sp']}/"
                                 f"{_HMMER_DIRS[wc.direction]['q_fa']}.{wc.i}")
def _target(wc):  return ancient(f"{PROT_ROOT}/fasta/{_HMMER_DIRS[wc.direction]['t_fa']}")
def _shards(direction):
    p = _HMMER_DIRS[direction]["prefix"]
    return [ancient(f"{PROT_ROOT}/hm/{direction}/{p}_{i}.tbl") for i in range(1, N_CHUNKS + 1)]

# -----------------------------------------------------------------------------
# SEARCH — one phmmer job per query chunk (Snakemake fans the 1..32 array).
# Chunker note: chunks/{hs,mm}/*.fa.{1..32} already exist and are ancient() inputs here.
# The splitter is NOT esl-ssplit (confirmed absent from HMMER 3.4 + uncalled) and is currently
# unidentified — irrelevant to the frozen build; only a full from-FASTA rebuild would need it.
rule phmmer_chunk:
    input:
        chunk  = _chunk,
        target = _target,
    output:
        tbl = f"{PROT_ROOT}/hm/{{direction}}/{{prefix}}_{{i}}.tbl",
    params:
        phmmer = PHMMER,
    threads: 4
    resources:
        mem_mb  = 8000,      # matches --mem=8G in the array script
        runtime = 360,       # 6h
    log:
        f"{PROT_ROOT}/hm/{{direction}}/{{prefix}}_{{i}}.log",   # phmmer's own stdout log (the .log check.sh counts)
    shell:
        # phmmer line VERBATIM from phmmer_{hs_vs_mm,mm_vs_hs}.sh; ABSOLUTE binary (SLURM won't source .bashrc).
        r"""
        {params.phmmer} --cpu {threads} -E {PHMMER_EVALUE} \
            --tblout {output.tbl} \
            {input.chunk} {input.target} > {log}
        """

# -----------------------------------------------------------------------------
# AGGREGATE — two explicit rules (the map basename isn't a substring of the direction,
# so a single wildcard rule can't name the output). saturn env. Reads the 32 shards +
# both FASTAs (the script builds protein->symbol maps from them).
rule phmmer_aggregate_hs_to_mm:
    input:
        shards = _shards("hu_vs_mu"),
        hs_fa  = ancient(f"{PROT_ROOT}/fasta/Homo_sapiens.GRCh38.pep.all.fa"),
        mm_fa  = ancient(f"{PROT_ROOT}/fasta/Mus_musculus.GRCm39.pep.all.fa"),
        script = ancient(f"{SAMAP_ROOT}/scripts/phmmer_to_samap_symbols.py"),
    output:
        m = f"{SAMAP_ROOT}/homology/hsmm_hmmer/hsmm/hs_to_mm.txt",
    params:
        py = SATURN_PY,
    log:
        f"{SAMAP_ROOT}/logs/phmmer_to_symbols_hu_vs_mu.log",
    shell:
        r"""
        {params.py} {input.script} hu_vs_mu --prot-dir {PROT_ROOT} --out-dir $(dirname {output.m}) > {log} 2>&1
        """

rule phmmer_aggregate_mm_to_hs:
    input:
        shards = _shards("mu_vs_hu"),
        hs_fa  = ancient(f"{PROT_ROOT}/fasta/Homo_sapiens.GRCh38.pep.all.fa"),
        mm_fa  = ancient(f"{PROT_ROOT}/fasta/Mus_musculus.GRCm39.pep.all.fa"),
        script = ancient(f"{SAMAP_ROOT}/scripts/phmmer_to_samap_symbols.py"),
    output:
        m = f"{SAMAP_ROOT}/homology/hsmm_hmmer/hsmm/mm_to_hs.txt",
    params:
        py = SATURN_PY,
    log:
        f"{SAMAP_ROOT}/logs/phmmer_to_symbols_mu_vs_hu.log",
    shell:
        r"""
        {params.py} {input.script} mu_vs_hu --prot-dir {PROT_ROOT} --out-dir $(dirname {output.m}) > {log} 2>&1
        """

# -----------------------------------------------------------------------------
# Explicit target. Dormant: both maps exist + inputs ancient -> "nothing to do".
# Force a rebuild by deleting a map (NOT --forcerun), same as run_samap.
rule homology_build:
    input:
        f"{SAMAP_ROOT}/homology/hsmm_hmmer/hsmm/hs_to_mm.txt",
        f"{SAMAP_ROOT}/homology/hsmm_hmmer/hsmm/mm_to_hs.txt",
