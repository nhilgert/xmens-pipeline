# Homology maps for SAMap (upstream of 10_samap.smk): a checked-in build that runs
# but stays dormant. Every input is ancient() and the finished maps already exist on
# disk, so nothing fires unless a map is deleted (or forced). Same discipline as
# run_samap: the artifact is the source of truth; this file documents how to remake it.
#
# Reads the human and mouse proteomes (chunked FASTA) under config[samap][proteomes_root],
# runs phmmer in both directions, and writes the symbol maps hs_to_mm.txt / mm_to_hs.txt
# under config[samap][root]/homology/hsmm_hmmer/hsmm/. Target: homology_build.
#
# Recovered end to end from proteomes/slurm/phmmer_{hs_vs_mm,mm_vs_hs}.sh. The phmmer
# line below is verbatim. -E 1e-5 is the one scientific knob.
#
# Three execution contexts; do not collapse them:
#   * chunking      : the FASTA splitter is unidentified. It is not esl-ssplit (that
#                     miniapp is not built in HMMER 3.4 and nothing calls it).
#                     chunks/{hs,mm}/*.fa.{1..32} already exist and are ancient()
#                     inputs, so no chunker rule is needed for the frozen build.
#   * phmmer search : absolute binary via config[samap][phmmer] (HMMER 3.4 build),
#                     not bare `phmmer` on PATH, because SLURM jobs may not source .bashrc.
#   * aggregate     : config[samap][saturn_python] (the saturn_uv interpreter).
#
# config additions:
#   samap:
#     proteomes_root: <proteomes directory: fasta/, chunks/, hm/>
#     saturn_python:  <saturn_uv python>
#     phmmer:         <absolute path to hmmer-3.4/src/phmmer>
# This file reads its own config values, so include order vs 10_samap.smk does not matter.
#
# phmmer_to_samap_symbols.py takes --prot-dir/--out-dir, and the aggregate rules pass
# both explicitly, so the declared outputs are where the script writes. A rebuild into
# scratch was byte-identical to the frozen maps (results/tables/homology_rebuild_check.txt).

SAMAP_ROOT    = config["samap"]["root"]
PROT_ROOT     = config["samap"]["proteomes_root"]
SATURN_PY     = config["samap"]["saturn_python"]
PHMMER        = config["samap"]["phmmer"]
N_CHUNKS      = 32          # SLURM array 1-32 in both phmmer_*.sh
PHMMER_EVALUE = "1e-5"      # -E from the phmmer scripts
# Completeness check (proteomes/check.sh) counts '# CPU time' lines in the phmmer
# logs, one per finished query protein: hu_vs_mu -> 123845 (human proteins),
# mu_vs_hu -> 66558 (mouse proteins). These are not hit counts; the shards hold
# ~12.8M hit lines per direction at E <= 1e-5.

# direction -> query species subdir, query/target FASTA basenames, tbl prefix,
# output map basename
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

# Search: one phmmer job per query chunk (Snakemake fans out the 1..32 array).
# The chunks are ancient() inputs; only a full from-FASTA rebuild would need the splitter.
rule phmmer_chunk:
    input:
        chunk  = _chunk,
        target = _target,
    output:
        tbl = f"{PROT_ROOT}/hm/{{direction}}/{{prefix}}_{{i}}.tbl",
    params:
        phmmer = PHMMER,
    log:
        # phmmer's own stdout log, the .log that check.sh counts
        f"{PROT_ROOT}/hm/{{direction}}/{{prefix}}_{{i}}.log",
    threads: 4
    resources:
        mem_mb  = 8000,      # MB; matches --mem=8G in the array script
        runtime = 360,       # minutes (6 h)
    shell:
        # phmmer line verbatim from phmmer_{hs_vs_mm,mm_vs_hs}.sh; absolute binary
        # because SLURM job shells do not source .bashrc.
        r"""
        {params.phmmer} --cpu {threads} -E {PHMMER_EVALUE} \
            --tblout {output.tbl} \
            {input.chunk} {input.target} > {log}
        """

# Aggregate: two explicit rules, because the map basename is not a substring of the
# direction, so a single wildcard rule cannot name the output. Inputs: the 32 shards and
# both FASTAs; the script is given the direction and --prot-dir (saturn env).
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


# Mouse-query direction of the aggregate step above.
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

# Target. Dormant: both maps exist and the inputs are ancient, so "nothing to do".
# Force a rebuild by deleting a map (not --forcerun), same as run_samap.
rule homology_build:
    input:
        f"{SAMAP_ROOT}/homology/hsmm_hmmer/hsmm/hs_to_mm.txt",
        f"{SAMAP_ROOT}/homology/hsmm_hmmer/hsmm/mm_to_hs.txt",
