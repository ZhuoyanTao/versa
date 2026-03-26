#!/bin/bash
#
# Slurm Launcher for VERSA Processing
# -------------------------------------------
# Usage:
#   ./launcher.sh <pred_wavscp> <gt_wavscp> <score_dir> <split_size> \
#       [--cpu-only|--gpu-only] [--text=FILE] [--reservation=NAME] \
#       [--max-parallel=N] [--cache-root=PATH]
#
# Examples:
#   # GPU+CPU with reservation and 8-way parallel arrays
#   ./launcher.sh pred.scp gt.scp outdir 16 --reservation=sup-17433 --max-parallel=8
#
#   # GPU-only (like your local run), shared cache and transcript file
#   ./launcher.sh /path/pred.scp /path/gt.scp /path/out 16 \
#       --gpu-only --max-parallel=8 \
#       --text=/path/gt_text.txt \
#       --reservation=sup-17433 \
#       --cache-root=/work/nvme/bbjs/ttao3/versa/versa_cache
#
set -euo pipefail

# If someone had HF in offline mode, undo it here
unset HF_HUB_OFFLINE || true

# Colors
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'

show_usage() {
  echo -e "${BLUE}Usage: $0 <pred_wavscp> <gt_wavscp> <score_dir> <split_size> [--cpu-only|--gpu-only] [--text=FILE] [--reservation=NAME] [--max-parallel=N] [--cache-root=PATH]${NC}"
  echo -e "  <pred_wavscp>: Path to prediction wav.scp file"
  echo -e "  <gt_wavscp>:   Path to ground truth wav.scp file (use \"None\" if not available)"
  echo -e "  <score_dir>:   Directory to store results (will be created)"
  echo -e "  <split_size>:  Number of chunks to split the data into"
  echo -e "  --cpu-only / --gpu-only: run only that job type (default: both)"
  echo -e "  --text=FILE:   Path to transcript file (optional; split alongside wav.scp)"
  echo -e "  --reservation=NAME: Slurm reservation to use (e.g., sup-17433)"
  echo -e "  --max-parallel=N: cap concurrent tasks via array %N (e.g., 8)"
  echo -e "  --cache-root=PATH: set shared cache root (HF_HOME, TRANSFORMERS_CACHE, TORCH_HOME, TMPDIR, symlink versa_cache)"
}

# ---------- Required args ----------
if [ $# -lt 4 ]; then
  echo -e "${RED}Error: Insufficient arguments${NC}"
  show_usage; exit 1
fi

PRED_WAVSCP=$1
GT_WAVSCP=$2
SCORE_DIR=$3
SPLIT_SIZE=$4
shift 4

# ---------- Optional args ----------
RUN_CPU=true
RUN_GPU=true
TEXT_FILE=""
RESERVATION=""
RESV_OPT=""
MAX_PARALLEL=""
CACHE_ROOT=""
IO_TYPE=${IO_TYPE:-soundfile}  # io backend for egs/* scripts (soundfile|kaldi|dir)
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

while [[ $# -gt 0 ]]; do
  case $1 in
    --cpu-only) RUN_CPU=true; RUN_GPU=false; echo -e "${YELLOW}CPU-only mode${NC}"; shift ;;
    --gpu-only) RUN_GPU=true; RUN_CPU=false; echo -e "${YELLOW}GPU-only mode${NC}"; shift ;;
    --text=*) TEXT_FILE="${1#*=}"; shift ;;
    --text) shift; TEXT_FILE="${1}"; shift ;;
    --reservation=*) RESERVATION="${1#*=}"; RESV_OPT="--reservation=${RESERVATION}"; shift ;;
    --reservation) shift; RESERVATION="$1"; RESV_OPT="--reservation=${RESERVATION}"; shift ;;
    --max-parallel=*) MAX_PARALLEL="${1#*=}"; shift ;;
    --max-parallel) shift; MAX_PARALLEL="$1"; shift ;;
    --cache-root=*) CACHE_ROOT="${1#*=}"; shift ;;
    --cache-root) shift; CACHE_ROOT="$1"; shift ;;
    *) echo -e "${RED}Error: Unknown option '$1'${NC}"; show_usage; exit 1 ;;
  esac
done

# ---------- Validate inputs ----------
if [ ! -f "${PRED_WAVSCP}" ]; then
  echo -e "${RED}Error: Prediction wav.scp '${PRED_WAVSCP}' not found${NC}"; exit 1
fi
if [ "${GT_WAVSCP}" != "None" ] && [ ! -f "${GT_WAVSCP}" ]; then
  echo -e "${RED}Error: Ground truth wav.scp '${GT_WAVSCP}' not found${NC}"; exit 1
fi
if ! [[ "${SPLIT_SIZE}" =~ ^[0-9]+$ ]]; then
  echo -e "${RED}Error: Split size must be a positive integer${NC}"; exit 1
fi
if [ -n "${TEXT_FILE}" ] && [ ! -f "${TEXT_FILE}" ]; then
  echo -e "${RED}Error: Text file '${TEXT_FILE}' not found${NC}"; exit 1
fi

# ---------- Cluster defaults (override via env) ----------
GPU_PART=${GPU_PARTITION:-gpuA100x4}
CPU_PART=${CPU_PARTITION:-cpu}
ACCOUNT_GPU=${ACCOUNT_GPU:-bbjs-delta-gpu}
ACCOUNT_CPU=${ACCOUNT_CPU:-bbjs-delta-cpu}
GPU_TIME=${GPU_TIME:-2-0:00:00}
CPU_TIME=${CPU_TIME:-2-0:00:00}
CPUS_PER_TASK_GPU=${CPUS_PER_TASK_GPU:-16}
MEM_GPU_TOTAL=${MEM_GPU_TOTAL:-32000M}
CPUS_PER_TASK_CPU=${CPUS_PER_TASK_CPU:-8}
MEM_PER_CPU_CPU=${MEM_PER_CPU_CPU:-2000M}
GPU_TYPE="${GPU_TYPE:-}"             # e.g., a40, v100; empty = any
GPU_OTHER_OPTS="${GPU_OTHER_OPTS:-}" # extra sbatch flags for GPU jobs
CPU_OTHER_OPTS="${CPU_OTHER_OPTS:-}" # extra sbatch flags for CPU jobs

# gres string
if [ -n "${GPU_TYPE}" ]; then
  GPU_GRES="--gres=gpu:${GPU_TYPE}:${GPUS_PER_TASK}"
else
  GPU_GRES="--gres=gpu:${GPUS_PER_TASK}"
fi

# ---------- Optional shared cache configuration ----------
if [ -n "${CACHE_ROOT}" ]; then
  export HF_HOME="${CACHE_ROOT}/hf"
  export TRANSFORMERS_CACHE="${HF_HOME}"
  export TORCH_HOME="${CACHE_ROOT}"
  export TMPDIR="${CACHE_ROOT}/tmp"
  mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"
  ln -snf "${CACHE_ROOT}" "${REPO_DIR}/versa_cache"  # point repo-local symlink to shared cache
  umask 0022
  echo -e "${YELLOW}Shared cache active at: ${CACHE_ROOT}${NC}"
fi

# ---------- Summary ----------
echo -e "${BLUE}=== Configuration Summary ===${NC}"
echo -e "Prediction WAV script: ${PRED_WAVSCP}"
echo -e "Ground truth WAV script: ${GT_WAVSCP}"
echo -e "Output directory: ${SCORE_DIR}"
echo -e "Split size: ${SPLIT_SIZE}"
echo -e "Text file: ${TEXT_FILE:-Not provided}"
echo -e "GPU processing: $(${RUN_GPU} && echo Enabled || echo Disabled)"
if ${RUN_GPU}; then
  echo -e "  Partition: ${GPU_PART} | Account: ${ACCOUNT_GPU} | Time: ${GPU_TIME}"
  echo -e "  CPUs per task: ${CPUS_PER_TASK_GPU} | Mem (total): ${MEM_GPU_TOTAL} | ${GPU_GRES}"
fi
echo -e "CPU processing: $(${RUN_CPU} && echo Enabled || echo Disabled)"
if ${RUN_CPU}; then
  echo -e "  Partition: ${CPU_PART} | Account: ${ACCOUNT_CPU} | Time: ${CPU_TIME}"
  echo -e "  CPUs per task: ${CPUS_PER_TASK_CPU} | Mem per CPU: ${MEM_PER_CPU_CPU}"
fi
echo -e "Reservation: ${RESERVATION:-none}"
echo -e "Max parallel (array cap): ${MAX_PARALLEL:-unlimited}"
echo -e "IO backend: ${IO_TYPE}"
echo ""

# ---------- Create directories ----------
echo -e "${GREEN}Creating directory structure...${NC}"
mkdir -p "${SCORE_DIR}/"{pred,gt,text,result,logs}

# ---------- Split files ----------
total_lines=$(wc -l < "${PRED_WAVSCP}")
source_wavscp=$(basename "${PRED_WAVSCP}")
lines_per_piece=$(( (total_lines + SPLIT_SIZE - 1) / SPLIT_SIZE ))
echo -e "${GREEN}Splitting ${total_lines} lines into ${SPLIT_SIZE} pieces (~${lines_per_piece}/piece)...${NC}"
split -l "${lines_per_piece}" -d -a 3 "${PRED_WAVSCP}" "${SCORE_DIR}/pred/${source_wavscp}_"
pred_list=("${SCORE_DIR}/pred/${source_wavscp}_"*)

if [ "${GT_WAVSCP}" = "None" ]; then
  echo -e "${YELLOW}No ground truth audio provided, evaluation may be reference-free${NC}"
  gt_list=()
else
  target_wavscp=$(basename "${GT_WAVSCP}")
  split -l "${lines_per_piece}" -d -a 3 "${GT_WAVSCP}" "${SCORE_DIR}/gt/${target_wavscp}_"
  gt_list=("${SCORE_DIR}/gt/${target_wavscp}_"*)
  if [ ${#pred_list[@]} -ne ${#gt_list[@]} ]; then
    echo -e "${RED}Error: split counts differ (pred=${#pred_list[@]} vs gt=${#gt_list[@]})${NC}"; exit 1
  fi
fi

text_list=()
if [ -n "${TEXT_FILE}" ]; then
  text_basename=$(basename "${TEXT_FILE}")
  split -l "${lines_per_piece}" -d -a 3 "${TEXT_FILE}" "${SCORE_DIR}/text/${text_basename}_"
  text_list=("${SCORE_DIR}/text/${text_basename}_"*)
  if [ ${#pred_list[@]} -ne ${#text_list[@]} ]; then
    echo -e "${RED}Error: split counts differ (pred=${#pred_list[@]} vs text=${#text_list[@]})${NC}"; exit 1
  fi
fi

# ---------- Submit jobs ----------
JOB_IDS_FILE="${SCORE_DIR}/job_ids.txt"
: > "${JOB_IDS_FILE}"

echo -e "${GREEN}Submitting jobs...${NC}"
num_chunks=${#pred_list[@]}
SUF_FMT="%03d"

if [ -n "${MAX_PARALLEL}" ]; then
  # Submit as job arrays (GPU and/or CPU), capped by %MAX_PARALLEL
  array_spec="0-$((num_chunks-1))%${MAX_PARALLEL}"
  base_pred="${SCORE_DIR}/pred/${source_wavscp}_"
  [ "${GT_WAVSCP}" = "None" ] || base_gt="${SCORE_DIR}/gt/$(basename "${GT_WAVSCP}")_"
  [ -z "${TEXT_FILE}" ] || base_txt="${SCORE_DIR}/text/$(basename "${TEXT_FILE}")_"

  if ${RUN_GPU}; then
    gpu_job_id=$(sbatch --parsable \
      --export=ALL \
      --account "${ACCOUNT_GPU}" -p "${GPU_PART}" ${RESV_OPT} \
      --time "${GPU_TIME}" --cpus-per-task "${CPUS_PER_TASK_GPU}" \
      --mem "${MEM_GPU_TOTAL}" ${GPU_GRES} ${GPU_OTHER_OPTS} \
      -J "gpu_${source_wavscp}" \
      -o "${SCORE_DIR}/logs/gpu_${source_wavscp}_%A_%a.out" \
      -e "${SCORE_DIR}/logs/gpu_${source_wavscp}_%A_%a.err" \
      --array="${array_spec}" \
      --wrap "
idx=\$SLURM_ARRAY_TASK_ID; suf=\$(printf '${SUF_FMT}' \"\$idx\");
pred='${base_pred}'\"\$suf\"
gt=${GT_WAVSCP:+\"${base_gt}\"\"\$suf\"}
txt=${TEXT_FILE:+\"${base_txt}\"\"\$suf\"}
out='${SCORE_DIR}/result/'\$(basename \"\$pred\").result.gpu.txt
${REPO_DIR}/egs/run_gpu.sh \"\$pred\" \"\${gt:-None}\" \"\$out\" ${REPO_DIR}/egs/titw_cpu.yaml \"${IO_TYPE}\" \"\${txt:-}\"
")
    echo "GPU:${gpu_job_id} ARRAY ${array_spec}" >> "${JOB_IDS_FILE}"
    echo -e "${GREEN}Submitted GPU array: ${gpu_job_id} [${array_spec}]${NC}"
  fi

  if ${RUN_CPU}; then
    cpu_job_id=$(sbatch --parsable \
      --export=ALL \
      --account "${ACCOUNT_CPU}" -p "${CPU_PART}" ${RESV_OPT} \
      --time "${CPU_TIME}" --cpus-per-task "${CPUS_PER_TASK_CPU}" \
      --mem-per-cpu "${MEM_PER_CPU_CPU}" ${CPU_OTHER_OPTS} \
      -J "cpu_${source_wavscp}" \
      -o "${SCORE_DIR}/logs/cpu_${source_wavscp}_%A_%a.out" \
      -e "${SCORE_DIR}/logs/cpu_${source_wavscp}_%A_%a.err" \
      --array="${array_spec}" \
      --wrap "
idx=\$SLURM_ARRAY_TASK_ID; suf=\$(printf '${SUF_FMT}' \"\$idx\");
pred='${base_pred}'\"\$suf\"
gt=${GT_WAVSCP:+\"${base_gt}\"\"\$suf\"}
txt=${TEXT_FILE:+\"${base_txt}\"\"\$suf\"}
out='${SCORE_DIR}/result/'\$(basename \"\$pred\").result.cpu.txt
${REPO_DIR}/egs/run_cpu.sh \"\$pred\" \"\${gt:-None}\" \"\$out\" ${REPO_DIR}/egs/titw_cpu.yaml \"${IO_TYPE}\" \"\${txt:-}\"
")
    echo "CPU:${cpu_job_id} ARRAY ${array_spec}" >> "${JOB_IDS_FILE}"
    echo -e "${GREEN}Submitted CPU array: ${cpu_job_id} [${array_spec}]${NC}"
  fi

else
  # One job per chunk
  for ((i=0; i<num_chunks; i++)); do
    sub_pred_wavscp=${pred_list[$i]}
    job_prefix=$(basename "${sub_pred_wavscp}")
    sub_gt_wavscp=$([ "${GT_WAVSCP}" = "None" ] && echo "None" || echo "${gt_list[$i]}")
    sub_text_file=$([ -n "${TEXT_FILE}" ] && echo "${text_list[$i]}" || echo "")
    echo -e "${BLUE}Processing chunk $((i+1))/${num_chunks}: ${sub_pred_wavscp}${NC}"
    if [ -n "${sub_text_file}" ]; then echo -e "${BLUE}  Text file: ${sub_text_file}${NC}"; fi

    if ${RUN_GPU}; then
      gpu_job_id=$(sbatch --parsable \
        --export=ALL \
        --account "${ACCOUNT_GPU}" -p "${GPU_PART}" ${RESV_OPT} \
        --time "${GPU_TIME}" --cpus-per-task "${CPUS_PER_TASK_GPU}" \
        --mem "${MEM_GPU_TOTAL}" ${GPU_GRES} ${GPU_OTHER_OPTS} \
        -J "gpu_${job_prefix}" \
        -o "${SCORE_DIR}/logs/gpu_${job_prefix}_%j.out" \
        -e "${SCORE_DIR}/logs/gpu_${job_prefix}_%j.err" \
        ${REPO_DIR}/egs/run_gpu.sh \
          "${sub_pred_wavscp}" \
          "${sub_gt_wavscp}" \
          "${SCORE_DIR}/result/$(basename "${sub_pred_wavscp}").result.gpu.txt" \
          ${REPO_DIR}/egs/titw_cpu.yaml \
          "${IO_TYPE}" \
          "${sub_text_file}")
      echo "GPU:${gpu_job_id} CHUNK:$((i+1))/${num_chunks} FILE:${job_prefix}" >> "${JOB_IDS_FILE}"
      echo -e "  Submitted GPU job: ${gpu_job_id}"
    fi

    if ${RUN_CPU}; then
      cpu_job_id=$(sbatch --parsable \
        --export=ALL \
        --account "${ACCOUNT_CPU}" -p "${CPU_PART}" ${RESV_OPT} \
        --time "${CPU_TIME}" --cpus-per-task "${CPUS_PER_TASK_CPU}" \
        --mem-per-cpu "${MEM_PER_CPU_CPU}" ${CPU_OTHER_OPTS} \
        -J "cpu_${job_prefix}" \
        -o "${SCORE_DIR}/logs/cpu_${job_prefix}_%j.out" \
        -e "${SCORE_DIR}/logs/cpu_${job_prefix}_%j.err" \
        ${REPO_DIR}/egs/run_cpu.sh \
          "${sub_pred_wavscp}" \
          "${sub_gt_wavscp}" \
          "${SCORE_DIR}/result/$(basename "${sub_pred_wavscp}").result.cpu.txt" \
          ${REPO_DIR}/egs/titw_cpu.yaml \
          "${IO_TYPE}" \
          "${sub_text_file}")
      echo "CPU:${cpu_job_id} CHUNK:$((i+1))/${num_chunks} FILE:${job_prefix}" >> "${JOB_IDS_FILE}"
      echo -e "  Submitted CPU job: ${cpu_job_id}"
    fi
  done
fi

echo -e "${GREEN}All jobs submitted. Job IDs saved to: ${JOB_IDS_FILE}${NC}"

# Create a handy dependent merge command (depends on arrays or per-chunk jobs)
if ${RUN_GPU} && ${RUN_CPU}; then
  gpu_ids=$(grep "^GPU:" "${JOB_IDS_FILE}" | awk '{print $1}' | sed 's/^GPU://' | paste -sd, -)
  cpu_ids=$(grep "^CPU:" "${JOB_IDS_FILE}" | awk '{print $1}' | sed 's/^CPU://' | paste -sd, -)
  all_ids="${gpu_ids}${gpu_ids:+,}${cpu_ids}"
  job_type="GPU and CPU"
elif ${RUN_GPU}; then
  all_ids=$(grep "^GPU:" "${JOB_IDS_FILE}" | awk '{print $1}' | sed 's/^GPU://' | paste -sd, -)
  job_type="GPU"
else
  all_ids=$(grep "^CPU:" "${JOB_IDS_FILE}" | awk '{print $1}' | sed 's/^CPU://' | paste -sd, -)
  job_type="CPU"
fi

echo -e "${YELLOW}To run a dependent merge after ALL ${job_type} tasks complete:${NC}"
echo -e "sbatch --dependency=afterok:${all_ids} ${REPO_DIR}/scripts/show_result.sh ${SCORE_DIR}/result ${SCORE_DIR}/final_results.txt"
echo -e "${GREEN}Done! Monitor with: squeue -u \$(whoami)${NC}"
