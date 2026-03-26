#!/usr/bin/env bash
#
# Local Launcher for VERSA Processing (with Vocos auto-discovery)
# ---------------------------------------------------------------
# Usage:
#   ./local_launcher.sh <pred_wavscp> <gt_wavscp> <score_dir> <split_size> \
#       [--cpu-only|--gpu-only] [--max-parallel=N] [--text=FILE] \
#       [--vocos-base=DIR] [--vocos-pattern=GLOB] [--vocos-subpath=REL] \
#       [--config=FILE] [--io-type=soundfile|torchaudio]
#
# Examples:
#   # Auto-collect predictions from Vocos outputs and run reference-free eval (CPU only)
#   ./local_launcher.sh IGNORE None /work/.../versa_scores 16 \
#     --cpu-only \
#     --max-parallel=8 \
#     --vocos-base=/work/.../decode_.../titw_easy_test/log \
#     --vocos-pattern='output.*' \
#     --vocos-subpath='vocos_wav/wav.scp' \
#     --config=/work/.../versa/egs/titw.yaml
#
#   # Manual paths (GPU only)
#   ./local_launcher.sh data/pred.scp data/gt.scp /work/.../versa_scores 16 \
#     --gpu-only --max-parallel=4 --config=/work/.../versa/egs/titw.yaml
#
set -euo pipefail

# ------------ Colors ------------
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'

show_usage() {
  echo -e "${BLUE}Usage: $0 <pred_wavscp> <gt_wavscp> <score_dir> <split_size> [options]${NC}"
  echo "Required:"
  echo "  <pred_wavscp>   Path to prediction wav.scp (ignored if --vocos-base is used)"
  echo "  <gt_wavscp>     Ground-truth wav.scp or 'None'"
  echo "  <score_dir>     Output directory for scores/logs"
  echo "  <split_size>    Number of chunks to split inputs into"
  echo "Options:"
  echo "  --cpu-only                  Run only CPU jobs"
  echo "  --gpu-only                  Run only GPU jobs"
  echo "  --max-parallel=N            Max parallel processes (default: nproc)"
  echo "  --text=FILE                 Optional text file aligned with wav.scp"
  echo "  --config=FILE               VERSA config (default: /work/nvme/bbjs/ttao3/versa/egs/titw.yaml)"
  echo "  --io-type=BACKEND           'soundfile' (default) or 'torchaudio'"
  echo "Auto-discovery of Vocos predictions:"
  echo "  --vocos-base=DIR            Base dir that contains output.* folders (e.g., .../log)"
  echo "  --vocos-pattern=GLOB        Glob for outputs (default: 'output.*')"
  echo "  --vocos-subpath=REL         Path to wav.scp inside each output (default: 'vocos_wav/wav.scp')"
}

# ------------ Args ------------
if [ $# -lt 4 ]; then
  echo -e "${RED}Error: Insufficient arguments${NC}"; show_usage; exit 1
fi

PRED_WAVSCP=$1
GT_WAVSCP=$2
SCORE_DIR=$3
SPLIT_SIZE=$4

# Defaults
RUN_CPU=true
RUN_GPU=true
MAX_PARALLEL=$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)
TEXT_FILE=""
IO_TYPE="${IO_TYPE:-soundfile}"   # can be overridden via --io-type=
CONFIG_FILE="/work/nvme/bbjs/ttao3/versa/egs/titw.yaml"

# Vocos auto-discovery defaults
VOCOS_BASE=""
VOCOS_PATTERN="output.*"
VOCOS_SUBPATH="vocos_wav/wav.scp"

shift 4
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu-only) RUN_GPU=false; RUN_CPU=true; echo -e "${YELLOW}CPU-only mode${NC}"; shift;;
    --gpu-only) RUN_GPU=true; RUN_CPU=false; echo -e "${YELLOW}GPU-only mode${NC}"; shift;;
    --max-parallel=*) MAX_PARALLEL="${1#*=}"
      if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || [ "$MAX_PARALLEL" -eq 0 ]; then
        echo -e "${RED}Error: --max-parallel must be positive integer${NC}"; exit 1
      fi
      echo -e "${YELLOW}Max parallel: ${MAX_PARALLEL}${NC}"; shift;;
    --text=*) TEXT_FILE="${1#*=}"
      [ -f "$TEXT_FILE" ] || { echo -e "${RED}Error: text file '$TEXT_FILE' not found${NC}"; exit 1; }
      echo -e "${YELLOW}Text: ${TEXT_FILE}${NC}"; shift;;
    --config=*) CONFIG_FILE="${1#*=}"
      [ -f "$CONFIG_FILE" ] || { echo -e "${RED}Error: config '$CONFIG_FILE' not found${NC}"; exit 1; }
      echo -e "${YELLOW}Config: ${CONFIG_FILE}${NC}"; shift;;
    --io-type=*) IO_TYPE="${1#*=}"; echo -e "${YELLOW}IO type: ${IO_TYPE}${NC}"; shift;;
    --vocos-base=*) VOCOS_BASE="${1#*=}"; echo -e "${YELLOW}Vocos base: ${VOCOS_BASE}${NC}"; shift;;
    --vocos-pattern=*) VOCOS_PATTERN="${1#*=}"; echo -e "${YELLOW}Vocos pattern: ${VOCOS_PATTERN}${NC}"; shift;;
    --vocos-subpath=*) VOCOS_SUBPATH="${1#*=}"; echo -e "${YELLOW}Vocos subpath: ${VOCOS_SUBPATH}${NC}"; shift;;
    -h|--help) show_usage; exit 0;;
    *)
      echo -e "${RED}Error: Unknown option '$1'${NC}"; show_usage; exit 1;;
  esac
done

# ------------ Validate / auto-discover preds ------------
mkdir -p "${SCORE_DIR}"/{pred,gt,text,result,logs}

if [ -n "${VOCOS_BASE}" ]; then
  echo -e "${GREEN}Collecting Vocos preds from: ${VOCOS_BASE}/${VOCOS_PATTERN}/${VOCOS_SUBPATH}${NC}"
  shopt -s nullglob
  declare -a found_scps=()
  for f in "${VOCOS_BASE}"/${VOCOS_PATTERN}/${VOCOS_SUBPATH}; do [ -f "$f" ] && found_scps+=("$f"); done
  # (also handle deeper nesting: output.N/anything/vocos_wav/wav.scp)
  for f in "${VOCOS_BASE}"/${VOCOS_PATTERN}/*/${VOCOS_SUBPATH}; do [ -f "$f" ] && found_scps+=("$f"); done
  shopt -u nullglob

  if [ ${#found_scps[@]} -eq 0 ]; then
    echo -e "${RED}Error: No wav.scp found under ${VOCOS_BASE}/${VOCOS_PATTERN}/${VOCOS_SUBPATH}${NC}"
    exit 1
  fi

  MERGED_PRED="${SCORE_DIR}/pred/all_vocos_wav.scp"
  # Merge & dedupe by utt-id (first column), keep first occurrence
  awk 'FNR==1{nextfile} { if(!seen[$1]++){print $0} }' "${found_scps[@]}" > "${MERGED_PRED}" || {
    # fallback for awk without nextfile
    > "${MERGED_PRED}"
    for f in "${found_scps[@]}"; do cat "$f"; done | awk '{ if(!seen[$1]++){print $0} }' >> "${MERGED_PRED}"
  }
  echo -e "${GREEN}Merged ${#found_scps[@]} wav.scp → ${MERGED_PRED} ($(wc -l < "${MERGED_PRED}") lines)${NC}"
  PRED_WAVSCP="${MERGED_PRED}"
fi

[ -f "${PRED_WAVSCP}" ] || { echo -e "${RED}Error: Prediction wav.scp '${PRED_WAVSCP}' not found${NC}"; exit 1; }
if [ "${GT_WAVSCP}" != "None" ] && [ ! -f "${GT_WAVSCP}" ]; then
  echo -e "${RED}Error: Ground truth wav.scp '${GT_WAVSCP}' not found${NC}"; exit 1
fi
[[ "${SPLIT_SIZE}" =~ ^[0-9]+$ ]] || { echo -e "${RED}Error: split_size must be integer${NC}"; exit 1; }

# ------------ GPU detection ------------
GPU_COUNT=0
if $RUN_GPU; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo -e "${YELLOW}Warning: nvidia-smi not found; assuming 1 GPU slot${NC}"
    GPU_COUNT=1
  else
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    echo -e "${GREEN}Found ${GPU_COUNT} GPU(s)${NC}"
    echo -e "${BLUE}Available GPUs:${NC}"; nvidia-smi -L | sed 's/^/  /'
  fi
fi

# Decide slots
GPU_MAX_PARALLEL=0; CPU_MAX_PARALLEL=0
if $RUN_GPU && $RUN_CPU; then
  if [ $GPU_COUNT -gt 0 ]; then
    GPU_MAX_PARALLEL=$(( (MAX_PARALLEL + 1) / 2 ))
    [ $GPU_MAX_PARALLEL -gt $GPU_COUNT ] && GPU_MAX_PARALLEL=$GPU_COUNT
    CPU_MAX_PARALLEL=$(( MAX_PARALLEL - GPU_MAX_PARALLEL ))
    [ $CPU_MAX_PARALLEL -lt 1 ] && CPU_MAX_PARALLEL=1
    [ $GPU_MAX_PARALLEL -lt 1 ] && GPU_MAX_PARALLEL=1
  else
    CPU_MAX_PARALLEL=$MAX_PARALLEL
  fi
elif $RUN_GPU; then
  GPU_MAX_PARALLEL=$(( MAX_PARALLEL > GPU_COUNT ? GPU_COUNT : MAX_PARALLEL ))
  [ $GPU_MAX_PARALLEL -lt 1 ] && GPU_MAX_PARALLEL=1
else
  CPU_MAX_PARALLEL=$MAX_PARALLEL
fi

# ------------ Summary ------------
echo -e "${BLUE}=== Configuration Summary ===${NC}"
echo -e "Prediction WAV script: ${PRED_WAVSCP}"
echo -e "Ground truth WAV script: ${GT_WAVSCP}"
echo -e "Output directory: ${SCORE_DIR}"
echo -e "Split size: ${SPLIT_SIZE}"
echo -e "Maximum parallel processes: ${MAX_PARALLEL}"
echo -e "Available CPU cores: $(command -v nproc >/dev/null 2>&1 && nproc || echo '?')"
echo -e "Config file: ${CONFIG_FILE}"
echo -e "IO type: ${IO_TYPE}"
[ -n "${TEXT_FILE}" ] && echo -e "Text file: ${TEXT_FILE}" || echo -e "Text file: Not provided"
$RUN_GPU && echo -e "GPU processing: Enabled (${GPU_COUNT} GPUs), slots: ${GPU_MAX_PARALLEL}" || echo -e "GPU processing: Disabled"
$RUN_CPU && echo -e "CPU processing: Enabled, slots: ${CPU_MAX_PARALLEL}" || echo -e "CPU processing: Disabled"
echo ""

# ------------ Directory structure ------------
mkdir -p "${SCORE_DIR}"/{pred,gt,text,result,logs}

# ------------ Split inputs ------------
total_lines=$(wc -l < "${PRED_WAVSCP}")
source_wavscp=$(basename "${PRED_WAVSCP}")
lines_per_piece=$(( (total_lines + SPLIT_SIZE - 1) / SPLIT_SIZE ))
echo -e "${GREEN}Splitting ${total_lines} lines into ${SPLIT_SIZE} pieces (~${lines_per_piece}/piece)...${NC}"
split -l "${lines_per_piece}" -d -a 3 "${PRED_WAVSCP}" "${SCORE_DIR}/pred/${source_wavscp}_"
pred_list=( "${SCORE_DIR}/pred/${source_wavscp}_"* )

if [ "${GT_WAVSCP}" = "None" ]; then
  echo -e "${YELLOW}No ground truth provided; running reference-free metrics${NC}"
  gt_list=()
else
  target_wavscp=$(basename "${GT_WAVSCP}")
  split -l "${lines_per_piece}" -d -a 3 "${GT_WAVSCP}" "${SCORE_DIR}/gt/${target_wavscp}_"
  gt_list=( "${SCORE_DIR}/gt/${target_wavscp}_"* )
  if [ ${#pred_list[@]} -ne ${#gt_list[@]} ]; then
    echo -e "${RED}Error: split counts differ: preds=${#pred_list[@]} gt=${#gt_list[@]}${NC}"; exit 1
  fi
fi

text_list=()
if [ -n "${TEXT_FILE}" ]; then
  text_base=$(basename "${TEXT_FILE}")
  split -l "${lines_per_piece}" -d -a 3 "${TEXT_FILE}" "${SCORE_DIR}/text/${text_base}_"
  text_list=( "${SCORE_DIR}/text/${text_base}_"* )
  if [ ${#pred_list[@]} -ne ${#text_list[@]} ]; then
    echo -e "${RED}Error: split counts differ: preds=${#pred_list[@]} text=${#text_list[@]}${NC}"; exit 1
  fi
fi

# ------------ Helpers ------------
run_job() {
  local job_type=$1 sub_pred=$2 sub_gt=$3 sub_text=$4 out_file=$5 gpu_rank=${6:-}
  local log_file="${SCORE_DIR}/logs/${job_type}_$(basename "${sub_pred}").log"

  if [ "${job_type}" = "gpu" ]; then
    if [ -n "${gpu_rank}" ]; then
      echo -e "${BLUE}Starting GPU job: $(basename "${sub_pred}") on GPU ${gpu_rank}${NC}"
      /work/nvme/bbjs/ttao3/versa/egs/run_gpu.sh \
        "${sub_pred}" "${sub_gt}" "${out_file}" "${CONFIG_FILE}" "${IO_TYPE}" "${sub_text}" "${gpu_rank}" \
        > "${log_file}" 2>&1
    else
      echo -e "${BLUE}Starting GPU job: $(basename "${sub_pred}")${NC}"
      /work/nvme/bbjs/ttao3/versa/egs/run_gpu.sh \
        "${sub_pred}" "${sub_gt}" "${out_file}" "${CONFIG_FILE}" "${IO_TYPE}" "${sub_text}" \
        > "${log_file}" 2>&1
    fi
  else
    echo -e "${BLUE}Starting CPU job: $(basename "${sub_pred}")${NC}"
    /work/nvme/bbjs/ttao3/versa/egs/run_cpu.sh \
      "${sub_pred}" "${sub_gt}" "${out_file}" "${CONFIG_FILE}" "${IO_TYPE}" "${sub_text}" \
      > "${log_file}" 2>&1
  fi
}

declare -a gpu_job_pids=() gpu_job_info=() gpu_ranks_in_use=()
declare -a cpu_job_pids=() cpu_job_info=()
RUNNING_JOBS_FILE="${SCORE_DIR}/running_jobs.txt"
COMPLETED_JOBS_FILE="${SCORE_DIR}/completed_jobs.txt"
FAILED_JOBS_FILE="${SCORE_DIR}/failed_jobs.txt"
: > "${RUNNING_JOBS_FILE}"; : > "${COMPLETED_JOBS_FILE}"; : > "${FAILED_JOBS_FILE}"

get_next_gpu_rank() {
  local rank used
  for ((rank=0; rank<GPU_COUNT; rank++)); do
    used=false
    for r in "${gpu_ranks_in_use[@]}"; do [ "$r" = "$rank" ] && { used=true; break; }; done
    $used || { echo "$rank"; return 0; }
  done
  echo -1
}

wait_for_gpu_slot() {
  local idx
  while [ ${#gpu_job_pids[@]} -ge $GPU_MAX_PARALLEL ]; do
    for idx in "${!gpu_job_pids[@]}"; do
      if ! kill -0 "${gpu_job_pids[$idx]}" 2>/dev/null; then
        wait "${gpu_job_pids[$idx]}"; local ec=$?
        local info="${gpu_job_info[$idx]}"
        if [[ $info =~ GPU_RANK:([0-9]+) ]]; then
          local freed="${BASH_REMATCH[1]}"; # free rank
          local tmp=(); for r in "${gpu_ranks_in_use[@]}"; do [ "$r" != "$freed" ] && tmp+=("$r"); done
          gpu_ranks_in_use=("${tmp[@]}")
        fi
        [ $ec -eq 0 ] && echo "$info" >> "${COMPLETED_JOBS_FILE}" || echo "$info" >> "${FAILED_JOBS_FILE}"
        unset gpu_job_pids[$idx] gpu_job_info[$idx]
        gpu_job_pids=("${gpu_job_pids[@]}"); gpu_job_info=("${gpu_job_info[@]}")
        break
      fi
    done
    sleep 1
  done
}

wait_for_cpu_slot() {
  local idx
  while [ ${#cpu_job_pids[@]} -ge $CPU_MAX_PARALLEL ]; do
    for idx in "${!cpu_job_pids[@]}"; do
      if ! kill -0 "${cpu_job_pids[$idx]}" 2>/dev/null; then
        wait "${cpu_job_pids[$idx]}"; local ec=$?
        local info="${cpu_job_info[$idx]}"
        [ $ec -eq 0 ] && echo "$info" >> "${COMPLETED_JOBS_FILE}" || echo "$info" >> "${FAILED_JOBS_FILE}"
        unset cpu_job_pids[$idx] cpu_job_info[$idx]
        cpu_job_pids=("${cpu_job_pids[@]}"); cpu_job_info=("${cpu_job_info[@]}")
        break
      fi
    done
    sleep 1
  done
}

echo -e "${GREEN}Starting parallel processing...${NC}"
for ((i=0; i<${#pred_list[@]}; i++)); do
  sub_pred="${pred_list[$i]}"
  chunk_info="$((i+1))/${#pred_list[@]}"
  sub_gt="None"; [ "${GT_WAVSCP}" != "None" ] && sub_gt="${gt_list[$i]}"
  sub_text="";   [ -n "${TEXT_FILE}" ] && sub_text="${text_list[$i]}"

  echo -e "${BLUE}Chunk ${chunk_info}: ${sub_pred}${NC}"
  [ -n "${sub_text}" ] && echo -e "${BLUE}  Text: ${sub_text}${NC}"

  # GPU job
  if $RUN_GPU; then
    wait_for_gpu_slot
    rank=$(get_next_gpu_rank)
    [ "$rank" -eq -1 ] && { echo -e "${RED}No available GPU rank${NC}"; exit 1; }
    gpu_ranks_in_use+=("$rank")

    run_job "gpu" "${sub_pred}" "${sub_gt}" "${sub_text}" \
      "${SCORE_DIR}/result/$(basename "${sub_pred}").result.gpu.txt" "${rank}" &

    pid=$!; gpu_job_pids+=("$pid")
    info="GPU:$pid GPU_RANK:${rank} CHUNK:${chunk_info} FILE:$(basename "${sub_pred}")"
    gpu_job_info+=("$info"); echo "$info" >> "${RUNNING_JOBS_FILE}"
    echo -e "  Started GPU job: PID ${pid} on GPU ${rank}"
  fi

  # CPU job
  if $RUN_CPU; then
    wait_for_cpu_slot
    run_job "cpu" "${sub_pred}" "${sub_gt}" "${sub_text}" \
      "${SCORE_DIR}/result/$(basename "${sub_pred}").result.cpu.txt" &

    pid=$!; cpu_job_pids+=("$pid")
    info="CPU:$pid CHUNK:${chunk_info} FILE:$(basename "${sub_pred}")"
    cpu_job_info+=("$info"); echo "$info" >> "${RUNNING_JOBS_FILE}"
    echo -e "  Started CPU job: PID ${pid}"
  fi
done

echo -e "${YELLOW}Waiting for all jobs to complete...${NC}"
for pid in "${gpu_job_pids[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    wait "$pid"; ec=$?
    for info in "${gpu_job_info[@]}"; do
      [[ "$info" == *":$pid "* ]] && { [ $ec -eq 0 ] && echo "$info" >> "${COMPLETED_JOBS_FILE}" || echo "$info" >> "${FAILED_JOBS_FILE}"; break; }
    done
  fi
done
for pid in "${cpu_job_pids[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    wait "$pid"; ec=$?
    for info in "${cpu_job_info[@]}"; do
      [[ "$info" == *":$pid "* ]] && { [ $ec -eq 0 ] && echo "$info" >> "${COMPLETED_JOBS_FILE}" || echo "$info" >> "${FAILED_JOBS_FILE}"; break; }
    done
  fi
done

echo -e "${GREEN}=== Processing Summary ===${NC}"
if [ -s "${COMPLETED_JOBS_FILE}" ]; then
  completed_count=$(wc -l < "${COMPLETED_JOBS_FILE}")
  echo -e "${GREEN}Completed jobs: ${completed_count}${NC}"
  if $RUN_GPU; then
    echo -e "${GREEN}GPU usage summary:${NC}"
    for ((r=0; r<GPU_COUNT; r++)); do
      cnt=$(grep -c "GPU_RANK:${r}" "${COMPLETED_JOBS_FILE}" 2>/dev/null || true)
      echo -e "  GPU ${r}: ${cnt} jobs completed"
    done
  fi
fi
if [ -s "${FAILED_JOBS_FILE}" ]; then
  failed_count=$(wc -l < "${FAILED_JOBS_FILE}")
  echo -e "${RED}Failed jobs: ${failed_count}${NC}"
  echo -e "${RED}Failed job details:${NC}"
  cat "${FAILED_JOBS_FILE}"
fi

# Merge helper
MERGE_SCRIPT="${SCORE_DIR}/merge_results.sh"
cat > "${MERGE_SCRIPT}" << 'EOF'
#!/usr/bin/env bash
RESULT_DIR=$1
OUTPUT_FILE=$2
if [ $# -ne 2 ]; then
  echo "Usage: $0 <result_dir> <output_file>"; exit 1
fi
echo "Merging results from ${RESULT_DIR} -> ${OUTPUT_FILE}"
if ls "${RESULT_DIR}"/*.result.gpu.txt 1>/dev/null 2>&1; then
  echo "Merging GPU results..."
  cat "${RESULT_DIR}"/*.result.gpu.txt > "${OUTPUT_FILE%.txt}.gpu.txt"
fi
if ls "${RESULT_DIR}"/*.result.cpu.txt 1>/dev/null 2>&1; then
  echo "Merging CPU results..."
  cat "${RESULT_DIR}"/*.result.cpu.txt > "${OUTPUT_FILE%.txt}.cpu.txt"
fi
echo "Done."
EOF
chmod +x "${MERGE_SCRIPT}"

echo -e "${YELLOW}To merge all results, run:${NC}"
echo -e "${MERGE_SCRIPT} ${SCORE_DIR}/result ${SCORE_DIR}/final_results.txt"
echo -e "${GREEN}All processing completed!${NC}"
echo -e "Logs:    ${SCORE_DIR}/logs/"
echo -e "Results: ${SCORE_DIR}/result/"
