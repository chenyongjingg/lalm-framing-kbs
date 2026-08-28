#!/usr/bin/env bash
# =============================================================================
# pipeline.sh — LALM Framing 研究流水线总控（v6.5 · 全新数据版 · 纯本地评分 · KBS 单一目标）
#
# 依据：prompt.md（v6.5，349 行）§1 + §14（执行与后台化强制）
#   - 四层交付物 + 自检 → L → D → P0 → P1-PILOT → [G1] → P1-FULL ∥ P0-C
#     → P2 → P2-C → [G2] → P2-B → F → R，P3 穿插
#   - checkpoint 跳过已完成阶段；PID 管理；僵死自动重启
#   - 退出码语义：0=成功 / 2=部分失败可继续 / 3=致命失败终止
#   - 任一时刻仅一个模型驻留显存；judge 评分只在实验间隙运行
#   - v6.5：评分器/LALM 全家族切换 Gemma-4-E4B/E2B（BF16，QAT 仓库 404），
#     新增异构交叉验证 Qwen2.5-3B（§4.3，不参与主推断）；G1(e) 软化（§5.3）
#
# 部署目标：~/lalm_framing_revision_v6/
# 启动：tmux new-session -d -s revision_v6 \
#         'bash ~/lalm_framing_revision_v6/pipeline.sh 2>&1 | tee logs/pipeline_main.log'
# =============================================================================
set -uo pipefail

ROOT="${LALM_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT" || exit 3
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
CONFIG="${LALM_CONFIG:-$ROOT/pipeline_config.yaml}"
PYTHON="${PYTHON_BIN:-/root/.venv/bin/python}"

# --- 禁用 torch.compile / torchdynamo（v6.2 修复 2026-08-04）---
# 评分阶段 StrongREJECT fallback 加载完成后，transformers 5.14.1 内部触发 torch.compile，
# 产生 16 个 compile_worker 并死锁 14h+（主进程 CPU 100% 忙等、零产出、僵死检测不触发）。
# 禁用后 compile 退化为 eager，功能等价，仅牺牲编译加速。
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_COMPILE_WORKER_TIMEOUT=600

# --- 单实例锁（防双 pipeline 并发 OOM）---
LOCK_FILE="$ROOT/logs/pipeline.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[$(date '+%F %T')] 另一个 pipeline.sh 实例已在运行（$LOCK_FILE 被占用），本次退出。"
  exit 0
fi
echo $$ > "$ROOT/logs/pipeline.pid"
trap 'rm -f "$ROOT/logs/pipeline.pid"' EXIT

# --- 目录 ---
mkdir -p logs checkpoints results responses models report gold gates run data figures artifact
FAILED_FILE="logs/FAILED"
: > "$FAILED_FILE"
T0=$(date +%s)

log() { echo "[$(date '+%F %T')] $*"; }

# --- 复用工具库（若 v4 存在则复制；v6 独立运行）---
if [ -d "$ROOT/../lalm_framing_revision_v4" ]; then
  V4_ROOT="$ROOT/../lalm_framing_revision_v4"
  for f in common_utils.py scorer_utils.py agreement_utils.py model_cache.py; do
    if [ -f "$V4_ROOT/$f" ] && [ ! -f "$ROOT/$f" ]; then
      cp "$V4_ROOT/$f" "$ROOT/$f" 2>/dev/null && log "复用工具库: $f"
    fi
  done
fi

# =============================================================================
# 部署自检：编译全部 Python 脚本
# =============================================================================
log "部署自检：编译 Python 脚本..."
SELFTEST_OK=1
for f in common_utils.py scorer_utils.py agreement_utils.py model_cache.py \
         stage_l_novelty.py stage_d_build_data.py stage_p0_measure.py \
         stage_p1_pilot.py stage_p1_full.py stage_p0c.py stage_p2_msrf.py \
         stage_p2c_adaptive.py stage_p2b.py stage_f_figures.py stage_p3.py \
         stage_p2_baselines.py stage_r_artifact.py gate_g1.py gate_g2.py \
         stage_p1_pilot_refill_dual.py stage_p0_dual_refill.py recalc_v64.py \
         freeze_pilot.py; do
  if [ -f "$ROOT/$f" ]; then
    if ! "$PYTHON" -m py_compile "$f" 2>>logs/selftest.err; then
      log "  ✗ $f 编译失败，见 logs/selftest.err"
      SELFTEST_OK=0
    fi
  else
    log "  ⚠ $f 尚未提供（该阶段将跳过）"
  fi
done
if [ "$SELFTEST_OK" -ne 1 ]; then
  log "部署自检未通过，流水线中止。请修复后重启。"
  exit 3
fi
log "部署自检通过"

# =============================================================================
# 心跳（每 10 分钟写 GPU/显存/阶段/进度/PID 到 logs/heartbeat.log）
# =============================================================================
HEARTBEAT_PID=""
start_heartbeat() {
  for pid in $(pgrep -f 'heartbeat.sh' 2>/dev/null); do
    [ "$pid" != "$$" ] && kill -9 "$pid" 2>/dev/null && log "清理旧心跳 pid=$pid"
  done
  nohup bash "$ROOT/heartbeat.sh" > logs/heartbeat_nohup.log 2>&1 < /dev/null &
  HEARTBEAT_PID=$!
  echo "$HEARTBEAT_PID" > logs/heartbeat.pid
  log "心跳已启动 (pid=$HEARTBEAT_PID, 每 10 分钟)"
}
start_heartbeat
# v6.5.28-fix（第八轮审查 🟡）：合并 EXIT 清理——原第 42 行的 `rm -f pipeline.pid`
# trap 被本行覆盖 → pipeline.pid 永不清理。单 trap 同时删 pid + 停心跳。
trap 'rm -f "$ROOT/logs/pipeline.pid"; log "停止心跳..."; [ -n "$HEARTBEAT_PID" ] && kill "$HEARTBEAT_PID" 2>/dev/null' EXIT

# =============================================================================
# 阶段执行器（checkpoint 跳过 + 退出码 0/2/3 + 僵死检测重启）
# =============================================================================
run_stage() {
  local name="$1"; shift
  local script="$1"; shift
  local extra_args=("$@")   # 透传阶段参数（如 --infer --evaluate）
  local marker="run/${name}.complete"
  local pidfile="run/${name}.pid"
  local timeout_sec="${STAGE_TIMEOUT:-172800}"
  local stage_t0=$(date +%s)   # v6.5.28-fix（阶段内计时，非流水线全局 T0）
  local max_retry=3   # v6.5.3：对齐提示词"失败重试 3 次"（原 2 次，上轮审查勘误）

  if [ -f "$marker" ]; then
    log "[SKIP] $name 已完成（$marker 存在）"
    return 0
  fi
  if [ ! -f "$ROOT/$script" ]; then
    log "[SKIP] $name：$script 未提供"
    echo "$name SKIPPED_MISSING_SCRIPT $(date '+%F %T')" >> "$FAILED_FILE"
    # v6.5.28-fix（纪律 #2）：跳过必须落盘 errors.jsonl（原只写 FAILED_FILE）
    mkdir -p logs
    printf '{"ts":"%s","stage":"%s","event":"skipped_missing_script","note":"%s 脚本未提供，阶段跳过（须在报告披露）"}\n' \
      "$(date '+%Y-%m-%dT%H:%M:%S')" "$name" "$script" >> logs/errors.jsonl
    return 0
  fi

  # v6.6.3-fix 2026-08-05：若已有同名进程在跑（外部启动，如手动 nohup），
  # 主链等待其完成而非重拉——防止双进程并发 GPU OOM 互踩（08-05 实锤）。
  if [ -f "$pidfile" ]; then
    local old_pid=$(cat "$pidfile" 2>/dev/null)
    if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
      log "[WAIT] $name 已有进程（pid=$old_pid）在运行，等待其完成（不重拉）"
      while kill -0 "$old_pid" 2>/dev/null; do
        sleep 60
      done
      if [ -f "$marker" ]; then
        log "[ OK ] $name（外部进程已完成）"
        return 0
      fi
      log "[WAIT] $name 外部进程已退出但无 marker，进入正常重试"
    fi
  fi

  # 阶段间清理（GPU 显存释放）
  "$PYTHON" - <<'EOF' 2>/dev/null || true
import torch, gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
EOF
  rm -rf /dev/shm/lalm_model_cache/* /tmp/lalm_model_cache/* 2>/dev/null

  local attempt=1
  while [ $attempt -le $max_retry ]; do
    # v6.5.28-fix（第三轮审查）：stage_t0 必须在 attempt 循环内重置——
    # 原循环外只设一次，超时 kill 后 attempt 2/3 的 elapsed 已 > timeout，
    # 首轮即被再 kill，重试形同虚设（白耗并误标阶段失败）。
    local stage_t0=$(date +%s)
    log "[RUN ] $name (attempt $attempt/$max_retry): $PYTHON $script --config $CONFIG --resume ${extra_args[*]:-}"
    if [ -f "logs/${name}.log" ]; then
      mv "logs/${name}.log" "logs/${name}.log.bak_$(date '+%H%M%S')" 2>/dev/null || true
    fi
    # 注意：${extra_args[@]} 为空数组时展开为 0 个参数（不能用 ${extra_args[@]:-}，会展开成 1 个空字符串参数）
    if [ ${#extra_args[@]} -gt 0 ]; then
      "$PYTHON" "$script" --config "$CONFIG" --resume "${extra_args[@]}" \
          >> "logs/${name}.log" 2>&1 &
    else
      "$PYTHON" "$script" --config "$CONFIG" --resume \
          >> "logs/${name}.log" 2>&1 &
    fi
    local pid=$!
    echo "$pid" > "$pidfile"
    log "  $name PID=$pid"

    local last_mtime=0 last_alert=0 last_cputime="" last_outcount=""
    while kill -0 "$pid" 2>/dev/null; do
      local mtime age cputime state
      mtime=$(stat -c %Y "logs/${name}.log" 2>/dev/null || echo 0)
      age=$(( $(date +%s) - mtime ))
      cputime=$(ps -o time= -p "$pid" 2>/dev/null | awk -F: '{printf "%d", $1*3600+$2*60+$3}' 2>/dev/null || echo 0)
      state=$(ps -o stat= -p "$pid" 2>/dev/null | cut -c1 || echo X)
      local cputime_changed=0
      [ -n "$cputime" ] && [ "$cputime" != "$last_cputime" ] && cputime_changed=1
      last_cputime=$cputime
      # v6.6.6-fix 2026-08-05：产出活跃检查——日志静默但实际产出（文件数）增长
      # 不算僵死。实锤：p1_pilot TTS 阶段 tqdm 不落盘 + stdout 行缓冲 → 日志 30min
      # 静默是正常现象，但 wav 文件持续增长；15:53 watchdog 误判 kill 掉 attempt2
      # （wav 15:22→15:53 从 2242→2374 正常增长 132 条）。对 TTS/推理等产出型阶段，
      # 文件数增长即活跃证据；仅对"无产出型阶段"（纯计算）保留日志+CPU 判据。
      # v6.6.7-fix 2026-08-06：产出统计 = wav 数 + 响应 jsonl 行数之和（任一增长即活跃）。
      #   原实现只数 wav：推理阶段 wav 全量生成后恒 7200 不增长，响应 jsonl 增长被无视，
      #   out_growth 恒 0 → 活跃判据失效（仅靠 CPU 时间兜底）。p1_pilot 19:29 后日志静默
      #   13.5h 但响应持续增长，实锤该缺陷（R27 巡检）。
      local out_count="" out_growth=0
      case "$name" in
        p1_pilot|p0c|p1_full|p2|p2c|p2b|p0|p2c_acoustic|p2b_infer|p2b_score|p2b_frontier|p2_lora|p2baseline|p2baseline_infer|p2baseline_gradsafe)
          # v6.5.12-fix 2026-08-08：子阶段目录名不是 ${name^^}（p2c_acoustic→P2C
          # 而非 P2C_ACOUSTIC），统一按实际响应目录统计。
          # 原 case 只覆盖主阶段名，p2c_acoustic（TTS 合成）等子阶段走 * 分支
          # 无产出活跃检查 → 日志静默 30min 时仅有 CPU 判据，TTS 转码空窗
          # （wav 短暂不增长）存在误杀风险（08-07 attempt1 实证过同类误判）。
          case "$name" in
            p1_pilot) out_dir="P1_PILOT" ;;
            p0c) out_dir="P0C" ;;
            p1_full) out_dir="P1_FULL" ;;
            p2|p2_lora) out_dir="P2" ;;
            # v6.5.28-fix（第三轮审查）：p2c_acoustic 实际写 responses/P2C/
            # audio_acoustic/ + attacks_acoustic_disguise_audio_*.jsonl（原错
            # 映射 P0C → 产出活跃检查对 P2C 族失效，日志静默时仅 CPU 判据）。
            p2c|p2c_acoustic) out_dir="P2C" ;;
            p2b|p2b_infer|p2b_score|p2b_frontier) out_dir="P2B" ;;
            p0) out_dir="P0" ;;
            # v6.5.23-fix（问题 96）：stage_p2_baselines.py 实际输出
            # responses/P2B/shieldgemma_scores.jsonl（非 P2），原 out_dir="P2"
            # → 产出活跃检查统计错目录，out_growth 恒 0，僵死判据退化
            p2baseline|p2baseline_infer|p2baseline_gradsafe) out_dir="P2B" ;;
            *) out_dir="" ;;
          esac
          out_count=$(ls "${ROOT}/responses/${out_dir}/audio/"*.wav 2>/dev/null | wc -l)
          # 响应行数累加（wav 数 + 响应行数之和；任一增长即活跃）
          for rf in "${ROOT}/responses/${out_dir}/"*_responses.jsonl; do
            [ -f "$rf" ] || continue
            n=$(wc -l < "$rf" 2>/dev/null || echo 0)
            out_count=$((out_count + n))
          done
          # v6.5.23-fix（问题 96）：shieldgemma_scores.jsonl 为 append 单文件，
          # 不匹配 *_responses.jsonl glob → 单独统计（P2-BASELINE 活跃判据）
          for sf in "${ROOT}/responses/${out_dir}/"shieldgemma_scores.jsonl; do
            [ -f "$sf" ] || continue
            n=$(wc -l < "$sf" 2>/dev/null || echo 0)
            out_count=$((out_count + n))
          done
          # v6.5.28-fix（第三轮审查）：P2-C 的 attacks_*.jsonl（不匹配
          # *_responses.jsonl）+ audio_acoustic/*.wav 单独统计——p2c/p2c_acoustic
          # 产出活跃判据（原恒 0，日志静默时仅 CPU 判据兜底）。
          case "$name" in
            p2c|p2c_acoustic)
              for _af in "${ROOT}/responses/P2C/"attacks_*.jsonl; do
                [ -f "$_af" ] || continue
                _n=$(wc -l < "$_af" 2>/dev/null || echo 0)
                out_count=$((out_count + _n))
              done
              _awc=$(ls "${ROOT}/responses/P2C/audio_acoustic/"*.wav 2>/dev/null | wc -l)
              out_count=$((out_count + _awc))
              ;;
          esac
          ;;
        *)
          out_count=""
          ;;
      esac
      if [ -n "$out_count" ] && [ -n "$last_outcount" ] \
         && [ "$out_count" != "$last_outcount" ] 2>/dev/null; then
        out_growth=1
      fi
      last_outcount=$out_count
      # 僵死判定：日志超 30 分钟 + CPU 时间未增长 + 进程非 D 状态 + 无产出增长
      if [ "$age" -gt 1800 ] && [ "$cputime_changed" -eq 0 ] && [ "$state" != "D" ] \
         && [ "$out_growth" -eq 0 ]; then
        log "  ⚠ $name 日志 ${age}s 未更新 且 CPU 时间未增长 且非I/O等待 且无产出增长，疑似僵死（pid=$pid），kill 重启"
        last_alert=$age
        kill -9 "$pid" 2>/dev/null
        break
      fi
      local elapsed=$(( $(date +%s) - stage_t0 ))
      if [ "$elapsed" -gt "$timeout_sec" ]; then
        log "  ⚠ $name 超时 ${timeout_sec}s（阶段内计时，v6.5.28-fix），kill"
        kill -9 "$pid" 2>/dev/null
        break
      fi
      sleep 60
    done
    wait "$pid" 2>/dev/null
    local code=$?
    rm -f "$pidfile"
    log "  $name 进程退出，code=$code"

    if [ "$code" -eq 0 ]; then
      touch "$marker"
      log "[ OK ] $name"
      return 0
    elif [ "$code" -eq 2 ]; then
      touch "$marker"
      # v6.6.5-fix 2026-08-05：code=2（部分失败）必须真实落盘 errors.jsonl（纪律#2）
      # 原实现只打印"已记 errors.jsonl"但从未写入——声明与事实不符，违反最高纪律#2
      mkdir -p logs
      printf '{"ts":"%s","stage":"%s","event":"partial_failed","code":2,"note":"部分失败，阶段继续","attempt":%d}\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S')" "$name" "$attempt" >> logs/errors.jsonl
      log "[PARTIAL] $name（部分失败，已记 errors.jsonl，继续）"
      return 2
    else
      echo "$name code=$code $(date '+%F %T')" >> "$FAILED_FILE"
      # v6.6.5-fix 2026-08-05：失败必须真实落盘 errors.jsonl（纪律#2）
      mkdir -p logs
      printf '{"ts":"%s","stage":"%s","event":"stage_failed","code":%d,"attempt":%d,"max_retry":%d}\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S')" "$name" "$code" "$attempt" "$max_retry" >> logs/errors.jsonl
      log "[FAIL] $name（code=$code）"
      if [ "$code" -eq 3 ]; then
        log "  $name 致命失败，终止该分支"
        return 3
      fi
      if [ $attempt -ge $max_retry ]; then
        log "  $name 已达最大重试，继续下一阶段"
        return 1
      fi
      attempt=$((attempt + 1))
      sleep 30
    fi
  done
  return 1
}


# =============================================================================
# 阶段返回值处理（v6.7-hotfix: 替代 `|| true`，防止静默吞噬致命错误）
# 0=成功/2=部分(继续)/非0非2=失败(记录)/3=致命(终止)
# =============================================================================
handle_stage_rc() {
  local name="$1"; local rc="$2"
  if [ "$rc" -eq 0 ] || [ "$rc" -eq 2 ]; then
    return 0
  elif [ "$rc" -eq 3 ]; then
    log "[FATAL] $name 返回 code=3 致命错误，流水线终止"
    exit 3
  else
    log "[WARN] $name 返回 code=$rc (非致命)，流水线继续（已记录 FAILED）"
    return 0
  fi
}

# =============================================================================
# 闸门执行器
# v6.5.18-fix（问题 54）：原实现对 gate 返回 1（判定不通过）执行 `return 0`
# 将信号掩码为"通过"——G1_RC/G2_RC 恒 0 或 3，主链永远无法识别"不通过 →
# 探索性叙事/机制主导版"分支，且不通过无任何落盘（违反最高纪律#2 无静默丢失）。
# 修复：0=通过 / 1=不通过（如实透传，主链按 §5/§8 协议继续但不误报通过）/
#       3=无法判定（致命）。不通过同时写入 logs/errors.jsonl（纪律#2）。
# =============================================================================
run_gate() {
  local gate="$1"; shift
  local extra=("$@")   # v6.5.30-fix (AUDIT #186)：透传闸门额外参数（如 --effects）
  local script="gate_${gate}.py"
  # v6.5.28-fix（第八轮审查 🔴）：gate 脚本写入 gates/G1.json / G2.json（大写，
  # gate_g1.py:252 / gate_g2.py:240），原 `gates/${gate}.json`（g1 小写）在 Linux
  # 大小写敏感下读不到 → passed="" → 闸门真实通过也被误判"不通过"，主链误走
  # 探索性/机制主导分支 + 误报 FAILED。统一大写（${gate^^}）。
  local json="gates/${gate^^}.json"
  if [ ! -f "$ROOT/$script" ]; then
    log "[GATE] $gate：$script 未提供"
    # v6.5.28-fix（第三轮审查）：run_gate 缺脚本必须落盘 errors.jsonl 并返回
    # 非 0（原 return 0 使 G1/G2 缺失被当作"通过"，REVISION_REPORT 误报闸门
    # 通过——与 run_stage 缺脚本落盘不一致，纪律 #2/#3）。
    mkdir -p logs
    printf '{"ts":"%s","stage":"gate_%s","event":"gate_missing_script","note":"%s 未提供，闸门未判定（不得误报通过）"}\n' \
      "$(date '+%Y-%m-%dT%H:%M:%S')" "$gate" "$script" >> logs/errors.jsonl
    echo "$gate MISSING_SCRIPT $(date '+%F %T')" >> "$FAILED_FILE"
    return 2
  fi
  log "[GATE] $gate 判定中..."
  "$PYTHON" "$script" --config "$CONFIG" "${extra[@]}" >> "logs/${gate}.log" 2>&1
  local code=$?
  if [ "$code" -eq 3 ]; then
    log "[GATE] $gate 无法判定（code=3，致命）"
    return 3
  elif [ "$code" -ne 0 ]; then
    # 判定不通过：如实透传（返回 1），不再掩码为 0；落盘 errors.jsonl
    log "[GATE] $gate 判定为不通过（code=$code，按 §5/§8 协议转探索性叙事/机制主导版）"
    mkdir -p logs
    printf '{"ts":"%s","stage":"gate_%s","event":"gate_not_passed","code":%d,"note":"闸门判定不通过，主链按协议继续但不误报通过"}\n' \
      "$(date '+%Y-%m-%dT%H:%M:%S')" "$gate" "$code" >> logs/errors.jsonl
    return 1
  fi
  # code=0 → 读取 passed 字段确认（防 gate 返回 0 但 JSON 未写 passed 的异常）
  local passed
  passed=$(grep -o '"passed": *[a-z]*' "$json" 2>/dev/null | head -1 | awk '{print $2}')
  log "[GATE] $gate = $passed"
  if [ "$passed" = "true" ]; then
    return 0
  else
    log "[GATE] $gate 返回 0 但 passed 字段缺失/非 true（数据异常，按不通过处理）"
    printf '{"ts":"%s","stage":"gate_%s","event":"gate_json_anomaly","code":0,"note":"passed 字段缺失/非 true"}\n' \
      "$(date '+%Y-%m-%dT%H:%M:%S')" "$gate" >> logs/errors.jsonl
    return 1
  fi
}

# =============================================================================
# 双 GPU 显存检查 + 并行执行器（v6.5-D2-8 双卡隔离调度，2026-08-11）
# =============================================================================
# 查询指定 GPU 空闲显存（MiB）；$1=GPU 索引（0/1），默认 0
gpu_free_mb() {
  local gpu_idx="${1:-0}"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    --id="$gpu_idx" 2>/dev/null | head -1 | tr -d ' '
}

# GPU 数量检测
gpu_count() {
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' '
}

# 双 GPU 隔离执行器：在指定 GPU 上跑阶段（CUDA_VISIBLE_DEVICES 隔离）
# 用法: run_stage_on_gpu <gpu_id> <stage_name> <script> [extra_args...]
# 与 run_stage 相同语义，但增加 CUDA_VISIBLE_DEVICES 环境变量。

# M5-fix（AUDIT #172）：阶段产出计数（wav 数 + *_responses.jsonl 行数 +
# shieldgemma_scores.jsonl + P2C attacks_*.jsonl 特判），供双卡僵死检测。
# 与 run_stage 内联逻辑同源（GPU 推理阶段日志常静默但产出增长，产出增长即活跃）。
_stage_output_count() {
  local name="$1" out_dir="" n=0
  case "$name" in
    p1_pilot) out_dir="P1_PILOT" ;;
    p0c) out_dir="P0C" ;;
    p1_full) out_dir="P1_FULL" ;;
    p2|p2_lora) out_dir="P2" ;;
    p2c|p2c_acoustic) out_dir="P2C" ;;
    p2b|p2b_infer|p2b_score|p2b_frontier) out_dir="P2B" ;;
    p0) out_dir="P0" ;;
    p2baseline|p2baseline_infer|p2baseline_gradsafe) out_dir="P2B" ;;
    *) echo ""; return 0 ;;
  esac
  n=$(ls "${ROOT}/responses/${out_dir}/audio/"*.wav 2>/dev/null | wc -l)
  for rf in "${ROOT}/responses/${out_dir}/"*_responses.jsonl; do
    [ -f "$rf" ] || continue
    n=$((n + $(wc -l < "$rf" 2>/dev/null || echo 0)))
  done
  for sf in "${ROOT}/responses/${out_dir}/"shieldgemma_scores.jsonl; do
    [ -f "$sf" ] || continue
    n=$((n + $(wc -l < "$sf" 2>/dev/null || echo 0)))
  done
  case "$name" in
    p2c|p2c_acoustic)
      for _af in "${ROOT}/responses/P2C/"attacks_*.jsonl; do
        [ -f "$_af" ] || continue
        n=$((n + $(wc -l < "$_af" 2>/dev/null || echo 0)))
      done
      n=$((n + $(ls "${ROOT}/responses/P2C/audio_acoustic/"*.wav 2>/dev/null | wc -l)))
      ;;
  esac
  echo "$n"
}

run_stage_on_gpu() {
  local gpu_id="$1"; shift
  local name="$1"; shift
  local script="$1"; shift
  local extra_args=("$@")
  local marker="run/${name}.complete"
  local pidfile="run/${name}.pid"
  local max_retry=3

  if [ -f "$marker" ]; then
    log "[SKIP] $name 已完成（$marker 存在）[GPU $gpu_id]"
    return 0
  fi
  if [ ! -f "$ROOT/$script" ]; then
    log "[SKIP] $name：$script 未提供 [GPU $gpu_id]"
    echo "$name SKIPPED_MISSING_SCRIPT $(date '+%F %T')" >> "$FAILED_FILE"
    printf '{"ts":"%s","stage":"%s","event":"skipped_missing_script","gpu":%s}\n' \
      "$(date '+%Y-%m-%dT%H:%M:%S')" "$name" "$gpu_id" >> logs/errors.jsonl
    return 0
  fi

  if [ -f "$pidfile" ]; then
    local old_pid=$(cat "$pidfile" 2>/dev/null)
    if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
      log "[WAIT] $name 已有进程（pid=$old_pid）[GPU $gpu_id]，等待其完成"
      while kill -0 "$old_pid" 2>/dev/null; do sleep 60; done
      if [ -f "$marker" ]; then
        log "[ OK ] $name（外部进程已完成）[GPU $gpu_id]"
        return 0
      fi
    fi
  fi

  local attempt=1
  while [ $attempt -le $max_retry ]; do
    local stage_t0=$(date +%s)
    log "[RUN ] $name (attempt $attempt/$max_retry) [GPU $gpu_id]: $PYTHON $script"
    if [ -f "logs/${name}.log" ]; then
      mv "logs/${name}.log" "logs/${name}.log.bak_$(date '+%H%M%S')" 2>/dev/null || true
    fi
    if [ ${#extra_args[@]} -gt 0 ]; then
      CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON" "$script" --config "$CONFIG" --resume "${extra_args[@]}" \
          >> "logs/${name}.log" 2>&1 &
    else
      CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON" "$script" --config "$CONFIG" --resume \
          >> "logs/${name}.log" 2>&1 &
    fi
    local pid=$!
    echo "$pid" > "$pidfile"
    # M5-fix（AUDIT #172）：僵死检测（与 run_stage 同逻辑）——原等待循环仅
    # STAGE_TIMEOUT 超时检查，进程死锁/静默空转时永不触发，双卡槽位被占用至
    # 全局超时（M5 双卡无僵死检测）。判定：日志 >30min 未更新 + CPU 时间未增长
    # + 非 D(I/O) 状态 + 无产出增长 → kill 重启（GPU 推理阶段日志常静默但产出
    # 增长，产出计数来自 _stage_output_count，任一信号活跃即不判僵死）。
    local last_cputime="" last_outcount=""
    local elapsed=0
    while kill -0 "$pid" 2>/dev/null; do
      sleep 30
      elapsed=$(( $(date +%s) - stage_t0 ))
      local cputime state out_count
      cputime=$(ps -o time= -p "$pid" 2>/dev/null | awk -F: '{printf "%d", $1*3600+$2*60+$3}' 2>/dev/null || echo 0)
      state=$(ps -o stat= -p "$pid" 2>/dev/null | cut -c1 || echo X)
      out_count=$(_stage_output_count "$name")
      local cputime_changed=0 out_growth=0
      [ -n "$cputime" ] && [ "$cputime" != "$last_cputime" ] && cputime_changed=1
      last_cputime=$cputime
      if [ -n "$out_count" ] && [ -n "$last_outcount" ] \
         && [ "$out_count" != "$last_outcount" ] 2>/dev/null; then
        out_growth=1
      fi
      last_outcount=$out_count
      if [ $(( $(date +%s) - $(stat -c %Y "logs/${name}.log" 2>/dev/null || echo 0) )) -gt 1800 ] \
         && [ "$cputime_changed" -eq 0 ] && [ "$state" != "D" ] && [ "$out_growth" -eq 0 ]; then
        log "  ⚠ $name [GPU $gpu_id] 疑似僵死（日志>30min + CPU 未增长 + 非I/O + 无产出），kill pid=$pid"
        kill -9 "$pid" 2>/dev/null || true
        break
      fi
      if [ $elapsed -ge "${STAGE_TIMEOUT:-172800}" ]; then
        log "[TIME] $name 超时 ${elapsed}s，kill pid=$pid [GPU $gpu_id]"
        kill -9 "$pid" 2>/dev/null || true
        break
      fi
    done
    if [ -f "$marker" ]; then
      log "[ OK ] $name [GPU $gpu_id]"
      return 0
    fi
    # M-fix (KBS 2026-08-20): 删除 || true -- 超时/僵死 kill 后 wait 已 reap 进程，外层
    # wait 返回 127 被 || true 掩盖为 0 → 超时阶段误判 rc=0 并 touch marker（P1-FULL 92%
    # / P0-C 76.8% 事故根因）。去掉后捕获真实 rc（kill 后为 137）→ 走 RETRY 续跑，与 run_stage 语义一致。
    wait "$pid" 2>/dev/null
    local rc=$?
    if [ $rc -eq 0 ]; then
      touch "$marker"
      log "[ OK ] $name (rc=0) [GPU $gpu_id]"
      return 0
    elif [ $rc -eq 2 ]; then
      # M4-fix（AUDIT #172）：code=2（部分失败）语义与 run_stage 一致——
      # 原 run_stage_on_gpu 无 code=2 分支：部分完成被当失败无限重试（max_retry
      # 耗尽 return 3），且从不 touch marker → 双卡槽位卡死/下游误判。修复：
      # touch marker + 真实落盘 errors.jsonl + return 2（上游 DAG 继续不重试）。
      touch "$marker"
      mkdir -p logs
      printf '{"ts":"%s","stage":"%s","event":"partial_failed","code":2,"note":"部分失败，阶段继续","gpu":%s,"attempt":%d}\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S')" "$name" "$gpu_id" "$attempt" >> logs/errors.jsonl
      log "[PARTIAL] $name（部分失败，已记 errors.jsonl，继续）[GPU $gpu_id]"
      return 2
    fi
    log "[RETRY] $name rc=$rc attempt=$attempt/$max_retry [GPU $gpu_id]"
    echo "$name rc=$rc attempt=$attempt $(date '+%F %T')" >> "$FAILED_FILE"
    attempt=$((attempt + 1))
    sleep 10
  done
  log "[FAIL] $name 耗尽重试次数（$max_retry）[GPU $gpu_id]"
  return 3
}

# 双 GPU 并行启动器：同时在 GPU A 和 GPU B 上跑两个阶段，等待两者完成
# 用法（无额外参数）:
#   launch_dual <name_a> <script_a> <gpu_a> ::: <name_b> <script_b> <gpu_b>
# 用法（带每 GPU 额外参数）:
#   launch_dual <name_a> <script_a> <gpu_a> <args_a...> ::: <name_b> <script_b> <gpu_b> <args_b...>
# ":::" 为左右参数组分隔符。
# 返回: 0=全部成功 / 2=部分失败 / 3=全部致命失败
# v6.5-D2-9 修复（2026-08-11）：原实现仅接受 6 个位置参数，额外参数被静默丢弃，
#   导致 ShieldGemma∥WildGuard 双卡路径不带 --infer/--wildguard-infer 标志运行。
launch_dual() {
  # --- 解析左侧参数 ---
  local name_a="$1" script_a="$2" gpu_a="$3"; shift 3
  local args_a=()
  while [ $# -gt 0 ]; do
    if [ "$1" = ":::" ]; then
      shift
      break
    fi
    args_a+=("$1")
    shift
  done

  # --- 解析右侧参数 ---
  local name_b="$1" script_b="$2" gpu_b="$3"; shift 3
  local args_b=()
  while [ $# -gt 0 ]; do
    args_b+=("$1")
    shift
  done

  local rc_a=0 rc_b=0 pid_a="" pid_b=""

  log "[DUAL] 双卡并行: $name_a [GPU $gpu_a] ∥ $name_b [GPU $gpu_b]"

  # 启动 GPU A 上的阶段（后台）
  if [ ${#args_a[@]} -gt 0 ]; then
    run_stage_on_gpu "$gpu_a" "$name_a" "$script_a" "${args_a[@]}" &
  else
    run_stage_on_gpu "$gpu_a" "$name_a" "$script_a" &
  fi
  pid_a=$!

  # 启动 GPU B 上的阶段（后台）
  if [ ${#args_b[@]} -gt 0 ]; then
    run_stage_on_gpu "$gpu_b" "$name_b" "$script_b" "${args_b[@]}" &
  else
    run_stage_on_gpu "$gpu_b" "$name_b" "$script_b" &
  fi
  pid_b=$!

  # 等待两者完成
  wait "$pid_a" 2>/dev/null; rc_a=$?
  wait "$pid_b" 2>/dev/null; rc_b=$?

  log "[DUAL] 完成: $name_a rc=$rc_a | $name_b rc=$rc_b"

  # 退出码合并逻辑
  if [ "$rc_a" -eq 0 ] && [ "$rc_b" -eq 0 ]; then
    return 0
  elif [ "$rc_a" -eq 3 ] && [ "$rc_b" -eq 3 ]; then
    return 3
  else
    return 2
  fi
}

# =============================================================================
# 模型就绪等待（v6.5：P0 双 judge 依赖 Gemma-4-E4B/E2B；下载完成前不启动评分阶段）
# =============================================================================
wait_for_models() {
  local max_wait="${MODEL_WAIT_SEC:-21600}"  # 默认最长 6h
  local waited=0
  local miss=1
  while [ "$waited" -lt "$max_wait" ]; do
    miss=0
    # Gemma-4-E4B 权重（本地 /root/models，v6.5 主评分器）
    ls /root/models/gemma-4-E4B-it/model.safetensors \
        >/dev/null 2>&1 || miss=1
    # Gemma-4-E2B 权重（v6.5 轻量评分器）
    ls /root/models/gemma-4-E2B-it/model.safetensors \
        >/dev/null 2>&1 || miss=1
    if [ "$miss" -eq 0 ]; then
      log "模型就绪（Gemma-4-E4B/E2B 权重齐全），等待 ${waited}s 后继续"
      return 0
    fi
    if [ $((waited % 600)) -eq 0 ]; then
      log "等待模型下载中（已 ${waited}s）: $(du -sh /root/models/gemma-4-E4B-it 2>/dev/null | cut -f1) / $(du -sh /root/models/gemma-4-E2B-it 2>/dev/null | cut -f1)"
    fi
    sleep 60
    waited=$((waited + 60))
  done
  log "⚠ 模型等待超时 ${max_wait}s，继续（P0 将降级处理）"
  return 0
}

# =============================================================================
# 主 DAG（STAGE_CONTRACTS §2）
# =============================================================================
log "=============================================="
log " LALM Framing 研究流水线 v6.5 启动（全新数据版 · KBS 单一目标 · Gemma 4 家族）"
log " ROOT=$ROOT  CONFIG=$CONFIG"
log "=============================================="

# [L] 新颖性 + 引用核验（最先执行）
run_stage l stage_l_novelty.py; handle_stage_rc l $?

# [D] 数据从零构建
run_stage d stage_d_build_data.py; handle_stage_rc d $?

# [模型就绪等待]（L/D 不依赖评分器权重；P0 双 judge 需要 Gemma-4-E4B/E2B）
# v6.5.24-fix：原注释"不依赖 Mistral"为 v6.4 残留（v6.5 双 judge = Gemma-4 家族）
wait_for_models

# [P0] 全自动测量体系（4 评分器 + 异构交叉验证 + 公开基准验证；v6.5.17 修正原"6 评分器"旧口径注释）
run_stage p0 stage_p0_measure.py; handle_stage_rc p0 $?

# [P1-PILOT] 配对析因预实验
run_stage p1_pilot stage_p1_pilot.py; handle_stage_rc p1_pilot $?

# [方案 A: DualJudge 补跑]（v6.5：Gemma-4-E4B/E2B 两轮）
# 历史遗留：旧主进程跑旧 scorer_utils（.device 修复前）→ dual_judge 全 None。
# v6.5 起主进程直接 4 票制落盘，此段通常跳过；仅当 scored.parquet 存在且
# judge 列全空时补跑（E4B/E2B 两票写回）。
if [ -f "$ROOT/results/p1_pilot_scored.parquet" ] && \
   [ ! -f "$ROOT/results/p1_pilot_dual_refill.done" ]; then
  log "[方案A] parquet 已落盘 → DualJudge 补跑（E4B+E2B 两票）"
  "$PYTHON" "$ROOT/stage_p1_pilot_refill_dual.py" --config "$CONFIG" \
    >>logs/p1_pilot_refill.log 2>&1
  REFILL_RC=$?
  if [ "$REFILL_RC" -eq 0 ]; then
    touch "$ROOT/results/p1_pilot_dual_refill.done"
    log "[方案A] DualJudge 补跑完成（已写回 parquet）"
  else
    # v6.7-r4-fix 2026-08-07：rc 必须在 $? 之后立即捕获（原实现 $? 已被 [ 吞掉，
    # 显示的是 [ 的返回码而非补跑返回码，日志失真）。纪律 #2：失败如实记录。
    log "[方案A] DualJudge 补跑失败（rc=$REFILL_RC），后续 G1 将披露 dual_judge 缺失"
  fi
else
  log "[方案A] 跳过补跑（parquet 缺失或已补跑）"
fi

# [口径重算兼容段] P1-PILOT 评分落盘后，仅当 effects 非 v6.5 口径才重算（零 GPU，秒级）
# v6.5 起 p1_pilot 直接 4 票制落盘并写 version=v6.5 → 此段自动跳过；
# gate_g1 内也有相同 version 检查（双保险）。仅旧口径（v6.4 6 票制）产物触发。
if [ -f "$ROOT/results/p1_pilot_scored.parquet" ]; then
  if "$PYTHON" -c "
import json, os, sys
p = '$ROOT/results/p1_pilot_effects.json'
if not os.path.exists(p):
    sys.exit(1)
d = json.load(open(p, encoding='utf-8'))
sys.exit(0 if d.get('version') == 'v6.5' else 1)
" 2>/dev/null; then
    log "[v6.5] effects 已是 v6.5 口径，跳过重算"
  else
    log "[v6.5] effects 非 v6.5 口径（或缺失）→ 4 票制口径重算"
    "$PYTHON" "$ROOT/recalc_v64.py" --config "$CONFIG" >>logs/pipeline_main.log 2>&1
    log "[v6.5] 重算完成（rc=$?）"
  fi
else
  log "[v6.5] 无 scored.parquet（评分未完成），跳过重算"
fi

# [G1] 闸门
# v6.5.30-fix (AUDIT #186)：FULL 版 G1 判定（用户决策"构造 FULL effects 重判"）。
# 不传 --effects 时 gate_g1.py 默认读 p1_pilot_effects.json（PILOT 8.18pp<10）→
# 恢复 DAG 会判回不通过并覆写 gates/G1.json。显式指向 FULL effects（v6.5 已验证）。
run_gate g1 --effects "$ROOT/results/p1_full_effects.json"
G1_RC=$?
if [ "$G1_RC" -eq 3 ]; then
  log "[FATAL] G1 无法判定（数据缺失），流水线终止"
  exit 3
elif [ "$G1_RC" -ne 0 ]; then
  log "[G1] 闸门未通过（探索性叙事继续——§5 协议）"
  # v6.5.18-fix（问题 54 连带）：不通过如实落盘 FAILED_FILE（纪律#2）
  echo "G1 NOT_PASSED $(date '+%F %T')" >> "$FAILED_FILE"
fi

# [P0-DUAL-REFILL] P0 dual_judge 英文验证补跑（v6.5.20-fix 问题 70）
#   v6.5.20-fix 2026-08-08：原触发条件查 'judge_mistral'（v6.4 Mistral-24B 键），
#   v6.5 双 judge = Gemma-4-E4B/E2B，validation 键为 judge_big/judge_small——
#   'judge_mistral' 永远缺失 → 每次主链重跑误触发补跑脚本，且该脚本读不存在的
#   judge_mistral_model 键（None）→ 加载失败。现改为查 judge_small（v6.5 键）。
#   背景：v6.4 时代 P0 运行中 Mistral-24B 8bit（≈26-30G）超 24G 显存 → dual 段
#   失败，judge_mistral 英文验证缺失。v6.5 双 judge = Gemma-4-E4B/E2B（BF16
#   16G/10G 顺序加载不超 24G），正常不再触发。保留窗口仅当 P0_scorers.json
#   缺 judge_big 或 judge_small 时幂等补跑。
#   注意：E4B 16G + E2B 10.25G 顺序加载不超 24G。
if [ -f "$ROOT/gates/P0_scorers.json" ]; then
  # 精确触发：仅当 validation 段缺 judge_small（E2B，v6.5 键）时补跑
  if ! "$PYTHON" -c "
import json, sys
d = json.load(open('$ROOT/gates/P0_scorers.json', encoding='utf-8'))
val = d.get('validation', {})
sys.exit(0 if 'judge_small' in val else 1)
"; then
    log "[v6.5] P0 validation 缺 judge_small → 补跑窗口（GPU）"
    "$PYTHON" "$ROOT/stage_p0_dual_refill.py" --config "$CONFIG" \
      >>logs/p0_dual_refill.log 2>&1
    log "[v6.5] P0 dual_judge 补跑结束（rc=$?）"
  else
    log "[v6.5] P0 validation 已有 judge_small，跳过补跑"
  fi
else
  log "[v6.5] P0_scorers.json 缺失（P0 未完成），跳过 dual 补跑"
fi

# [P1-FULL ∥ P0-C] 双 GPU 隔离并行（v6.5-D2-8，2026-08-11）
# 原实现：无 CUDA_VISIBLE_DEVICES 隔离 → 两阶段争抢 GPU 0，GPU 1 闲置；
# vram_free_mb 只查 GPU 0 → 双卡环境中 GPU 1 永远不被发现。
# 新实现：查询双卡独立显存，≥16GB 空闲则隔离并行，否则回退串行。
GPU_COUNT=$(gpu_count)
VRAM_G0=$(gpu_free_mb 0)
VRAM_G1=$(gpu_free_mb 1)

if [ "$GPU_COUNT" -ge 2 ] && \
   [ -n "$VRAM_G0" ] && [ "$VRAM_G0" -ge 16000 ] && \
   [ -n "$VRAM_G1" ] && [ "$VRAM_G1" -ge 16000 ]; then
  log "[DAG] P1-FULL ∥ P0-C 双卡隔离并行（GPU0 ${VRAM_G0}MB / GPU1 ${VRAM_G1}MB）"
  launch_dual p1_full stage_p1_full.py 0 ::: p0c stage_p0c.py 1
  RC_DUAL=$?
  RC_P1F=0; RC_P0C=0
  # launch_dual 返回合并码；各阶段 marker 独立判断
  [ -f "run/p1_full.complete" ] || RC_P1F=3
  [ -f "run/p0c.complete" ] || RC_P0C=3
  if [ "$RC_DUAL" -eq 2 ]; then
    log "[DUAL] 部分失败——检查各阶段 marker 决定后续"
  fi
else
  log "[DAG] P1-FULL → P0-C 串行（GPU 不足: GPU0=${VRAM_G0:-?}MB GPU1=${VRAM_G1:-?}MB, 需各≥16GB）"
  run_stage p1_full stage_p1_full.py; RC_P1F=$?
  run_stage p0c stage_p0c.py; RC_P0C=$?
fi
handle_stage_rc p1_full "$RC_P1F"
handle_stage_rc p0c "$RC_P0C"

# [P2-B-INFER] 真实基线推理（v6.5.1 新增 GPU 窗口#1）
#   放在 P1-FULL/P0-C 之后：此时 GPU 重载已结束，显存空闲，是天然的推理窗口。
#   本子阶段执行 --infer-baselines（文本基线 + best_of_n + 降级音频重推理），
#   幂等：已推理条目 checkpoint 自动跳过；再次运行（如中途失败重启）不重复推理。
#   依赖 P1-FULL 评分 parquet（baseline 登记）——P1-FULL 之后必然可用。
#   v6.5.29（裁决补实现 §10.1/§10.2）：同窗口 --score 对 P2B 响应用 HarmBench
#   评分（ASR 与降级分析的真实数据源；--score 幂等，已评分条目跳过）。
log "[v6.5.1] P2-B 真实基线推理窗口（GPU）"
run_stage p2b_infer stage_p2b.py --infer-baselines; handle_stage_rc p2b_infer $?
run_stage p2b_score stage_p2b.py --score; handle_stage_rc p2b_score $?

# [P2-B-FRONTIER] 前沿基线（v6.4-align §10.2）
#   v6.5.28-fix（第六轮审查 🔴）：`--infer-frontier` 参数在 stage_p2b.py 从未实现
#   （argparse 无此旗标 → 未知参数报错 → 阶段必失败）。移除主链调用，PJ-Break/
#   StyleBreak/Now-You-Hear-Me 以 reported_value（§10.3 报告值，source 标注、不混入
#   实测列）呈现，并落盘披露未复现（纪律 #2）。真实复现列为待办。
mkdir -p logs
printf '{"ts":"%s","stage":"p2b_frontier","event":"frontier_not_reproduced","note":"PJ-Break/StyleBreak/Now-You-Hear-Me 真实复现未实现（--infer-frontier 参数缺失），以 reported_value 呈现（§10.3），须在 REVISION_REPORT limitation 披露"}\n' \
  "$(date '+%Y-%m-%dT%H:%M:%S')" >> logs/errors.jsonl
log "[v6.5.28] P2-B 前沿基线：真实复现未实现，以 reported_value 呈现（见 errors.jsonl）"

# [P2-LORA] Intent 分支 LoRA 微调（v6.6.4 补：v9 §8 "有害意图语义判别 → LoRA 微调"）
#   --train-intent-lora：真实 peft 微调 Gemma-4-E2B-it（判别式二分类，5 种子）
#   （v6.5.20-fix 问题 69：原注释"Qwen2.5-1.5B-Instruct"为 v6.4 残留；
#   v6.5 §8 已切 Intent 底座 Gemma-4-E2B-it，见 config p2.lora.model）
#   输入 P1-FULL 评分 parquet 银标签（三方一致子集）→ 输出 results/intent_lora/lora_{seed}/
#   幂等：adapter_config.json 已存在则跳过该种子；P2 主流程检测到 adapter 自动启用
#   lora_prob 特征（v9 §8 意图分支真实化，不再回退逻辑回归占位）。
#   GPU 窗口#1.5（位于 p2b_infer 之后、P2 主流程之前：显存空闲、P1-FULL parquet 已就绪）
log "[v6.6.4] P2 Intent LoRA 微调窗口（GPU，v9 §8 判别式；底座 Gemma-4-E2B-it）"
run_stage p2_lora stage_p2_msrf.py --train-intent-lora; handle_stage_rc p2_lora $?

# [P2] MSRF 融合防御（v6.5：真实融合器落盘 msrf_fusion.pkl + 真实 ROC/PR 曲线点）
run_stage p2 stage_p2_msrf.py; handle_stage_rc p2 $?

# [P2-BASELINE] P2 外部安全基线真实化（v6.5.2 新增，替代纯清单）
#   --evaluate：纯 CPU 汇总 GradSafe + ShieldGemma + WildGuard 推理产物
#   输出 report/external_baselines.md + results/external_baselines.json
log "[v6.5.2] P2 外部基线评估（GradSafe + ShieldGemma + WildGuard 真实实现）"
run_stage p2baseline stage_p2_baselines.py --evaluate; handle_stage_rc p2baseline $?
# v6.5-D2-8（2026-08-11）双 GPU 隔离并行：ShieldGemma [GPU 0] ∥ WildGuard [GPU 1]
# WildGuard（Allen AI, NeurIPS 2024, Apache 2.0）——第二开源安全分类器基线，
# 匹配 GPT-4 性能，较 Llama-Guard2 拒答 F1 提升 25.3%。
# 两者均为 4bit 独立推理（各需 ~8GB VRAM），无依赖关系，天然并行。
GPU_COUNT=$(gpu_count)
VRAM_G0=$(gpu_free_mb 0)
VRAM_G1=$(gpu_free_mb 1)

if [ "$GPU_COUNT" -ge 2 ] && \
   [ -n "$VRAM_G0" ] && [ "$VRAM_G0" -ge 12000 ] && \
   [ -n "$VRAM_G1" ] && [ "$VRAM_G1" -ge 12000 ]; then
  log "[DUAL] ShieldGemma [GPU 0] ∥ WildGuard [GPU 1] 双卡隔离并行（GPU0 ${VRAM_G0}MB / GPU1 ${VRAM_G1}MB）"
  launch_dual p2baseline_shield stage_p2_baselines.py 0 --infer --evaluate ::: \
            p2baseline_wildguard stage_p2_baselines.py 1 --wildguard-infer
  RC_DUAL=$?
  # 个别 marker 检查决定 handle_stage_rc
  [ -f "run/p2baseline_shield.complete" ] || handle_stage_rc p2baseline_shield 3
  [ -f "run/p2baseline_wildguard.complete" ] || handle_stage_rc p2baseline_wildguard 3
  if [ "$RC_DUAL" -eq 2 ]; then
    log "[DUAL] ShieldGemma/WildGuard 部分失败——检查各阶段 marker 决定后续"
  fi
else
  log "[v6.5.2] ShieldGemma → WildGuard 串行（GPU 不足: GPU0=${VRAM_G0:-?}MB GPU1=${VRAM_G1:-?}MB, 需各≥12GB）"
  log "[v6.5.2] P2 外部基线 --infer（ShieldGemma 4bit 真实打分）"
  run_stage p2baseline_shield stage_p2_baselines.py --infer --evaluate; handle_stage_rc p2baseline_shield $?
  log "[v6.5-D2-7] P2 外部基线 --wildguard-infer（WildGuard 4bit 真实打分）"
  run_stage p2baseline_wildguard stage_p2_baselines.py --wildguard-infer; handle_stage_rc p2baseline_wildguard $?
fi
# v6.5.29（裁决：真实复现 GradSafe，ACL 2024）——GPU 窗口运行真实梯度检测
# （gradsafe_real.py 忠实复现：合规响应配对 NLL 梯度 + 参考余弦，Qwen2.5-3B 基座）。
# 幂等（--gradsafe-infer 产物存在则跳过）。不依赖 msrf_fusion.pkl。
# v6.5-D2-8：改为独立 run_stage，便于 checkpoint 和 rc 追踪（原 inline 无 marker）。
log "[v6.5.29] P2 外部基线 --gradsafe-infer（真实 GradSafe 梯度检测）"
run_stage p2baseline_gradsafe stage_p2_baselines.py --gradsafe-infer --evaluate; handle_stage_rc p2baseline_gradsafe $?

# [P2-C] 自适应攻击评估（v6.5：加载 P2 真实融合器；v6.5.1 自动补推理）
#   --infer（GPU 攻击查询推理，checkpoint 幂等）+ --evaluate（CPU 真实判定）。
#   --evaluate 需 P2 已产出 msrf_fusion.pkl；若缺失则如实记录（不伪造模拟结果），
#   并在 P2 完成后的下一轮循环（主链重跑）自动补齐推理。
if [ -f "$ROOT/results/msrf_fusion.pkl" ]; then
  # [v6.6.4] 声学伪装窗口（v9 §9：真实音频层平静朗读情感化文本，分离 E_t/A_s 信号）
  #   --infer-audio-acoustic：TTS 合成 styled 情感化文本（中性 voice Xiaoxiao）
  #   → 音频模态 LALM 推理 → responses/P2C/attacks_acoustic_disguise_audio_{model}.jsonl
  #   必须先于 --infer --evaluate 执行：该分支独占返回且不写 done（共享 P2-C checkpoint），
  #   evaluate 端按文件名标注 attack=graybox_acoustic_disguise_audio 参与攻击分布统计。
  #   幂等：lalm_infer_audio_text 按 checkpoint 跳过已推理条目，重跑不重复合成/推理。
  log "[v6.6.4] P2-C 声学伪装窗口（GPU，v9 §9 真实音频层）"
  run_stage p2c_acoustic stage_p2c_adaptive.py --infer-audio-acoustic; handle_stage_rc p2c_acoustic $?
  log "[v6.5.1] P2-C 使用真实 MSRF 融合器，自动执行 推理+评估（GPU 窗口#2）"
  run_stage p2c stage_p2c_adaptive.py --infer --evaluate; handle_stage_rc p2c $?
else
  log "[v6.5] P2-C 跳过：msrf_fusion.pkl 缺失（P2 未产出真实融合器）。"
  log "       推理将在 P2 完成后的主链重跑中自动补齐（--infer --evaluate）"
  echo "p2c SKIPPED_NO_FUSION $(date '+%F %T')" >> "$FAILED_FILE"
  # v6.5.28-fix（纪律 #2）：跳过必须落盘 errors.jsonl（原只写 FAILED_FILE）。
  # 注意：P2-C（§9 防御论文生死线）缺失融合器被跳过时主链仍跑完 G2/P2-B/F/R，
  # 自适应攻击评估缺失须在最终报告 limitation 段披露。
  mkdir -p logs
  printf '{"ts":"%s","stage":"p2c","event":"skipped_no_fusion","note":"msrf_fusion.pkl 缺失，P2-C 跳过（无阻断）——自适应攻击评估缺失须在 REVISION_REPORT limitation 披露"}\n' \
    "$(date '+%Y-%m-%dT%H:%M:%S')" >> logs/errors.jsonl
fi

# [G2] 闸门
run_gate g2
G2_RC=$?
if [ "$G2_RC" -eq 3 ]; then
  log "[FATAL] G2 无法判定（数据缺失），流水线终止"
  exit 3
elif [ "$G2_RC" -ne 0 ]; then
  log "[G2] 闸门未通过（转为机制主导版——§8 协议）"
  # v6.5.18-fix（问题 54 连带）：不通过如实落盘 FAILED_FILE（纪律#2）
  echo "G2 NOT_PASSED $(date '+%F %T')" >> "$FAILED_FILE"
fi

# [P2-B] 降级音频 + 前沿基线（v6.5.1：报告阶段，零 GPU）
#   真实推理已由 p2b_infer 完成（若本 run 已执行或此前手动执行过），
#   此处 --report-only 读取 P2B 真实推理结果 → measured 登记 + 降级文件生成 + 报告值分离披露。
#   幂等：不写 done 标记，主链重跑时若 p2b_infer 未完成可自动补推理。
run_stage p2b stage_p2b.py --report-only; handle_stage_rc p2b $?

# [F] 出版级图表
run_stage f stage_f_figures.py; handle_stage_rc f $?

# [P3] 统计补强（穿插）
run_stage p3 stage_p3.py; handle_stage_rc p3 $?

# [R] 可复现性工件包
run_stage r stage_r_artifact.py; handle_stage_rc r $?

# =============================================================================
# 汇总
# =============================================================================
T1=$(date +%s)
ELAPSED=$(( (T1 - T0) / 60 ))
echo ""
echo "============================================================"
if [ -s "$FAILED_FILE" ]; then
  echo "PIPELINE FINISHED WITH FAILURES"
  echo "失败/跳过清单："
  cat "$FAILED_FILE"
else
  echo "PIPELINE COMPLETE"
fi
echo "总耗时: ${ELAPSED} 分钟"
echo "============================================================"

