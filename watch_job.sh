#!/bin/bash
# watch_job.sh — 自动监控 SLURM job，完成后打印摘要
# 用法: bash watch_job.sh <JOB_ID> [log_dir]
#
# 功能:
#   - 每30秒检查一次 job 状态
#   - Job 结束后显示 stdout/stderr 末尾内容
#   - 如果检测到常见错误，给出修复建议

set -euo pipefail

JOB_ID=${1:?Usage: $0 <JOB_ID> [log_dir]}
LOG_DIR=${2:-logs}
OUT_LOG="$LOG_DIR/llava7b_${JOB_ID}.out"
ERR_LOG="$LOG_DIR/llava7b_${JOB_ID}.err"

echo "=== Watching job $JOB_ID ==="
echo "    stdout: $OUT_LOG"
echo "    stderr: $ERR_LOG"
echo ""

# Wait for log files to appear
for i in $(seq 1 20); do
    if [[ -f "$ERR_LOG" ]]; then break; fi
    echo "Waiting for log files... ($i)"
    sleep 5
done

# Poll until job ends
while true; do
    STATUS=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null || echo "DONE")
    TIME=$(squeue -j "$JOB_ID" -h -o "%M" 2>/dev/null || echo "-")

    if [[ "$STATUS" == "DONE" || -z "$STATUS" ]]; then
        echo ""
        echo "=== Job $JOB_ID finished ==="
        break
    fi

    printf "\r[%s] Status: %s  Time: %s   " "$(date +%H:%M:%S)" "$STATUS" "$TIME"
    sleep 30
done

# ── Show output ────────────────────────────────────────────────────────────────
echo ""
echo "--- STDOUT (last 30 lines) ---"
tail -30 "$OUT_LOG" 2>/dev/null || echo "(no stdout log)"

echo ""
echo "--- STDERR (last 40 lines) ---"
tail -40 "$ERR_LOG" 2>/dev/null || echo "(no stderr log)"

# ── Error detection ────────────────────────────────────────────────────────────
echo ""
echo "=== Error Analysis ==="

ERRORS=0

if grep -q "ModuleNotFoundError" "$ERR_LOG" 2>/dev/null; then
    MODULE=$(grep "ModuleNotFoundError" "$ERR_LOG" | head -1 | grep -oP "No module named '\K[^']+")
    echo "❌ Missing Python module: $MODULE"
    echo "   Fix: add 'pip install $MODULE' to submit script"
    ERRORS=$((ERRORS+1))
fi

if grep -q "no kernel image is available" "$ERR_LOG" 2>/dev/null; then
    echo "❌ CUDA kernel mismatch (torch built for different GPU CC)"
    echo "   Fix: check GPU CC with 'squeue -o %b', use matching torch+cuXXX build"
    ERRORS=$((ERRORS+1))
fi

if grep -q "Permission denied.*scratch" "$ERR_LOG" 2>/dev/null; then
    echo "❌ /scratch permission denied"
    echo "   Fix: use /work/users/t/x/\$USER instead of /scratch"
    ERRORS=$((ERRORS+1))
fi

if grep -q "No module named 'torch'" "$ERR_LOG" 2>/dev/null; then
    echo "❌ torch not installed in active environment"
    echo "   Fix: ensure venv is activated before running python3"
    ERRORS=$((ERRORS+1))
fi

if grep -q "CUDA error" "$ERR_LOG" 2>/dev/null; then
    echo "❌ CUDA runtime error detected"
    echo "   Fix: check cuda module version matches torch build (e.g. cuda/11.8 + cu118)"
    ERRORS=$((ERRORS+1))
fi

if grep -q "Permission denied.*scratch\|No such file.*activate" "$ERR_LOG" 2>/dev/null; then
    echo "❌ venv activation failed"
    echo "   Fix: ensure WORKFS path is correct and accessible from compute node"
    ERRORS=$((ERRORS+1))
fi

if grep -q "Done\." "$OUT_LOG" 2>/dev/null; then
    SUCCESS_COUNT=$(grep "Done\." "$OUT_LOG" | grep -oP '\d+ evaluated' | grep -oP '\d+' || echo "?")
    ERR_COUNT=$(grep "Done\." "$OUT_LOG" | grep -oP '\d+ errors' | grep -oP '\d+' || echo "?")
    echo "✅ Evaluation finished: $SUCCESS_COUNT evaluated, $ERR_COUNT errors"
fi

if [[ $ERRORS -eq 0 ]]; then
    echo "✅ No known errors detected"
fi
