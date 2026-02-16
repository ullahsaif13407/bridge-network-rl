#!/usr/bin/env bash
set -e

DEVICE="${1:-cuda:0}"
DIR="$(cd "$(dirname "$0")" && pwd)"
SESSION="rl_train"

CMD_OFFPOLICY="python3 ${DIR}/Trainers/torch_compiled_fused_adam/gpu_offpolicy_training.py \
    --wandb --wandb-name offpolicy --device ${DEVICE} \
    --checkpoint-dir ${DIR}/Trainers/torch_compiled_fused_adam/checkpoints_offpolicy; bash"

CMD_PPO="python3 ${DIR}/Trainers/ppo_torch_compiled_fused_adam/gpu_ppo_training.py \
    --wandb --wandb-name ppo --device ${DEVICE} \
    --checkpoint-dir ${DIR}/Trainers/ppo_torch_compiled_fused_adam/checkpoints_ppo; bash"

CMD_GRPO="python3 ${DIR}/Trainers/grpo_torch_compiled_fused_adam/gpu_grpo_training.py \
    --wandb --wandb-name grpo --device ${DEVICE} \
    --checkpoint-dir ${DIR}/Trainers/grpo_torch_compiled_fused_adam/checkpoints_grpo; bash"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create session with first pane — Off-policy
tmux new-session -d -s "$SESSION" -n train "$CMD_OFFPOLICY"

# Split horizontally — PPO
tmux split-window -h -t "$SESSION" "$CMD_PPO"

# Split vertically — GRPO
tmux split-window -v -t "$SESSION" "$CMD_GRPO"

# Even out pane layout
tmux select-layout -t "$SESSION" even-horizontal

echo "Attaching to tmux session '$SESSION' ..."
echo "  Ctrl+B then D to detach | tmux attach -t $SESSION to reattach"
tmux attach -t "$SESSION"
