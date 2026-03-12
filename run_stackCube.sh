ENV=PullCube-v1
NENV=256
STEPS=5000000 # use 40M for long and 5M for short
for SEED in 1 42 123; do
    CUDA_VISIBLE_DEVICES=0 python examples/baselines/ppo/ppo_fast.py \
    --env_id $ENV \
    --reach_variant tanh \
    --gate_variant hard --terminal_variant hard_jump \
    --num_envs $NENV --total_timesteps $STEPS \
    --track --wandb_entity agv_rego --wandb_project_name REP --wandb_group REP \
    --seed $SEED \
    & \
    CUDA_VISIBLE_DEVICES=1 python examples/baselines/ppo/ppo_fast.py \
    --env_id $ENV \
    --reach_variant concave_truncated \
    --gate_variant hard --terminal_variant hard_jump \
    --num_envs $NENV --total_timesteps $STEPS \
    --track --wandb_entity agv_rego --wandb_project_name REP --wandb_group REP \
    --seed $SEED \
    & \
    CUDA_VISIBLE_DEVICES=0 python examples/baselines/ppo/ppo_fast.py \
    --env_id $ENV \
    --reach_variant adaptive_tanh_stage \
    --gate_variant hard --terminal_variant hard_jump \
    --num_envs $NENV --total_timesteps $STEPS \
    --track --wandb_entity agv_rego --wandb_project_name REP --wandb_group REP \
    --seed $SEED \
    & \
    CUDA_VISIBLE_DEVICES=1 python examples/baselines/ppo/ppo_fast.py \
    --env_id $ENV \
    --reach_variant adaptive_concave_stage \
    --gate_variant hard --terminal_variant hard_jump \
    --num_envs $NENV --total_timesteps $STEPS \
    --track --wandb_entity agv_rego --wandb_project_name REP --wandb_group REP \
    --seed $SEED
done

# CUDA_VISIBLE_DEVICES=0 python examples/baselines/ppo/ppo_fast.py \
# --env_id StackCube-v1 \
# --reach_variant adaptive_concave_threshold \
# --gate_variant hard --terminal_variant hard_jump \
# --num_envs 1024 --total_timesteps 40000000 \
# --track --wandb_entity agv_rego --wandb_project_name REP --wandb_group REP \
# --seed 1 \
# & \
# CUDA_VISIBLE_DEVICES=1 python examples/baselines/ppo/ppo_fast.py \
# --env_id StackCube-v1 \
# --reach_variant adaptive_concave_threshold \
# --gate_variant hard --terminal_variant hard_jump \
# --num_envs 1024 --total_timesteps 40000000 \
# --track --wandb_entity agv_rego --wandb_project_name REP --wandb_group REP \
# --seed 42 \
# && \
#   CUDA_VISIBLE_DEVICES=0 python examples/baselines/ppo/ppo_fast.py \
# --env_id StackCube-v1 \
# --reach_variant adaptive_concave_threshold \
# --gate_variant hard --terminal_variant hard_jump \
# --num_envs 1024 --total_timesteps 40000000 \
# --track --wandb_entity agv_rego --wandb_project_name REP --wandb_group REP \
# --seed 123
