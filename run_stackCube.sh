# ENV=PullCube-v1
# NENV=256
# STEPS=5000000 # use 40M for long and 5M for short
# for SEED in 1 42 123; do
#     CUDA_VISIBLE_DEVICES=0 python examples/baselines/ppo/ppo_fast.py \
#     --env_id $ENV \
#     --reach_variant tanh \
#     --gate_variant hard --terminal_variant hard_jump \
#     --num_envs $NENV --total_timesteps $STEPS \
#     --track --wandb_entity agv_rego --wandb_project_name REP --wandb_group REP \
#     --seed $SEED \
#     & \
#     CUDA_VISIBLE_DEVICES=1 python examples/baselines/ppo/ppo_fast.py \
#     --env_id $ENV \
#     --reach_variant concave_truncated \
#     --gate_variant hard --terminal_variant hard_jump \
#     --num_envs $NENV --total_timesteps $STEPS \
#     --track --wandb_entity agv_rego --wandb_project_name REP --wandb_group REP \
#     --seed $SEED \
#     & \
#     CUDA_VISIBLE_DEVICES=0 python examples/baselines/ppo/ppo_fast.py \
#     --env_id $ENV \
#     --reach_variant adaptive_tanh_stage \
#     --gate_variant hard --terminal_variant hard_jump \
#     --num_envs $NENV --total_timesteps $STEPS \
#     --track --wandb_entity agv_rego --wandb_project_name REP --wandb_group REP \
#     --seed $SEED \
#     & \
#     CUDA_VISIBLE_DEVICES=1 python examples/baselines/ppo/ppo_fast.py \
#     --env_id $ENV \
#     --reach_variant adaptive_concave_stage \
#     --gate_variant hard --terminal_variant hard_jump \
#     --num_envs $NENV --total_timesteps $STEPS \
#     --track --wandb_entity agv_rego --wandb_project_name REP --wandb_group REP \
#     --seed $SEED
# done


for SEED in 1 42 123; do
  python examples/baselines/sac/spda_sac.py \
    --demo_path demos/demos.pt \
    --demo_sampling_ratio 0.5 \
    --critic_threshold -5.0 \
    --demo_distance_threshold 0.1 \
    --num_envs 16 \
    --total_timesteps 1000000 \
    --track \
    --wandb_entity agv_rego \
    --wandb_project_name SPDA \
    --wandb_group SPDA \
    --seed $SEED
done

# for SEED in 1 42 123; do
#   python examples/baselines/sac/sac.py \
#     --num_envs 16 --total_timesteps 500000 \
#     --track --wandb_entity agv_rego \
#     --wandb_project_name SPDA --wandb_group SPDA \
#     --seed $SEED
# done

