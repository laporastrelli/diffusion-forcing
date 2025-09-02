name="df_double_pendulum"
dataset="video_double_pendulum"
val_batch_size=25
wandb_run_id=rukd3ifm
context_length=60
val_limit_batch=2

CUDA_VISIBLE_DEVICES=1 python main.py \
    +name=${name} \
    load=${wandb_run_id} \
    dataset=${dataset}\
    experiment.validation.batch_size=${val_batch_size} \
    experiment.tasks=["validation"] \
    algorithm.metrics=["mse"] \
    dataset.context_length=${context_length} \
    experiment.validation.limit_batch=${val_limit_batch}