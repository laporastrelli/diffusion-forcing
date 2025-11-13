name="df_two_body"
dataset="video_two_body"
val_batch_size=25
wandb_run_id=2c0zj0k6
context_length=60
val_limit_batch=2

CUDA_VISIBLE_DEVICES=3 python main.py \
    +name=${name} \
    load=${wandb_run_id} \
    dataset=${dataset}\
    experiment.validation.batch_size=${val_batch_size} \
    experiment.tasks=["validation"] \
    algorithm.metrics=["mse"] \
    dataset.context_length=${context_length} \
    experiment.validation.limit_batch=${val_limit_batch}