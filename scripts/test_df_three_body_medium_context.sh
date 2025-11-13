name="df_three_body"
dataset="video_three_body"
val_batch_size=15
wandb_run_id=94qmixi5
context_length=180
val_limit_batch=1

CUDA_VISIBLE_DEVICES=1 python main.py \
    +name=${name} \
    load=${wandb_run_id} \
    dataset=${dataset}\
    experiment.validation.batch_size=${val_batch_size} \
    experiment.tasks=["validation"] \
    algorithm.metrics=["mse"] \
    dataset.context_length=${context_length} \
    experiment.validation.limit_batch=${val_limit_batch}