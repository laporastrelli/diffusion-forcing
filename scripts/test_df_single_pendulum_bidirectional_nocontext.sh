name="df_single_pendulum_full"
dataset="video_single_pendulum_full"
wandb_run_id=f6lvhc7e
val_batch_size=25
context_length=0
val_limit_batch=2

for chunk_size in 60; do

    CUDA_VISIBLE_DEVICES=1 python main.py \
        +name=${name} \
        load=${wandb_run_id} \
        dataset=${dataset} \
        experiment.validation.batch_size=${val_batch_size} \
        experiment.tasks=["validation"] \
        dataset.context_length=${context_length} \
        experiment.validation.limit_batch=${val_limit_batch} \
        +algorithm.metrics=["mse"] \
        algorithm.chunk_size=${chunk_size} \
        algorithm.diffusion.transition_type="bi-gru"
done