name="df_single_pendulum_full"
dataset="video_single_pendulum_full"
val_batch_size=25
wandb_run_id=dzt8h52w
context_length=60
val_limit_batch=2

for chunk_size in 1 10 30 60; do

    CUDA_VISIBLE_DEVICES=0 python main.py \
        +name=${name} \
        load=${wandb_run_id} \
        dataset=${dataset} \
        experiment.validation.batch_size=${val_batch_size} \
        experiment.tasks=["validation"] \
        dataset.context_length=${context_length} \
        experiment.validation.limit_batch=${val_limit_batch} \
        +algorithm.metrics=["mse"] \
        algorithm.chunk_size=${chunk_size} \
        algorithm.diffusion.transition_type="forward-only"
done