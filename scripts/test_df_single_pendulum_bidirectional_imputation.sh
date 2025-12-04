name="df_single_pendulum_full"
dataset="video_single_pendulum_full"
wandb_run_id=f6lvhc7e
val_batch_size=25
val_limit_batch=2
imputation_as_val=1 # set to 1 to enable bidirectional imputation during validation
symmetric_context=1 # set to 1 to use symmetric context frames around the missing frames

for context_length in 58; do
    chunk_size=$((60 - context_length))  # total frames = 60 for single pendulum full

    CUDA_VISIBLE_DEVICES=2 python main.py \
        +name=${name} \
        load=${wandb_run_id} \
        dataset=${dataset} \
        experiment.validation.batch_size=${val_batch_size} \
        experiment.tasks=["validation"] \
        dataset.context_length=${context_length} \
        experiment.validation.limit_batch=${val_limit_batch} \
        +algorithm.metrics=["mse"] \
        algorithm.chunk_size=${chunk_size} \
        algorithm.diffusion.transition_type="bi-gru" \
        algorithm.diffusion.imputation_as_val=${imputation_as_val} \
        algorithm.diffusion.symmetric_context=${symmetric_context}
done