name="df_single_pendulum_full_latents"
dataset="video_single_pendulum_full_vae_latents"
batch_size=20
epochs=150
checkpointing_frequency=1200
val_every_n_epoch=151
val_batch_size=100
device=2
vae_checkpoint=""

export PYTHONPATH="/data2/users/lr4617"

CUDA_VISIBLE_DEVICES=${device} python main.py \
    +name=${name} \
    dataset=${dataset}\
    experiment.training.batch_size=${batch_size} \
    experiment.training.max_epochs=${epochs} \
    experiment.training.checkpointing.every_n_epochs=${checkpointing_frequency} \
    experiment.validation.val_every_n_epoch=${val_every_n_epoch} \
    experiment.validation.batch_size=${val_batch_size} \
    experiment.use_callbacks=true \
    experiment.tasks=["training","validation"] \
    experiment.device=${device}\
    algorithm.metrics=["mse"] \
