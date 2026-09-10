# Scheduler oracle

`sigmas.f32le` records Diffusers 0.41.0.dev0 FlowMatchEulerDiscreteScheduler
CPU outputs from the retired `test_schedule.py`, captured before removing its runtime
dependency. Configuration: `{'base_image_seq_len': 256, 'base_shift': 0.5, 'max_image_seq_len': 6400, 'max_shift': 1.15, 'num_train_timesteps': 1000, 'shift': 1.0, 'time_shift_type': 'exponential', 'use_dynamic_shifting': True}`.

Little-endian f32 values, for image token counts 0 (Turbo), 16, 256, 589, 4096,
6400, 16384, then step counts 1 through 100. Each grid includes the terminal zero.
Raw mu is the linear interpolation between (256, 0.5) and (6400, 1.15).
The 589-token case detects premature rounding of mu. This immutable external
reference is compared bit for bit by Rust; no Python environment is needed.
