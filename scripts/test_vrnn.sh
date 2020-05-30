python train_vrnn.py --epsilon=0.8 --resampling_neff=0.5 --scaling=0.9 \
                     --convergence_threshold=1e-3  \
                     --initial_lr=0.01 --decay 0.9 --decay_steps=200 \
                     --n_iter=1000 --max_iter=1000  --n_particles=100 \
                     --out_dir="./test"  \
                     --filter_seed=42 --data_seed=0 -nofixed_filter_seed
