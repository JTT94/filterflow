python train_vrnn.py --epsilon=0.5 --resampling_neff=0.5 --scaling=0.9 \
                     --convergence_threshold=0.5  \
                     --initial_lr=0.01 --decay 0.75 --decay_steps=75 \
                     --n_iter=500 --max_iter=1000  \
                     --out_dir=".\charts"  \
                     --filter_seed=43 --data_seed=0 -fixed_filter_seed &

python train_vrnn.py --epsilon=0.5 --resampling_neff=0.5 --scaling=0.9 \
                     --convergence_threshold=0.5  \
                     --initial_lr=0.01 --decay 0.75 --decay_steps=75 \
                     --n_iter=500 --max_iter=1000  \
                     --out_dir=".\charts"  \
                     --filter_seed=43 --data_seed=0 -nofixed_filter_seed &

