python train_vrnn.py --resampling_method="reg" --epsilon=0.5 --resampling_neff=0.5 --scaling=0.9 \
                     --convergence_threshold=1e-3  \
                     --initial_lr=0.001 --decay 0.99 --decay_steps=250 \
                     --n_iter=10000 --max_iter=1000  --n_particles=25\
                     --out_dir="./vrnn5"  \
                     --filter_seed=42 --data_seed=0 -nofixed_filter_seed &

python train_vrnn.py --resampling_method="mult" --epsilon=0.5 --resampling_neff=0.5 --scaling=0.9 \
                     --convergence_threshold=1e-3  \
                     --initial_lr=0.001 --decay 0.99 --decay_steps=250 \
                     --n_iter=10000 --max_iter=1000  --n_particles=25\
                     --out_dir="./vrnn5"  \
                     --filter_seed=42 --data_seed=0 -nofixed_filter_seed &

python train_vrnn.py --resampling_method="reg" --epsilon=0.5 --resampling_neff=0.5 --scaling=0.9 \
                     --convergence_threshold=1e-3  \
                     --initial_lr=0.005 --decay 0.99 --decay_steps=250 \
                     --n_iter=10000 --max_iter=1000  --n_particles=25\
                     --out_dir="./vrnn5"  \
                     --filter_seed=42 --data_seed=0 -nofixed_filter_seed  &

python train_vrnn.py --resampling_method="mult" --epsilon=0.25 --resampling_neff=0.5 --scaling=0.9 \
                     --convergence_threshold=1e-3  \
                     --initial_lr=0.005 --decay 0.99 --decay_steps=250 \
                     --n_iter=10000 --max_iter=1000  --n_particles=25\
                     --out_dir="./vrnn5"  \
                     --filter_seed=42 --data_seed=0 -nofixed_filter_seed &


