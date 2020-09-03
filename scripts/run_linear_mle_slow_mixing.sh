python ./simple_linear_mle.py --resampling_method=3 --batch_data=100 --change_seed=True --batch_size=4 --epsilon=0.5 --phi=0.9 --n_particles=33 -savefig --T=150 --scaling=0.8 --convergence_threshold=1e-5 --learning_rate=1e-5 --resampling_neff=1. &
python ./simple_linear_mle.py --resampling_method=0 --batch_data=100 --change_seed=True --batch_size=4 --epsilon=0.5 --phi=0.9 --n_particles=5000 -savefig --T=150 --scaling=0.8 --convergence_threshold=1e-5 --learning_rate=1e-5 --resampling_neff=1. &

python ./simple_linear_mle.py --resampling_method=3 --batch_data=100 --change_seed=False --batch_size=4 --epsilon=0.5 --phi=0.9 --n_particles=33 -savefig --T=150 --scaling=0.8 --convergence_threshold=1e-5 --learning_rate=1e-5 --resampling_neff=1. &
python ./simple_linear_mle.py --resampling_method=0 --batch_data=100 --change_seed=False --batch_size=4 --epsilon=0.5 --phi=0.9 --n_particles=5000 -savefig --T=150 --scaling=0.8 --convergence_threshold=1e-5 --learning_rate=1e-5 --resampling_neff=1. &
