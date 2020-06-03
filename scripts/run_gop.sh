n_data=10
dx=50

python ./global_optimal_proposal_variational.py --resampling_method=3 --n_data=$n_data --change_seed=True --dx=$dx --dy=1 --batch_size=4 --epsilon=0.5 --n_particles=25 -savefig --T=150 --scaling=0.9 --convergence_threshold=1e-3 --resampling_neff=0.5 &
python ./global_optimal_proposal_variational.py --resampling_method=3 --n_data=$n_data --change_seed=False --dx=$dx --dy=1 --batch_size=4 --epsilon=0.5 --n_particles=25 -savefig --T=150 --scaling=0.9 --convergence_threshold=1e-3 --resampling_neff=0.5 &
python ./global_optimal_proposal_variational.py --resampling_method=0 --n_data=$n_data --change_seed=True --dx=$dx --dy=1 --batch_size=4 --epsilon=0.5 --n_particles=25 -savefig --T=150 --scaling=0.9 --convergence_threshold=1e-3 --resampling_neff=0.5 &
python ./global_optimal_proposal_variational.py --resampling_method=0 --n_data=$n_data --change_seed=False --dx=$dx --dy=1 --batch_size=4 --epsilon=0.5 --n_particles=25 -savefig --T=150 --scaling=0.9 --convergence_threshold=1e-3 --resampling_neff=0.5 &

python ./global_optimal_proposal_variational.py --resampling_method=0 --n_data=$n_data --change_seed=True --dx=$dx --dy=1 --batch_size=1 --epsilon=0.5 --n_particles=25 -savefig --T=150 --scaling=0.9 --convergence_threshold=1e-3 --resampling_neff=0.5 &
python ./global_optimal_proposal_variational.py --resampling_method=0 --n_data=$n_data --change_seed=False --dx=$dx --dy=1 --batch_size=1 --epsilon=0.5 --n_particles=25 -savefig --T=150 --scaling=0.9 --convergence_threshold=1e-3 --resampling_neff=0.5 &
python ./global_optimal_proposal_variational.py --resampling_method=3 --n_data=$n_data --change_seed=False --dx=$dx --dy=1 --batch_size=1 --epsilon=0.5 --n_particles=25 -savefig --T=150 --scaling=0.9 --convergence_threshold=1e-3 --resampling_neff=0.5 &
python ./global_optimal_proposal_variational.py --resampling_method=3 --n_data=$n_data --change_seed=True --dx=$dx --dy=1 --batch_size=1 --epsilon=0.5 --n_particles=25 -savefig --T=150 --scaling=0.9 --convergence_threshold=1e-3 --resampling_neff=0.5 &

python ./global_optimal_proposal_variational.py --resampling_method=0 --n_data=$n_data --change_seed=True --dx=$dx --dy=1 --batch_size=10 --epsilon=0.5 --n_particles=25 -savefig --T=150 --scaling=0.9 --convergence_threshold=1e-3 --resampling_neff=0.5 &
python ./global_optimal_proposal_variational.py --resampling_method=0 --n_data=$n_data --change_seed=False --dx=$dx --dy=1 --batch_size=10 --epsilon=0.5 --n_particles=25 -savefig --T=150 --scaling=0.9 --convergence_threshold=1e-3 --resampling_neff=0.5 &
python ./global_optimal_proposal_variational.py --resampling_method=3 --n_data=$n_data --change_seed=False --dx=$dx --dy=1 --batch_size=10 --epsilon=0.5 --n_particles=25 -savefig --T=150 --scaling=0.9 --convergence_threshold=1e-3 --resampling_neff=0.5 &
python ./global_optimal_proposal_variational.py --resampling_method=3 --n_data=$n_data --change_seed=True --dx=$dx --dy=1 --batch_size=10 --epsilon=0.5 --n_particles=25 -savefig --T=150 --scaling=0.9 --convergence_threshold=1e-3 --resampling_neff=0.5 &


python ./global_optimal_proposal_variational.py --resampling_method=4 --n_data=$n_data --change_seed=True --dx=$dx --dy=1 --batch_size=4 --epsilon=0.5 --n_particles=25 -savefig --T=150 --scaling=0.9 --convergence_threshold=1e-3 --resampling_neff=0.5 &
python ./global_optimal_proposal_variational.py --resampling_method=4 --n_data=$n_data --change_seed=False --dx=$dx --dy=1 --batch_size=4 --epsilon=0.5 --n_particles=25 -savefig --T=150 --scaling=0.9 --convergence_threshold=1e-3 --resampling_neff=0.5 &
