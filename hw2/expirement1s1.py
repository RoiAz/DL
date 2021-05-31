import hw2.experiments as experiments
#from hw2.experiments import load_experiment
#from cs236781.plot import plot_fit

# Test experiment1 implementation on a few data samples and with a small model
L=[2,4,8,16]
K=[32,64]

for channel in K:
    for l in L:
        test_name = 'L'+str(l)+'_K'+str(channel)
        
        experiments.run_experiment( 
            'exp1_1_'+test_name, seed=seed, bs_train=50, batches=10, epochs=10, early_stopping=5,
            filters_per_layer=[channel], layers_per_block=L, pool_every=1, hidden_dims=[100],
            model_type='resnet',
            )   

