require('torch')
require('../net.lua')

args = {random_seed=0,
        config_file='testnet4.ini',
        experiment_data={experiment_id='testbork'},
        dp_params={{hdf5source='test.h5',
                    sourcelist={"data", "labels"},
                    batch_size=10,
                    postprocess={data={'flatten', {}},
                                 labels={'typer', {}}}}},
        outputPatterns={torch.ones(1)},
        num_batches=1000,
        weight_decay=.00005,
        momentum_params={base_momentum=.9},
        learning_rate_params={base_learning_rate=1},
        save_freq=100,
        write_freq=2,
        save_host='localhost',
        save_port=29101,
        db_name='test_fark',
        collection_name='blump'

        }

net.trainSGDMultiObjective(args)