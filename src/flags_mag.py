from absl import flags

# DATA

flags.DEFINE_string('dataset', 'CIFAR10', 'Dataset name.')
flags.DEFINE_string('exp_name', 'CIFAR10', 'Experiment name.')
flags.DEFINE_integer('image_size', 32, 'Input images size.')
flags.DEFINE_integer('n_channels', 3, 'Number of image channels.')
flags.DEFINE_integer('n_classes', 10, 'Number of classes.')

# NETWORK PARAMETERS

flags.DEFINE_integer('patch_size', 4, 'Patch size.')
flags.DEFINE_integer('patch_dim', 512, 'Patch feature dimension.')
flags.DEFINE_integer('batch_size', 512, 'Batch size.')
flags.DEFINE_integer('levels', 5, 'Columns levels.')
flags.DEFINE_integer('iters', None, 'Number of iterations for the columns (if None it will be set by the network).')
flags.DEFINE_integer('denoise_iter', -1, 'At which iteration to perform denoising.')
flags.DEFINE_float('dropout', 0.0, 'Dropout.')
flags.DEFINE_integer('decoder_dim', 512, 'Decoder dim.')
flags.DEFINE_integer('decoder_depth', 5, 'Decoder depth.')
flags.DEFINE_integer('decoder_heads', 8, 'Decoder heads.')

# TRAINING PARAMETERS

flags.DEFINE_boolean('resume_training', False,
                     'Resume training using a checkpoint.')
flags.DEFINE_bool('supervise', False, 'Supervise training.')
flags.DEFINE_string('mode', 'train', 'train/test/freeze.')
flags.DEFINE_string('load_checkpoint_dir', 'datasets/CIFAR_pretrained_token.ckpt',
                    'Load previous existing checkpoint.')
flags.DEFINE_integer('seed', 42, 'Seed.')
flags.DEFINE_integer('num_workers', 8, 'Number of workers.')
flags.DEFINE_float('noise_ratio', 0.5, 'Noise factor')

# OPTIMIZER

flags.DEFINE_integer('max_epochs', 800, 'Number of training epochs.')
flags.DEFINE_integer('max_epochs_finetune', 100, 'Number of fine-tuning epochs.')
flags.DEFINE_float('learning_rate', 1.5e-4, 'Learning rate.')
flags.DEFINE_float('min_lr', 1e-6, 'Minimum learning rate.')
flags.DEFINE_float('weight_decay', 0.05, 'Weight decay.')
flags.DEFINE_float('beta1', 0.9, 'AdamW beta1.')
flags.DEFINE_float('beta2', 0.95, 'AdamW beta1.')
flags.DEFINE_integer('warmup', 40, 'Learning rate scheduler warmup.')


flags.DEFINE_float('limit_train', 1.0, 'Limit train set.')
flags.DEFINE_float('limit_val', 1.0, 'Limit val set.')
flags.DEFINE_float('limit_test', 1.0, 'Limit test set.')
flags.DEFINE_float('masking_ratio', 0.75, 'Masked patches ratio.')


flags.DEFINE_bool('plot_islands', False, 'Plot islands of agreement.')
flags.DEFINE_bool('plot_reconstruction', False, 'Plot image reconstruction.')


flags.DEFINE_bool('use_token', False, 'Use token for classification.')
flags.DEFINE_bool('use_time_token', False, 'Use time token for recurrent iterations.')
flags.DEFINE_integer('num_tokens', 0, 'Numbers of tokens used (leave to 0).')