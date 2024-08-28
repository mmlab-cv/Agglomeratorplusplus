import flags_mag
from absl import app
from absl import flags
FLAGS = flags.FLAGS

def main(argv):
    FLAGS.append_flags_into_file('config/CIFAR10_pretrain.cfg')
    
if __name__ == '__main__':
    app.run(main)