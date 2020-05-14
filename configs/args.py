import argparse


parser = argparse.ArgumentParser()

# config path here
parser.add_argument('--base-config-fn', type=str, default='./configs/base_config.yml')
parser.add_argument('--extra-config-fns', type=str, nargs='+', default='./datasets/wisdom/wisdomConfig.yml')


# config general training setting here
parser.add_argument('--eval-first', action='store_true', default=False,
                    help='If True, the training will first eval the network(can be use to debug the val function')
parser.add_argument('--batch-size', type=int, default=0)

parser.add_argument('--resume', type=str, default='',
    help='resume dir which contain the checkpoint network')

# config log arguments
parser.add_argument('--log-name', type=str, default='')


args = parser.parse_args()
