"""This script performs inference from a trained model."""
import os
import logging
import argparse
from pprint import pformat
import numpy as np
import scipy.stats
import tensorflow as tf
from musegan.config import LOGLEVEL, LOG_FORMAT
from musegan.data import load_data, get_samples
from musegan.model import Model
from musegan.utils import make_sure_path_exists, load_yaml, update_not_none
LOGGER = logging.getLogger("musegan.inference")

#######################
#Sickonet
def andmask(train, ind):
    #function that computes the notes shared between all the different instrument tracks
    shape = train.shape
    a = np.zeros((shape[0], shape[1], shape[2], shape[3]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for m in range(shape[3]):
                    if train[i][j][k][m][0] and train[i][j][k][m][1] and train[i][j][k][m][2] and train[i][j][k][m][3] and train[i][j][k][m][4]:
                        a[i][j][k][m] = 1
    #print(i)
    #print(train.shape)
    #print(a.shape)
    train[..., ind] = a
    return train

    #print(shape)
def ormask(train):
    #function that computes the notes used by any of the different instrument tracks
    shape = train.shape
    o = np.zeros((shape[0], shape[1], shape[2], shape[3]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for m in range(shape[3]):
                    if train[i][j][k][m][0] or train[i][j][k][m][1] or train[i][j][k][m][2] or train[i][j][k][m][3] or train[i][j][k][m][4]:
                        o[i][j][k][m][0] = 1
    #train[..., params['condition_track_idx']] = o
    return o
def xormask(train):
    #function that computes the notes used by only one of the different instrument tracks
    shape = train.shape
    xo = np.zeros((shape[0], shape[1], shape[2], shape[3]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for m in range(shape[3]):
                    if train[i][j][k][m][0] ^ train[i][j][k][m][1] ^ train[i][j][k][m][2] ^ train[i][j][k][m][3] ^ train[i][j][k][m][4]:
                        xo[i][j][k][m][0] = 1
    #train[..., params['condition_track_idx']] = xo
    return xo


###########################



def parse_arguments():
    """Parse and return the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir',
                        help="Directory where the results are saved.")
    parser.add_argument('--checkpoint_dir',
                        help="Directory that contains checkpoints.")
    parser.add_argument('--params', '--params_file', '--params_file_path',
                        help="Path to the file that defines the "
                             "hyperparameters.")
    parser.add_argument('--config', help="Path to the configuration file.")
    parser.add_argument('--runs', type=int, default="1",
                        help="Times to run the inference process.")
    parser.add_argument('--rows', type=int, default=5,
                        help="Number of images per row to be generated.")
    parser.add_argument('--columns', type=int, default=5,
                        help="Number of images per column to be generated.")
    parser.add_argument('--lower', type=float, default=-2,
                        help="Lower bound of the truncated normal random "
                             "variables.")
    parser.add_argument('--upper', type=float, default=2,
                        help="Upper bound of the truncated normal random "
                             "variables.")
    parser.add_argument('--gpu', '--gpu_device_num', type=str, default="0",
                        help="The GPU device number to use.")
    args = parser.parse_args()
    return args

def setup():
    """Parse command line arguments, load model parameters, load configurations
    and setup environment."""
    # Parse the command line arguments
    args = parse_arguments()

    # Load parameters
    params = load_yaml(args.params)

    # Load training configurations
    config = load_yaml(args.config)
    update_not_none(config, vars(args))

    # Set unspecified schedule steps to default values
    for target in (config['learning_rate_schedule'], config['slope_schedule']):
        if target['start'] is None:
            target['start'] = 0
        if target['end'] is None:
            target['end'] = config['steps']

    # Make sure result directory exists
    make_sure_path_exists(config['result_dir'])

    # Setup GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']

    return params, config

def main():
    """Main function."""
    # Setup
    logging.basicConfig(level=LOGLEVEL, format=LOG_FORMAT)
    params, config = setup()
    LOGGER.info("Using parameters:\n%s", pformat(params))
    LOGGER.info("Using configurations:\n%s", pformat(config))

    # ============================== Placeholders ==============================
    placeholder_x = tf.placeholder(
        tf.float32, shape=([None] + params['data_shape']))
    placeholder_z = tf.placeholder(
        tf.float32, shape=(None, params['latent_dim']))
    placeholder_c = tf.placeholder(
        tf.float32, shape=([None] + params['data_shape'][:-1] + [1]))
    placeholder_suffix = tf.placeholder(tf.string)

    # ================================= Model ==================================
    # Create sampler configurations
    sampler_config = {
        'result_dir': config['result_dir'],
        'image_grid': (config['rows'], config['columns']),
        'suffix': placeholder_suffix, 'midi': config['midi'],
        'colormap': np.array(config['colormap']).T,
        'collect_save_arrays_op': config['save_array_samples'],
        'collect_save_images_op': config['save_image_samples'],
        'collect_save_pianorolls_op': config['save_pianoroll_samples']}

    # Build model
    model = Model(params)
    if params['is_accompaniment']:
        _ = model(
            x=placeholder_x, c=placeholder_c, z=placeholder_z, mode='train',
            params=params, config=config)
        predict_nodes = model(
            c=placeholder_c, z=placeholder_z, mode='predict', params=params,
            config=sampler_config)
    else:
        _ = model(
            x=placeholder_x, z=placeholder_z, mode='train', params=params,
            config=config)
        predict_nodes = model(
            z=placeholder_z, mode='predict', params=params,
            config=sampler_config)

    # Get sampler op
    sampler_op = tf.group([
        predict_nodes[key] for key in (
            'save_arrays_op', 'save_images_op', 'save_pianorolls_op')
        if key in predict_nodes])

    # ================================== Data ==================================
    if params['is_accompaniment']:
        data = load_data(config['data_source'], config['data_filename'])

    # ========================== Session Preparation ===========================
    # Get tensorflow session config
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # Create saver to restore variables
    saver = tf.train.Saver()

    # =========================== Tensorflow Session ===========================
    with tf.Session(config=tf_config) as sess:

        # Restore the latest checkpoint
        LOGGER.info("Restoring the latest checkpoint.")
        with open(os.path.join(config['checkpoint_dir'], 'checkpoint')) as f:
            checkpoint_name = os.path.basename(
                f.readline().split()[1].strip('"'))
        checkpoint_path = os.path.realpath(
            os.path.join(config['checkpoint_dir'], checkpoint_name))
        saver.restore(sess, checkpoint_path)

        # Run sampler op
        for i in range(config['runs']):
            feed_dict_sampler = {
                placeholder_z: scipy.stats.truncnorm.rvs(
                    config['lower'], config['upper'], size=(
                        (config['rows'] * config['columns']),
                        params['latent_dim'])),
                placeholder_suffix: str(i)}
            if params['is_accompaniment']:
                sample_x = get_samples(
                    (config['rows'] * config['columns']), data,
                    use_random_transpose=config['use_random_transpose'])
###################
##Sickonet
                sample_x = andmask(sample_x, params['condition_track_idx'])
################

                feed_dict_sampler[placeholder_c] = np.expand_dims(
                    sample_x[..., params['condition_track_idx']], -1)
            sess.run(sampler_op, feed_dict=feed_dict_sampler)

if __name__ == "__main__":
    main()
