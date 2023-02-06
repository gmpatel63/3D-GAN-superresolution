from flask import Flask, request, jsonify, make_response
import numpy as np
import base64
import json

import nibabel as nib
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from scipy.ndimage.filters import gaussian_filter
from utils import aggregate

from model import generator

app = Flask(__name__)

upsampling_factor = 2
feature_size = 64
residual_blocks = 8
subpixel_NN = True
nn = False
img_width = 172
img_height = 220
img_depth = 156
reuse = False
checkpoint_dir_restore = '/fs/scratch/PFS0238/gaurangpatel/adversarialML/srgan_output_data/oversample_8_30_e65/ckpt_dir'

# define model
t_orig_input = tf.placeholder('float32', [1, None, None, None],
                              name='original_patches')

t_2d_resize = tf.image.resize(
    t_orig_input, size=(img_width//2, img_height//2))

t_depth_resize = tf.strided_slice(t_2d_resize, [0, 0, 0, 0], [
    1, img_width//2, img_height//2, img_depth], [1, 1, 1, 2])

t_input_gen = tf.expand_dims(t_depth_resize, 4)


srgan_network = generator(input_gen=t_input_gen, kernel=3, nb=residual_blocks,
                          upscaling_factor=upsampling_factor, feature_size=feature_size, subpixel_NN=subpixel_NN,
                          img_height=img_height, img_width=img_width, img_depth=img_depth, nn=nn,
                          is_train=False, reuse=reuse)

# restore g
sess = tf.Session(config=tf.ConfigProto(
    allow_soft_placement=True, log_device_placement=False))

saver = tf.train.Saver(tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope="SRGAN_g"))
saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir_restore))


@app.route('/', methods=["POST"])
def index():
    # print("received request for mri processing")
    data = request.json
    base64_mri = data.get("base64_mri", "")
    # print(f"length of mri string: {len(base64_mri)}")
    mri = np.frombuffer(base64.b64decode(base64_mri),
                        dtype=float)
    # print(f'shape of flattened mri: {mri.shape}, mri type: {mri.dtype}')
    mri = mri.reshape((172, 220, 156, 1))
    mri = np.squeeze(mri)
    # print(f"shape: {mri.shape}")
    
    # start - srgan evaluate
    xt_total = np.expand_dims(mri, axis=0)
    normfactor = (np.amax(xt_total[0])) / 2
    x_generator = ((xt_total[0] - normfactor) / normfactor)
    x_generator = gaussian_filter(x_generator, sigma=1)
    xg_generated = sess.run(srgan_network.outputs, {
                            t_orig_input: x_generator[np.newaxis, :]})

    xg_generated = ((xg_generated + 1) * normfactor)
    volume_generated = xg_generated[0]

    volume_generated = np.squeeze(volume_generated)
    img_volume_gen = nib.Nifti1Image(volume_generated, np.eye(4))
    volume_generated = np.array(img_volume_gen.dataobj)
    # end - srgan evaluate

    # convert mri back to string
    # print(f'shape of volume generated: {volume_generated.shape}, type of volume genrated: {volume_generated.dtype}')
    array_str = base64.b64encode(volume_generated.tobytes()).decode('utf-8')
    # print(f"len of array while sending it back: {len(array_str)}")
    data = {"base64_mri": array_str}
    return data


app.run(host="0.0.0.0", port=5050)
