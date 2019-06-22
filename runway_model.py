# Imports


import os
import numpy as np
import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform
import time
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import runway
import random

@runway.setup(options={})
def setup(opts):
    model = models.InceptionV1()
    model.load_graphdef()
    return model


generate_inputs = {
    'z': runway.vector(length=1, sampling_std=0.5),
    'steps': runway.number(min=64, max=512, step=64, default=256)
}

@runway.command('generate', inputs=generate_inputs, outputs={'image': runway.image})
def convert(model, inputs):
    start = time.time()

    #set up parameters
    #neuron = int(np.clip(np.float(inputs['z']) * 1000, 0, 507))
    neuron = random.randint(0, 506)
    steps = int(inputs['steps'])

    #start rendering the images
    param_f = lambda: param.image(512, decorrelate=True)
    output = render.render_vis(model, "mixed4a_pre_relu:"+str(neuron),
                     param_f, thresholds=(steps,), verbose = False)

    #logging
    elapsed = time.time() - start
    print(f'neuron {neuron}: steps: {steps} time: {elapsed}')

    #output results
    image = output[0].squeeze() * 255
    return {'image': image.astype('uint8')}


if __name__ == '__main__':
    runway.run()