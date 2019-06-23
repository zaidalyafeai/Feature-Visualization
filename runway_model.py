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
import runway
import random

@runway.setup(options={})
def setup(opts):
    model = models.InceptionV1()
    model.load_graphdef()
    return model


generate_inputs = {
    'z': runway.vector(length=1, sampling_std=0.5),
    'layer': runway.category(choices=["mixed3a", "mixed4a", "mixed5a"]
    , default="mixed4a"),
    'steps': runway.number(min=64, max=512, step=10, default=128)
}

@runway.command('generate', inputs=generate_inputs, outputs={'image': runway.image})
def convert(model, inputs):
    print(inputs['z'])
    num_neurons = {'mixed3a':255, 'mixed4a':507, 'mixed5a':831}
    start = time.time()

    #set up parameters
    layer = inputs['layer']
    neuron = int(np.clip(np.float(inputs['z']) * 1000, 0, num_neurons[layer]))
    steps = int(inputs['steps'])
    

    #start rendering the images
    param_f = lambda: param.image(512, decorrelate=True)
    output = render.render_vis(model, layer+":"+str(neuron),
                     param_f, thresholds=(steps,), verbose = False)

    #logging
    elapsed = time.time() - start
    print(f'layer: {layer} neuron {neuron}: steps: {steps} time: {elapsed}')

    #output results
    image = output[0].squeeze() * 255
    return {'image': image.astype('uint8')}


if __name__ == '__main__':
    runway.run()