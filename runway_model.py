# Imports

import numpy as np
import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform

import time
import runway


@runway.setup(options={})
def setup(opts):
    model = models.InceptionV1()
    model.load_graphdef()
    return model


generate_inputs = {
    'z': runway.vector(length=1, sampling_std=0.5)
}

@runway.command('generate', inputs=generate_inputs, outputs={'image': runway.image})
def convert(model, inputs):
    latents = inputs['z']
    param_f = lambda: param.image(128, decorrelate=True)
    print('The current latents ', latents)
    output = render.render_vis(model, "mixed4a_pre_relu:476", param_f, thresholds=(256,),)
    image = output[0].squeeze() * 255
    return {'image': image.astype('uint8')}


if __name__ == '__main__':
    runway.run()