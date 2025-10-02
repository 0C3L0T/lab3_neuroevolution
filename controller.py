## Third party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt

from individual import Individual

def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
    individual: Individual
) -> npt.NDArray[np.float64]:
    '''
    this thingy only needs to load the right network and weigths from
    the Individual and run the network. Maybe we can pass the Individual
    as an argument?

    TODO: Maybe we can find some other structure to work as the brain?
    '''

    # Simple 3-layer neural network
    input_size = len(data.qpos)
    hidden_size = 8
    output_size = model.nu

    # init NN
    # load weights
    
    # Get inputs, in this case the positions of the actuator motors (hinges)
    inputs = data.qpos

    # inference (outputs = NN.forward(inputs))
     
    outputs = None

    # Scale the outputs
    return outputs * np.pi