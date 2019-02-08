
# Function to calculate the Receptive Field of a TCN with L layers, when all dilations
# are chosen to match the Receptive Field of the previos layer to ensure that
# all inputs are targetted once and the Total Receptive Field is maximum.

# INPUTS:
#   filter_size: Size of the filter for the 1D convolution.
#                It is assumed constant filter in every layer

#   layers:     Number of convolutional layers.
#               layers = 1 indicates that there is 1 x 1D convolution.

import numpy as np
import IPython as IP

def TCN_receptive_Field(filter_size, conv_layers):

    k = filter_size
    CL = conv_layers

    receptive_field = 0

    if (CL == 1):                #d1 = 1 by default
        receptive_field = k

    elif (CL == 2):
        d2 = k                  #d2 = k
        receptive_field = k + (k-1) * (d2-1)

    elif (CL > 2):
        d = np.zeros([1, CL])
        d[0,0] = 1
        d[0,1] = k

        for i in  range(2, CL):
            # IP.embed()
            d[0,i] = k + (k-1) * (d[0,i-1]-1) + (k-1)*np.sum(d[0,0:i-1])

        receptive_field = k + (k-1) * (d[0,-1]-1) + (k-1)*np.sum(d[0,0:-1])
        dilation_factors = d
    return receptive_field, dilation_factors


r, d = TCN_receptive_Field(4,4)
print (r)
print(d)


# Receptive Filed of a given convolutional layer
# INPUTS:
#   layer: layer number
#
#   dilations: vector containing the dilations of the previous layers
def  TCN_singleLayer_RF(filter_size, layer, dilations):

    receptive_field
