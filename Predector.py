import sys
import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import pylab
from copy import copy
import warnings
import json
import cv2
import matplotlib.image as mpimg
from Tkinter import *  



caffe_root = '../../caffe/' 
sys.path.append(caffe_root + 'python')
import caffe

caffe.set_mode_cpu()

model_def     = 'trainnet.prototxt'
model_weights = 'Pascal_Model.caffemodel'

labels = np.asarray(['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
					  'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
					  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])

net = caffe.Net(model_def, model_weights, caffe.TEST)


mu = np.array([104.0069879317889, 116.66876761696767, 122.6789143406786])  
print 'mean-subtracted values:', zip('BGR', mu)


# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227) 



def classifier(filename):
	
	print(filename)
	image = caffe.io.load_image(filename)
	#image = caffe.io.load_image('03.jpg')
	transformed_image = transformer.preprocess('data', image)
    
	net.blobs['data'].data[...] = transformed_image

	### perform classification
	output = net.forward()

	output_prob = output['score'][0]  # the output probability vector for the first image in the batch

	print 'predicted classs is:', output_prob.argmax()
	print 'output label:', labels[output_prob.argmax()]
	top_inds = output_prob.argsort()[::-1][:5]  
	#print top_inds
	print zip(output_prob[top_inds], labels[top_inds])

	print '-------------'

	gui_out_put(output_prob)

	pass


def gui_out_put(output):
	
	top = Tk()
	Lb1 = Listbox(top)
	top_inds = output.argsort()[::-1][:5]  

	for x in xrange(0,5):
		Lb1.insert( x, str(output[top_inds][x]) + " -- " + str(labels[top_inds][x]) )
		pass
	Lb1.pack()
	top.mainloop()
	
	pass

for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)

for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
'''
def vis_square(data,name):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')
    mpimg.imsave( name+".png", data)


labels = np.asarray(['conv1', 'pool1', 'norm1', 'conv2', 'pool2', 'norm2',
						'conv3', 'conv4', 'conv5', 'pool5'])


for label in labels :
	try:

		feat = net.blobs[label].data[0, :40]
		vis_square(feat, 'vis/' + label + 'B')
		pass
	except Exception, e:
		print label + "---- X (blobs)"

	try:
		filters = net.params[label][0].data
		vis_square(filters.transpose(0, 2, 3, 1), 'vis/' + label + 'P')

		pass
	except Exception, e:
		print label + "---- X (filter)" 
	pass

'''