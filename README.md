# Grasp Detector

Code to detect planar grasps using the model learnt in 
https://arxiv.org/pdf/1610.01685v1.pdf

![alt tag](http://www.cs.cmu.edu/~lerrelp/all_grasp_img.png)

## Getting Started

This should help you run a learnt grasp detector model on a sample image. Running this code requires the following dependencies:

* Python 2.7
* TensorFlow (version 1.0)
```
# Instructions to install TensorFlow 1.0
# Option 1
Install TensorFlow with GPU support from https://www.tensorflow.org/install/install_linux

# Option 2
# Check available tensorflow wheel files
curl -s https://storage.googleapis.com/tensorflow |xmllint --format - |grep whl | grep gpu
# Install the version that works for your computer
pip install https://storage.googleapis.com/tensorflow/<Replace with wheel name>
```
* argparse ('pip install argparse')
* cv2 ('conda install -c menpo opencv=2.4.11' or install opencv from source)
* numpy ('pip install numpy' or 'conda install numpy')

## Getting Grasp Models

Download the learnt grasp models from https://www.dropbox.com/s/85b483emhubr7l4/Grasp_model?dl=0 and move it to the folder models.

```
# From the repository
wget https://www.dropbox.com/s/85b483emhubr7l4/Grasp_model
mv Grasp_model ./models/.
```

## Example

Run grasp detector that should run the model on the image by sampling patches and displaying the best grasp on the image. Press any key to exit.

```
# For CPU 
python grasp_image.py --im ./approach.jpg --model ./models/Grasp_model --nbest 5 --nsamples 250 --gscale 0.234 --gpu -1

# For GPU
python grasp_image.py --im ./approach.jpg --model ./models/Grasp_model --nbest 5 --nsamples 1000 --gscale 0.234 --gpu 0
```

## Contact
Lerrel Pinto -- lerrelpATcsDOTcmuDOTedu.
