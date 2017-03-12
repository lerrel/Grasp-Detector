Code to detect grasps using the model learnt in 
https://arxiv.org/pdf/1610.01685v1.pdf

Dependencies:   Python 2.7
                tensorflow (version 0.9)
'''
#Check available tensorflow wheel files
curl -s https://storage.googleapis.com/tensorflow |xmllint --format - |grep whl | grep cpu/tensorflow-0.9
#Version I use is linux/cpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
#Install the version that works for your computer
pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
'''
                argparse ('pip install argparse')
                cv2 ('conda install -c menpo opencv=2.4.11' or install opencv from source)
                numpy ('pip install numpy' or 'conda install numpy')

Run grasp detector that should run the model on the image by sampling patches and displaying the best grasp on the image. Press any key to exit.
'''
python grasp_image.py --im ./approach.jpg --model ./models/Grasp_model --nbest 100 --nsamples 250 --gscale 0.234
'''
