DCGAN PyTorch official tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
(they don't use DCGAN filters size)

DCGAN TensorFlow official tutorial: https://www.tensorflow.org/tutorials/generative/dcgan

Using these shapes: http://www.timzhangyuxuan.com/project_dcgan/

Interesting insights:
- DCGAN architecture. D starting with 128 filters: G loss 0.0, D doesn't learn, always answer real.
- Same architecture, but D starting with 64 filters: G has struggle and it learns.
- Why D with 128 filters doesn't get always the correct answer? And simplyfing architecture it start to work?!