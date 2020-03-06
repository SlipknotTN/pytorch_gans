DCGAN PyTorch official tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
(they don't use DCGAN filters size)

FacebookResearch DCGAN (to check shapes): https://pytorch.org/hub/facebookresearch_pytorch-gan-zoo_dcgan/

DCGAN TensorFlow official tutorial: https://www.tensorflow.org/tutorials/generative/dcgan

Using these shapes: http://www.timzhangyuxuan.com/project_dcgan/


Interesting insights about DCGAN:
- DCGAN architecture. D starting with 128 filters: G loss 0.0, D doesn't learn, always answer real.
- Same architecture, but D starting with 64 filters: G has struggle and it learns.
- Why D with 128 filters doesn't get always the correct answer? And simplyfing architecture it start to work?!

Interesting insights about Conditional DCGAN:
- Starting with 1024 G channels doesn't work at all, D loss nearly 0.
- Starting with 128 G channels starts to work, again a simpler G is better for D, but it collapse easily (D loss nearly 0)