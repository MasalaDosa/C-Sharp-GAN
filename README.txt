A simple Generative Adversarial Network (GAN) written entirely in c#.

This GAN can (slowly) learn to produce fake MNIST digits.

Run the CSharpGAN program and select a range of digits to train for.
For example: 3, 4 will train the GAN to duplicate the digits 3 and 4.
Alternatively just press enter to train on all digits (this will take a *long* time - at least 2 hours per epoch in my experience).

At the end of each epoch example generated images will be written to a folder based on the local date time when the program was started.

The articles at the following URL were invaluable whilst writing this project.
https://www.tech-quantum.com/learn-coding-neural-network-in-csharp-understanding-what-we-are-going-to-do/

An example image is provided showing results after 5 epochs.

