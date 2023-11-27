# dashtoon_ Neural Style Transfer 
generative AI assignment and a new method

                                              DASHTOON campus placement assignment 
                                              Generative AI (neural style transfer)
Without wasting much of time in reading the theory I’ll directly explain some important points of my implementation and ideas : 

                                                    3 methods: 
                            
I tried to solve the problem in 3 different ways which include: 

1) the traditional way of using pre trained VGG 19 model for the feature extraction
2) the CNN based deep learning model for feature extraction from content and style images.
3) *Combination of Style transfer + pix2pix model (GAN)
                                                         
                                                    Notebooks

Total of 2 notebooks are attached

1) notebook 1 has the implementation of 1st and 2nd method together
   
2) notebook 2 consist of my newly created approach i.e., style transfer + pix2pix model.

                                           Details of used methods:

                                          1) VGG19 pretrained model :
 
VGG 19 model is a large computer vision model which can be used to extract the features from the images, by removing the last layers of the network,
It is capable of extracting low level as well as high level features of the network, as far as the networks gets deeper, we can find the high level features that’s why the content images features are taken from the last convolution layer while the style image’s features are extracted from first convolution layer of each block of network.

    Issues : 

although this method is highly accurate and can generate the satisfying results but the issue with this method is that the pretrained model like inception net, vgg, resnet are very large in size and can consume high RAM memory, which makes  this method hard to implement on live server because it takes time to get downloaded.

                                     2) CNN based deep neural network architecture: 

This time I used the autoencoder based architecture design, where I removed the decoder part and only used the encoder part to extract the features from the images, the method is still similar as of the previous approach. 

    Issues :

The issues with this method is, the accuracy is not up to the mark because we have to train the model to extract the features of images
The plus point of this method is that we can make an optimized small sized model for the feature extraction, in this assignment the model which I designed was of almost 1.5MB, which is roughly 15 times less than the pretrained model. hence the loading time of model is almost negligible.

                                      3) neural style transfer + pix2pix model combination 

I designed this approach from my past work experience, the reason for using this combination are :

•	we can solve some particular type of problems where the unnecessary things are not required, for example if we want to make the realistic looking water colour paintings, or want to apply only particular type of transitions to the image etc. 

•	GAN based methods are more accurate and can generate realistic images.

•	we can design the generator model according to our system requirement which can be more dynamic.

           How this method works : 

    Data preparation : data preparation is one of the most important task in generative models

•	I used the open source coco wiki art NST dataset from Kaggle (https://www.kaggle.com/datasets/shaorrran/coco-wikiart-nst-dataset-512-100000) which consists images of contents and styles. 

•	then I used tensor hub model’s google magenta model to generate the labels for my model training (model link : https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2 ) by giving it style and content images of the dataset (it took me around 10-15 sec to generate 600 images ) 

•	the generator is having Unet architecture which preserve a lot of image information and also minimize the chances of overfitting or vanishing gradients.

•	discriminator have patch Gan based architecture which benefits in better comparison of the generated and actual image by checking patch by patch part of image.

•	L1 loss and binary cross entropy loss are used, L1 is used to generate the highly realistic output images with no blur. Apart from this we can also use pre trained models like VGG 19 to extract the feature from target images and the compare it with generated images to obtain the best results.

•	Finally, I trained the model for 50 epochs and I got some results, results were not up to the mark but this can be a future approach for the generation of some particular art designs. 




                               Possible Solutions for Improvement of GAN architecture : 

There are still several issues in this type of method due to the large size of GAN model and of the large training time, which can be solved by using these steps(possibly)

•	instead of using less accurate unet based architecture in Generator we can use some resnet inspired architecture.

•	we can also try to use transformer-based generator which are having high accuracy in generating state of the art images.

•	use of some more advanced loss function can also benefit like perceptual loss and adversarial loss or the combination of these loss by using some constant parameters like lambda to get more perceptually clear images.





                                                             Summary 
all this ideas and implementations are the outputs of my past work experience in computer vision domain, in my 3 month of internship I did rigorous research work to enhance the generative image quality by removing the artifacts like JPEG Artifacts, image deblur, staircase artifacts etc. apart from these I knew how important is the model size and parameters for the memory consumption of and time processing of low end devices, hence in this assignment I put focused on designing the light weight model architecture, which I somewhat successfully completed. 
I was able to produce the results by using the above-mentioned methods results are not state of the art but they need some optimization, all the work I have done is the output of two days, hopefully by doing a bit of more research in generator model architecture we can create a noble deep learning model.








