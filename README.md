<h2>Introduction</h2> 

Self-driving cars or autonomous cars have been receiving tremendous attention due to Deep Neural Network (DNN) that automate a lot of manual tasks of driving the car. Autonomous cars are vehicles that are capable of sensing their environment and moving safely with little or no human input. The vehicles can interpret sensory information to identify appropriate navigation paths, as well as obstacles and relevant signage. The main motivation is to minimise road accident due to human error or fatigue while driving. 

There a six different level of autonomous classification:

• Level 0: All functionality and systems of the car are controlled by humans. No automation.

• Level 1: Minor things like cruise control, automatic braking, or detecting something in the blind spot may be controlled by the computer, one at a time

• Level 2: A human is still required for safe operation and emergency procedures. The computer can perform some simultaneous automated functions, such as acceleration and steering.

• Level 3: The computer can control all critical operations of the car simultaneously including accelerating, steering, stopping, navigation and parking under most conditions. A human driver is still expected to be present in case they are alerted of an emergency.

• Level 4: The car is fully autonomous, without any need for a human driver, in some driving scenarios (i.e snowing). 

• Level 5: The car is completely fully autonomous and capable of self-driving in every situation

![SAE3016](https://github.com/chanhanxiang/jetbot/assets/107524953/96b8fd5b-3a94-4647-978a-52722f65a135)

In this project, the car will be simulated using Jetson Nano Bots and the road that cars are driven on are simulated with a black single lane as the marker that the car would track on. The car will be self-driven on the road based on the video stream captured by the front-facing camera on the car.

<h3>Problem statement</h3>

Problem Articulation: 

We will simulate a self-driving car and develop a deep learning model to detect the lane and map the position of the lane to direction to move (left or right) to keep the car on track of the road. 

The objectives of this project are:

i)	Develop a NN-based model that track the lane line.

ii)	Develop steering prediction algorithm to keep the car on the road.

iii)	Develop a NN-based acceleration/deceleration of the car. The car would accelerate to max road speed if the road is straight and de-acceleration when it is turning at the corner (left/right corners).

iv)	Develop a NN-based traffic light and road sign detection and change the speed of the car to obey the traffic rules.

The problem is best framed as a both a object detection task which predicts lane directions and traffic signs/lights classes as well as the location and dimensions of the objects indicated by bounding boxes. We will be starting with Tensorflow Object Detection model (SSD-Resnet50) which comprises of Single-Shot MultiBox Detector (SSD) for object detection and ResNet 50 for classification in the bounding box. This model serves as baseline when experimenting with more complex models.

<h2>Preprocessing</h2>

First, the image dataset for self-driving car is generated with the front camera on Jetson Nano Bot. These images can be classified by five classes of objects. They are: Left, Right, Straight, Traffic Light (Red, Yellow and Green) and Traffic sign (Stop etc). Below show images collected for different classes:

![Screenshot from 2024-02-19 11-05-49](https://github.com/chanhanxiang/jetbot/assets/107524953/941902df-73e3-425f-ada7-c3e5b1ef62ae)

The image dataset is annotated and used to train on the Tensorflow Object Detection model (SSD-Resnet50) to detect the lane, traffic light and traffic sign. We will be using Single-Shot MultiBox Detector (SSD) for object detection and ResNet 50 for classification in the bounding box. The figure below shows the block diagram on how the self-driving car perform lane detection and steering of the wheels.

<h3>Deep learning model</h3>

The trained model is deployed to the Jetson Nano Bot and the car is placed on a lane to begin driving around the track. The images captured by the front camera are resize to 224 x224 pixel before sending to the SSD-ResNet50 model for inferencing. It will detect objects like straight lane, left corner, right corner, traffic light and it’s state (red, yellow or green), and traffic sign such as stop sign. The model will give an output with 5 parameters for each object detected: Class, top, bottom, left and right pixel position. 

A prediction module will analyse the objects that were being detected and return command for driving direction such as “Turn Left 30 degree” or “Drive Straight”. When the lane is straight, the module can determine if it is able to accelerate the car. If a corner is detected far away, it will maintain the speed of the car. When the corner is near, it will return command “Apply Brake” to slow down the car to turn safely around the bend/corner.

If traffic light is detected ahead, it will interpret the state of the traffic light such as red, yellow or green and return command such as “applied brake”. If traffic sign such as stop sign is detected, it will infer the distance from the car and return command such as “applied brake” when the sign is near so that the car will stop just before the stop sign.

The commands from the prediction module are sent to the steering module which will convert the command to low-level steering wheel signal. If command such as “Accelerate” or “Apply brake” is received, the steering module will sent accelerate or brake signal to the Jetson Nano Bot.

![Screenshot from 2024-02-19 11-07-43](https://github.com/chanhanxiang/jetbot/assets/107524953/f40e1bb1-5c8d-4b17-8511-a00ee2d14074)

To test the deep learning module for the self-driving car, a test track on the left is created for testing purpose. The car will be tested by running on this track. We will monitor whether the car is able to navigate through the course.

We may also experiment with other models in the TensorFlow 1 Detection Model Zoo such as “SSD Inception” or “SSD mobilenet” as well as more accurate models in TensorFlow 2 Detection Model Zoo such as EfficientDet and CenterNet (which are not SSD based) and family of R-CNN nets, if time permits.

<h5>Performance tuning and optimisation</h5>

Below are some of the things we can do to improve performance/accuracy of the model:

- Fine-tune the model by performing a transfer learning with pre-trained model from TFOD model zoo.

- Remove the last fully-connected layer and change the number of classes detected to the number of objects acquired in the self-driving dataset.

- Set all the classification layers in SSD to be trainable.

- Set some of top Convolution layers to be trainable since our objects are very different from the images acquired from pre-trained CoCo dataset.

- Set more top Convolution layers to be trainable to further improve on the accuracy if necessary.

- Add batch normalisation layer after each convolution layers.

<h5>Application-based integration</h5>

During post-training quantization, TensorFlow Lite Converter will be used to convert the trained model in Tensorflow (saved model) or Keras (h5) formats to TensorFlow Lite format, thereby reducing the model size while also improving CPU and hardware accelerator latency, with little degradation in model accuracy.

![Real time obj](https://github.com/chanhanxiang/jetbot/assets/107524953/90c80147-406b-47f4-8957-e1b831a87c24)

<h2>Data collection</h2>

In this project, we will be doing data collection using the Jetson Nano Bot to generate the dataset with game controller as shown in the figure below. The live camera feed from the onboard lens will capture images of what is in front of the jetbot car (see below left). This can be done by using jupyter notebook to control the car movements (forward, backward, stop, left and right) and then take pictures at various positions where the car is in. The use of ipywidgets serves as a way for us to interact with the car’s movement controls and facilitate the image capturing. 

![Screenshot from 2024-02-19 11-12-40](https://github.com/chanhanxiang/jetbot/assets/107524953/c985576b-d0f3-4afb-8c03-b5d07471e294)

Ipywidge Controller is configured with the following joystick/button setting as shown below:

![Screenshot from 2024-02-19 11-12-23](https://github.com/chanhanxiang/jetbot/assets/107524953/0ab85528-200d-4831-b810-33ed266e0087)

Note: The USB thumbdrive to communicate wirelessly with the gamepad controller must been inserted to the client machine on the remote side running Jetson Nano Bot’s Jupyterlab.

Data collected from the images are placed into the respective folders representing different classes of object. These images can be classified by five classes of objects. Other objects such as Traffic Light and Traffic sign will be added soon. Traffic Light can be further classified as Red Light, Yellow Light and Green Light. Traffic sign can have Stop sign, Right Turn, No Entry etc. Images are in jpeg format.

For inference using regression method, the images gathered for training dataset are 1000 images.
For inference using Object Detection method, the image gathered for training dataset are 816 images and validation dataset are 20 images.
The test dataset for both methods are 168 images.

Image pre-processing is performed on these image capture by Jetson Nano Bot. These images are resized to 224 x 224 pixels and reshape to (1, 224,224, 3) to add a new dimension for batch size. Augmentation is used to improve the training accuracy by generating additional images with random horizontal flip.

<h2>Annotation</h2>

![Screenshot from 2024-02-19 11-15-07](https://github.com/chanhanxiang/jetbot/assets/107524953/7b9ebe70-9186-4dbb-95e5-5cab7e08de34)

For the regression method, the image is annotated using the gamepad controller to move the cursor to identify/locate the spot where the desired direction for the Jetson Nano Bot to moved towards (see above left figure). The annotated position given by the x-coordinate and y- coordinate are written to the filename of the image. For example, the filename is called “xy_045_097_300e4e88-63cc-11eb-b891-72b5f773b75d” where x = 45 and y = 97. A unique identifier is appended to the back of the filename so that all filename would be unique. A total of 1000 images are annotated.

To label objects in the object detection method, the objects in the image are labelled by drawing a bounding box over the area of the desired object (see above right figure) using LabelImg. The targeted class of that object is selected and this information is stored as an XML file in Pascal VOC formal. A total of 816 images are annotated for the training dataset.

TFOD requires dataset to be saved in Tfrecord format. To create the required TFRecord for training set and testing set from the image files and xml files in training and testing directory respectively. After tfrecords are created, These tfrecords (“project_train.record-00000-of-00001” (816 images) and “project_test.record-00000-of-00001” (20 images) they are stored in the data directory.

<h2>Implementation plan</h2>

We intended to do 3 modules. For the road following, we decided to try on Deep Learning on Regression model and Object Detection. For the Object of Pedestrian, Vehicle, Traffic Sign and Traffic Lights detection, we intended to combine with existing road following so that the solution is as a whole.

1. Navigation using regression

2. Navigation using Object Detection

3. Navigation using Pedestrian, Vehicle, Traffic Sign and Traffic Lights detection

<h5>1. Navigation using regression</h5>

Initially, we start off using classification of 3 classes (left, center, right). After experimenting with different combination of motor steering power and speed, the most it can have 2 rounds of successful navigation but will soon it will run off the tracks unless we have more classes to define different turning angle depending on the car lane position. It was found that either regression or objection detection can be used to determine the left and right motor speed. 

Since regression is applied, we will need to get the desired target location of x and y where the desired direction for the car to move toward to. Based on the official paper from the Nvidia and Jetbot video in the youtube, using a gamepad/joystick would be easier help to pin the location and capture the x and y co-ordinate and save as part of the file name of the image. It was decided to invest to get a joystick to help us on data collection. The detail of data collection is discussed in the “Data Collection” section.

After collected the data to about 1000+ images, the preferable model shall be trained using GPU power as we have a lot of image to process it using NN model. Since Jetbot has some limitations, we need to have a model which is suitable for mobile device to fix in and powerful enough to be train and predict the result. From the official web site, we notice there is pre-trained model (ResNet18 model trained on millions of images) for a new task that has possibly much less data available. This pre-trained ResNet-18 model is decided to be used for PyTorch TorchVision. 

To use of transfer learning for the regression model for road navigation; The robot takes in a frame from the camera stream and the model’s goal is to get (x, y) coordinate for the green dot which the robot follows. Description of the ResNet18 model:

![Screenshot from 2024-02-19 11-22-27](https://github.com/chanhanxiang/jetbot/assets/107524953/d300ac52-0891-47f3-871a-fed0ea650414)

After extracting the (x, y) coordinates from the name of each picture path from the images that were taken in the data collection, pre-processing is done. The pre-processing function returns the image with a tensor of the (x, y) coordinates of the green dot, the image and its label.

After dataset has been prepared, training, testing and evaluation to find out the accuracy of the model is done. Training the regression model, for this the image and the (x, y) label is loaded onto ResNet18. For each epoch iteration, the loss with the mean squared error function, and the optimization is with Adam optimizer is calculated. Then make some fine tuning to the regression model, good results in the training of the model can be achieved. The accuracy on the test set almost approaches that of the training set. The changes were increasing the test set slightly, reducing the jitter distortions and increasing the length of training.

![Screenshot from 2024-02-19 11-28-03](https://github.com/chanhanxiang/jetbot/assets/107524953/9b629b53-c2e6-4677-a7b2-614b9f8eabb2)

After training the model, quantization of the model needs to be done so that the performing computations and storing tensors at lower bit-widths than floating point precision will allow Jetson Nano to have a more compact model representation and the use of high performance vectorised operations.

After loading the model onto Jetbot, fine turning of the Jetbot based on the speed gain, steering gain needs to be done. As such ipwidgets tools has to be used to help us interactively adjust it. Not only that, the Jetbot wheel motor speed values has to be calculated. Thanks to Nvida official web site, they provide formulae to calculate it as follows:

Getting the angle using arc tangent of x and y co-ordinates:

Based on the angle from the center of the Jetbot and sum the angle of steering gain plus the differences of current angle and previous angle and multiple with D gain to get now much to increase or decrease the left and right motor.

    speed_slider.value = speed_gain_slider.value
    angle = np.arctan2(x, y)
    pid = angle * steering_gain_slider.value + (angle - angle_last) * steering_dgain_slider.value
    angle_last = angle
    steering_slider.value = pid + steering_bias_slider.value
   
    robot.left_motor.value = max(min(speed_slider.value + steering_slider.value, 1.0), 0.0)
    robot.right_motor.value = max(min(speed_slider.value - steering_slider.value, 1.0), 0.0)

![Screenshot from 2024-02-19 11-28-21](https://github.com/chanhanxiang/jetbot/assets/107524953/dc713029-85ed-4f49-af0a-6e3e6857e65b)

Based on NVIDIA stats, TensorRT is a better high-performance deep learning inference than just using it on the quantized file in Jetbot. Surprisingly, the Jetbot with TensorRT engine can run up to max speed of 0.31 with no off the track running Jetbot for half an hour. As for Jetbot without TensorRT engine (only Quantization), the max speed is only of 0.21 with no off the track running Jetbot for half an hour. This proves that the TensorRT is a much better high-performance deep learning inference than the quantized model.

<h4>2. Navigation using Object Detection</h4>

The source code came from Nvida, with only slight modification to suit this exercise's requirement. Enhancement is done to include object detection of Pedestrian, Vehicle, Traffic Sign and Traffic Lights detection. Since there is only a single lane on the test track, experiment to manoeuvre the car to another lane cannot be carried out. Therefore, some of the Traffic signs for lane changing are not suitable. We can only set the car to go forward or to stop. Based on that, we have two classes namely stop or go and we decide to do the binary classes of classification NN model. NVIDIA also provides functions for collision avoidance, from which it was taken and customised for this project. For the collision avoidance of the object detected, it will either move left or right based on customised design. For this project, instead of manoeuvring the car to another direction, the car stops by object detection such as Traffic light (Red, Amber) and Traffic sign such as Stop sign etc. Here are some of the collected data samples:

![Screenshot from 2024-02-19 12-03-59](https://github.com/chanhanxiang/jetbot/assets/107524953/0b69fda0-087b-4444-8d10-fb0979e7a704)

Likewise, pre-trained ResNet-18 model with 1000 class labels is also used. Since the classes labels only have two classes (Blocked and Free), the final layer is replaced with a new untrained layer that only has two outputs and train with “Blocked” and “Free” image. After training the model, the model is converted into TensorRT format.

In the Jetbot, there are 2 models (Road Following and Road Obeying which follow traffic sign, traffic light and avoid collision). A check is created to see if there is blocked object detection detect based on the probability of more than 80%, the robot stops, else it will continue with road following. Below is the source code to choose between road following or road object detection:

![Screenshot from 2024-02-19 12-06-45](https://github.com/chanhanxiang/jetbot/assets/107524953/148f8d60-f43a-46ef-a0d4-f78eed271891)

There seems to be a latency issue in camera and upon investigation, too many images were captured which caused a lot of backlog. Hence the root cause for camera lag and sometimes, Jetbot crashes due to error log full as it could not keep up the log recycle. To resolve this issue, unnecessary logs can be disabled by entering the following commands in the Jetbot terminal:

sudo service rsyslog stop
sudo systemctl disable rsyslog

Another solution is to use Zmq camera instead of Jetbot camera library. The Zmq camera can be run using a python script from the backend. This can reduce the screen speed to capture 21fps and clean up excess screen input so that there is no backlog and screen capture will be real time. Also this resolves the camera lag issues and greatly improves the speed of the Jetbot by increase the speed gain up to 0.7.

<h4>3. Object Detection</h4>

In the second part of self-driving car project, object detection for road tracking shall be tried out. The advantage is that it can do more tasks such as it can perceive the environment and act accordingly in real time. The car can accelerate/decelerate based on the condition of the road by adjusting the power of the motor on the left and right wheels. It can detect objects like traffic lights, signs and pedestrians using the same DL model.

The cons being time taken to perform inferencing may exceed the limit due to the hardware platform used. In future, we can use better GPU on devices, a different quantisation method to optimise further and using Nvidia TensorRT for models.

Applications suitable for deployment are: robots in the factory floor to transfer goods, cleaning and disinfection of shopping center and workplace, and patrolling of building complex for security and abnormalities in premise.

Three pre-trained models (SSD MobileNet V2, SSD MobileNet V1 FPN and SSD ResNet50) were tried out. They are selected because SSD are the only detector support by TFLite in TensorFlow V2.3.1 for conversion to tflite files. MobileNet is selected due to the speed of the inference and ResNet50 for its accuracy in detection of the objects. 

Transfer learning is also performed by initializing the model with the pre-trained weights and perform the training on all the classification layers and detection layers.  The number of classes it can detect are 5 classes: Straight, Approach corner, Right Corner, Exit Corner and Exit Corner Completed as shown on the right.


