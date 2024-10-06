# basketball-shot-tracker
cv application that tracks basketball shots as makes or misses

## Description of Project

This project aims to track basketball shots as makes or misses, and will offer you a percentage based on how many shots it detects. From a high level, I'll need to run a training set on various basketball hoops and basketball so that my live videostream will be able to pick up the hoop and the basketball in real time. Looking for a dataset will be challenging, but I've found one [here](https://universe.roboflow.com/034-ganesh-kumar-m-v-cs-r2lwe/basketball-lhqoe/dataset/1/images). This dataset provides me with three hundred images, partitioned into 70/20/10 splits. These splits will be useful as we train our model on this dataset to detect for basketball hoop and basketballs.

The fun part of the project is detecting whether or not the shot was a make or miss. My initial idea would be to utilize the YOLO detection algorithm for object detection, as this would be a pretty good start for determining whether or not the ball was a miss or make. One of the key distinguishers of whether or not the ball would be particularly good for me to use as a "basketball" would be if the ball is spherical. I believe that this would have a little bit of computation, as it would have to be able to tell me the first time that the ball is completely below or above the rim.

If the ball is found above the rim initially and then falls outside of the bounds of the net, then it would make sense for it to be a miss. However, if the ball happens to be found in the rim/net area, it must mean that it's a make. I'm a little unclear on how initially I'd perform this, but I'm sure with the YOLO detection I could get a coordinate point and somehow "map" the point to a specific point on the axis. Another key here will be finding and performing the math that is involved with the box surrounding the basket/net. I also think it might be useful for me to draw points on the path of the basketball. If at any point it's above the rim (which we'll have to draw a line to understand what y value that is at) and then falls into the rim (width of the rim), then we're good and we have a basket.

I also want to reiterate the dataset and the lack of data that is out there. Luckily enough, I really only have to detect two objects – the basketball and the basketball hoop. The person and anything else in the frame is somewhat irrelevant. I believe that one of the distinguishing factors is that we'll want to ensure that there's only one hoop in the frame so that the math computation will not get wonky with the basketball distinguishing. I'm really excited to work on this project and am excited to see the final result, which I think will be a really neat culmination of learning various object detection methodologies. I have worked with FasterRCNN a bit in the past for object detection, but am excited to try my hand with YOLO and see what I can make with it!

## Part 2

Prepare a short description of a database that you acquired (no need to upload an actual database into GitHub or Google Drive). Push the report (under "Part 2" section in your readme.md) to your semester project repo by the deadline. Include these elements into your report:

    source (download link and associated paper(s) offering the dataset(s))
    differences between the train and validation subsets, which you think are important from your project point of view
    number of distinct objects/subjects represented in the data, number of samples per object/subject (if applies)
    brief characterization of samples: resolution, sensors used, illumination wavelength, ambient conditions, etc. (whichever applies)


I've acquired two datasets for usage, each with 300 images in each dataset [here](https://universe.roboflow.com/034-ganesh-kumar-m-v-cs-r2lwe/basketball-lhqoe/dataset/1) and [here](https://universe.roboflow.com/rodney-virtualassistant-gmail-com/basketball-annotation-project/dataset/2). These datasets together will give me a fair amount of data to train on just two different objects, a basketball and a basketball hoop. I think both the training and validation subsets are important, but the training is probably more important as I'll want to be able to train my neural network's weights.

These pictures are sometimes captured by camera or were found on the internet, most of them are in daytime ambient conditions or without any color interference with the background. The resolution varies from picture to picture, but is strong enough to determine/decipher a basketball and basketball hoop.
