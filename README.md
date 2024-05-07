Overview of the project

Plastic waste detection using YOLOv8 aims to address the pressing environmental issue of plastic pollution by leveraging computer vision technology. This project focuses on detecting various types of plastic waste objects in images and videos, enabling automated monitoring and management of plastic waste in different environments.
Plastic pollution is a significant global challenge that poses serious threats to ecosystems, wildlife, and human health. By accurately identifying and quantifying plastic waste through automated detection, this project aims to support efforts to mitigate the environmental impact of plastic pollution and promote sustainable waste management practices.

Objective of the project

The objective of plastic waste detection is to identify and classify plastic waste items within images or videos. This task serves several purposes and can be approached from various angles, depending on the specific context and objectives. Here are some of the key objectives of plastic waste detection:

The key goals of this project include:

Developing a robust and accurate object detection model using the YOLOv8 architecture to identify plastic waste objects in images and videos.
Providing a user-friendly interface or script for easily running the detection process on new datasets or real-world scenarios.
Empowering researchers, environmental organizations, and policymakers with valuable insights into the distribution, abundance, and characteristics of plastic waste in different regions.

Technologies and tools used

Python 3.10
OpenCV
Numpy
Visual Studio Code
Google Colab
labelImg

Implementation steps

1.	Firstly, Images are gathered from the Open Images dataset - 513 images of plastic bags, 800 images of bottles, and 800 images of tin cans (Note- Images are in jpg format only).
2.	Preprocessing and Annotation is performed. A text file is generated for each image. These files contain the location(s) of object instances in the images together with their class identities. Files contain this information in YOLO format (class id, object centers (x, y), object width, and object height). These numbers are normalized by the real width and height of the images respectively. Text files are generated using a tool – labelImg.

labelImg
Windows + Anaconda
Download and install Anaconda (Python 3+)
Open the Anaconda Prompt and go to the labelImg directory
conda install pyqt=5
conda install -c anaconda lxml
pyrcc5 -o libs/resources.py resources.qrc
python labelImg.py
python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]

3.	Training is done on Google’s colab. Online GPU is utilized to speed up the process. Further, the advantage of pre-trained weights is taken and weights are downloaded and tested after every 2000 iterations. Overall, 6000 iterations are performed i.e., approximately 9 hours of training.

