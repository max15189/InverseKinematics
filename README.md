# InverseKinematics with NN for 6DOF robot
This is a learning project that i was inspire to try after learning about modern robotics 

The goal of this project is to train a NN to predict the joint configuration of a  Viper300x 6DOF robot given the  desired tranformation matrix of the End Effector.The predicted configuration will be used as a warm start for the clasical numerical netwon raphson approach.The aim is to reduce the number of iteration needed for convergence . in the current state it is effective when the target pose is far from the current joint configuration.

The features are : position of the end effector , first 2 rows of the rotation matrix(the third is not needed as it contains no extra information) and the current configuration of the EE

