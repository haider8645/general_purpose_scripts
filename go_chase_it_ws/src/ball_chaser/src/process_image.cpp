#include "ros/ros.h"
#include "ball_chaser/DriveToTarget.h"
#include <sensor_msgs/Image.h>
#include <typeinfo>

// Define a global client that can request services
ros::ServiceClient client;

// This function calls the command_robot service to drive the robot in the specified direction
void drive_robot(float lin_x, float ang_z)
{
    // TODO: Request a service and pass the velocities to it to drive the robot

    ball_chaser::DriveToTarget srv;
    srv.request.linear_x  = lin_x;
    srv.request.angular_z = ang_z;
    
    if (client.call(srv))
    {

    }
}

// This callback function continuously executes and reads the image data
void process_image_callback(const sensor_msgs::Image img)
{

    int white_pixel = 255;

    // TODO: Loop through each pixel in the image and check if there's a bright white one
    // Then, identify if this pixel falls in the left, mid, or right side of the image
    // Depending on the white ball position, call the drive_robot function and pass velocities to it
    // Request a stop when there's no white ball seen by the camera
    
    int height{0};
    bool white_ball_detected{false};		
    int pixel_count {-1};
    int step = img.step;	

    int lower_limit_left   {0};
    int lower_limit_middle {step/3};
    int lower_limit_right  {step/2};  

    int upper_limit_left   {(step/3)-1};
    int upper_limit_middle {(step/2)-1};
    int upper_limit_right  {step-1};
   
    for (int i{}; i < img.data.size();i+=3){
	pixel_count++;
	if (pixel_count == step){
		pixel_count = 0;
		height++;
	}

    	if (img.data[i] == 255){

	    white_ball_detected = true;	

            if (pixel_count >= lower_limit_left && pixel_count <= upper_limit_left){
               	//Left
		drive_robot(0.1,0.1);	
            }   	        

            else if (pixel_count >= lower_limit_middle && pixel_count <= upper_limit_middle){
                //Middle
		drive_robot(0.1,0.0);
            }   	        

            else if (pixel_count >= lower_limit_right && pixel_count <= upper_limit_right){
                //Right
		drive_robot(0.1,-0.1);
            }   
	break;	        

       }
       white_ball_detected = false;
	
    }	

    if (white_ball_detected == false){
	drive_robot(0.0,0.0);	
    }
}

int main(int argc, char** argv)
{
    // Initialize the process_image node and create a handle to it
    ros::init(argc, argv, "process_image");
    ros::NodeHandle n;

    // Define a client service capable of requesting services from command_robot
    client = n.serviceClient<ball_chaser::DriveToTarget>("/ball_chaser/command_robot");

    // Subscribe to /camera/rgb/image_raw topic to read the image data inside the process_image_callback function
    ros::Subscriber sub1 = n.subscribe("/camera/rgb/image_raw", 10, process_image_callback);

    // Handle ROS communication events
    ros::spin();

    return 0;
}
