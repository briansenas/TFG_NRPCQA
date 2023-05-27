#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/common/transforms.h>

#include <pcl/io/ply_io.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/compression/octree_pointcloud_compression.h>

#include <stdio.h>
#include <sstream>
#include <stdlib.h>

#include <string.h>    
#include <iostream>
#include <string> 
#include <vector>

#ifdef WIN32
# define sleep(x) Sleep((x)*1000)
#endif

int main (int argc, char** argv)
{
  if (argc <= 3) {
    std::cerr << "[ERROR]: You must specify a directory for the PLY file, "
              << "the resolution and the output directory" << std::endl; 
    std::cerr << "./point_cloud_compression <plydir> <resolution> <outputdir>" << std::endl; 
    std::cerr << "[OPTIONAL]: <visualize> " << std::endl; 
    exit(-1); 
  }

  std::string plydir  = argv[1]; 
  float resolution   = atof(argv[2]); 
  std::string outdir  = argv[3];  
  
  uint8_t visualize = 0; 
  if (argc > 4){ 
    visualize = atoi(argv[4]); 
  } 

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

  if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(plydir, *cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    return (-1);
  }

  std::string basename = plydir.substr(plydir.find_last_of("/\\") + 1); 
  std::cout << "Loaded "
    << cloud->width * cloud->height
    << " data points from " << basename 
    << std::endl;

  bool showStatistics = true;

  // for a full list of profiles see: /io/include/pcl/compression/compression_profiles.h
  pcl::io::compression_Profiles_e compressionProfile = pcl::io::MED_RES_ONLINE_COMPRESSION_WITH_COLOR;

  pcl::io::OctreePointCloudCompression<pcl::PointXYZRGB>* PointCloudEncoder;
  pcl::io::OctreePointCloudCompression<pcl::PointXYZRGB>* PointCloudDecoder;

  // instantiate point cloud compression for encoding and decoding
  PointCloudEncoder = new pcl::io::OctreePointCloudCompression<pcl::PointXYZRGB> (compressionProfile, showStatistics, 
                                                                                   0.01, resolution);
  PointCloudDecoder = new pcl::io::OctreePointCloudCompression<pcl::PointXYZRGB> ();

  // Calculate the centroid of the point cloud
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*cloud, centroid);

  // Calculate the transformation matrix for axis alignment
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform.block<3, 1>(0, 3) = -centroid.head<3>();

  // Apply the transformation to the point cloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::transformPointCloud(*cloud, *aligned_cloud, transform);


  std::stringstream compressedData;
  // output pointcloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudOut (new pcl::PointCloud<pcl::PointXYZRGB> ());

  const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud_ (cloud); 
  // compress point cloud
  PointCloudEncoder->encodePointCloud (cloud_, compressedData);

  // decompress point cloud
  PointCloudDecoder->decodePointCloud (compressedData, cloudOut);

  if (visualize){
    pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
    viewer.showCloud (cloudOut);
    while (!viewer.wasStopped ())
    {
    }
  }
    
  return (0);
}
