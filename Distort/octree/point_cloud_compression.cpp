#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/common/transforms.h>

#include <pcl/io/ply_io.h>
#include <pcl/features/normal_3d.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/compression/octree_pointcloud_compression.h>

#include <stdio.h>
#include <sstream>
#include <stdlib.h>
#include <filesystem> 

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
    std::cerr << "[OPTIONAL]: <ShowStatistics> <visualize> " << std::endl; 
    exit(-1); 
  }

  const std::string plydir  = argv[1]; 
  float resolution   = atof(argv[2]); 
  const std::string outdir  = argv[3];  
  const std::string _outdir = outdir.substr(0, outdir.find_last_of("/\\")); 
  std::filesystem::create_directory(_outdir);   
  uint8_t visualize = 0; 
  uint8_t showStatistics = 0; 
  if (argc > 4){ 
    showStatistics = atoi(argv[4]); 
  } 
  if (argc > 5){ 
    visualize = atoi(argv[5]); 
  } 

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

  if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(plydir, *cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    return (-1);
  }

  std::string basename = plydir.substr(plydir.find_last_of("/\\") + 1); 

  // for a full list of profiles see: /io/include/pcl/compression/compression_profiles.h
  pcl::io::compression_Profiles_e compressionProfile = pcl::io::MANUAL_CONFIGURATION;

  pcl::io::OctreePointCloudCompression<pcl::PointXYZRGB>* PointCloudEncoder;
  pcl::io::OctreePointCloudCompression<pcl::PointXYZRGB>* PointCloudDecoder;

  // instantiate point cloud compression for encoding and decoding
  PointCloudEncoder = new pcl::io::OctreePointCloudCompression<pcl::PointXYZRGB> (compressionProfile, showStatistics, 
                                                                                   0.001, resolution, true);
  PointCloudDecoder = new pcl::io::OctreePointCloudCompression<pcl::PointXYZRGB> ();

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
  pcl::PointCloud<pcl::Normal>::Ptr cloudNormals(new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloudAll(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normalEstimation;
	normalEstimation.setInputCloud(cloudOut);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
	normalEstimation.setSearchMethod(kdtree);
	normalEstimation.setKSearch(128);
	normalEstimation.compute(*cloudNormals);


	// Concatenate the fields (PointXYZ + Normal = PointNormal).
	pcl::concatenateFields(*cloudOut, *cloudNormals, *cloudAll);

  std::cout << cloudNormals->size() << " - " << cloudOut->size() << " - " << cloudAll->size() << std::endl; 

  std::cout << cloudAll->points[0].normal[1] << std::endl; 

  pcl::io::savePLYFileASCII(outdir, *pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr (cloudAll));  

  return (0);
}
