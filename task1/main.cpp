#include <iostream>
#include <fstream>

#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage:\n";
        std::cerr << "./shot_descriptor cloud.ply keypoints.ply output.csv\n";
        return -1;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPLYFile(argv[1], *cloud) == -1 || 
        pcl::io::loadPLYFile(argv[2], *keypoints) == -1)
    {
        std::cerr << "Failed to load input files\n";
        return -1;
    }

    // 1. Compute Normals (SHOT requires normals)
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(0.02); // Adjust normal radius according to your voxel size
    ne.compute(*normals);

    // 2. Compute SHOT descriptors
    pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
    pcl::PointCloud<pcl::SHOT352>::Ptr descriptors(new pcl::PointCloud<pcl::SHOT352>);
    
    shot.setRadiusSearch(0.05); // Adjust feature radius (usually ~2.5x to 5x voxel size)
    shot.setInputCloud(keypoints);
    shot.setSearchSurface(cloud);
    shot.setInputNormals(normals);
    shot.compute(*descriptors);

    // 3. Save descriptors to CSV
    std::ofstream out(argv[3]);
    for (size_t i = 0; i < descriptors->size(); ++i)
    {
        for (int j = 0; j < 352; ++j)
        {
            out << descriptors->at(i).descriptor[j];
            if (j < 351) out << ",";
        }
        out << "\n";
    }
    out.close();

    std::cout << "Computed " << descriptors->size() << " SHOT descriptors\n";

    return 0;
}