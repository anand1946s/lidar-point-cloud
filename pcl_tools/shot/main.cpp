#include <cmath>
#include <fstream>
#include <iostream>

#include <pcl/common/point_tests.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

namespace
{
bool hasSuffix(const std::string& value, const std::string& suffix)
{
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int loadPointCloud(
    const std::string& path,
    pcl::PointCloud<pcl::PointXYZ>& cloud)
{
    if (hasSuffix(path, ".pcd"))
    {
        return pcl::io::loadPCDFile(path, cloud);
    }

    return pcl::io::loadPLYFile(path, cloud);
}
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage:\n";
        std::cerr << "./shot input.ply keypoints.ply output.csv\n";
        return -1;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(
        new pcl::PointCloud<pcl::PointXYZ>);

    if (loadPointCloud(argv[1], *cloud) == -1 ||
        loadPointCloud(argv[2], *keypoints) == -1)
    {
        std::cerr << "Failed to load input file\n";
        return -1;
    }

    if (keypoints->empty())
    {
        std::cerr << "Keypoints cloud is empty\n";
        return -1;
    }

    if (cloud->empty())
    {
        std::cerr << "Input cloud is empty\n";
        return -1;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr finite_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    finite_cloud->reserve(cloud->size());
    for (const auto& point : *cloud)
    {
        if (pcl::isFinite(point))
        {
            finite_cloud->push_back(point);
        }
    }
    cloud = finite_cloud;

    for (const auto& point : *keypoints)
    {
        if (!pcl::isFinite(point))
        {
            std::cerr << "Keypoints contain NaN or infinite coordinates\n";
            return -1;
        }
    }

    if (cloud->empty())
    {
        std::cerr << "Input cloud contains no finite points\n";
        return -1;
    }

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(0.02);
    ne.compute(*normals);

    pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
    pcl::PointCloud<pcl::SHOT352>::Ptr descriptors(
        new pcl::PointCloud<pcl::SHOT352>);

    shot.setRadiusSearch(0.05);
    shot.setInputCloud(keypoints);
    shot.setSearchSurface(cloud);
    shot.setInputNormals(normals);
    shot.compute(*descriptors);

    if (descriptors->empty())
    {
        std::cerr << "No SHOT descriptors were computed\n";
        return -1;
    }

    if (descriptors->size() != keypoints->size())
    {
        std::cerr << "SHOT descriptor count does not match keypoint count\n";
        return -1;
    }

    std::ofstream out(argv[3]);
    if (!out.is_open())
    {
        std::cerr << "Failed to open output file\n";
        return -1;
    }

    for (size_t i = 0; i < descriptors->size(); ++i)
    {
        for (int j = 0; j < 352; ++j)
        {
            const float value = descriptors->at(i).descriptor[j];
            out << (std::isfinite(value) ? value : NAN);
            if (j < 351)
            {
                out << ",";
            }
        }
        out << "\n";
    }

    std::cout << "Computed " << descriptors->size() << " SHOT descriptors\n";

    return 0;
}
