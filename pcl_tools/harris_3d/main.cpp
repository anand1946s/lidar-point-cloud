#include <string>
#include <iostream>

#include <pcl/common/point_tests.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

#include <pcl/keypoints/harris_3d.h>

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

int savePointCloud(
    const std::string& path,
    const pcl::PointCloud<pcl::PointXYZ>& cloud)
{
    if (hasSuffix(path, ".pcd"))
    {
        return pcl::io::savePCDFileBinary(path, cloud);
    }

    return pcl::io::savePLYFileBinary(path, cloud);
}
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage:\n";
        std::cerr << "./harris3d input.{ply,pcd} output.{ply,pcd}\n";
        return -1;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    if (loadPointCloud(argv[1], *cloud) == -1)
    {
        std::cerr << "Failed to load input file\n";
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

    if (cloud->empty())
    {
        std::cerr << "Input cloud contains no finite points\n";
        return -1;
    }

    pcl::HarrisKeypoint3D<
        pcl::PointXYZ,
        pcl::PointXYZI
    > detector;

    detector.setInputCloud(cloud);

    detector.setRadius(0.01);

    pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints(
        new pcl::PointCloud<pcl::PointXYZI>);

    detector.compute(*keypoints);

    std::cout
        << "Detected "
        << keypoints->size()
        << " keypoints\n";

    pcl::PointCloud<pcl::PointXYZ>::Ptr output(
        new pcl::PointCloud<pcl::PointXYZ>);

    for (const auto& p : keypoints->points)
    {
        output->push_back(
            pcl::PointXYZ(
                p.x,
                p.y,
                p.z
            )
        );
    }

    if (output->empty())
    {
        std::cerr << "Harris3D detected no keypoints\n";
        return -1;
    }

    if (savePointCloud(argv[2], *output) == -1)
    {
        std::cerr << "Failed to save Harris3D keypoints\n";
        return -1;
    }

    return 0;
}
