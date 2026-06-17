#include <cstdlib>
#include <iostream>
#include <string>

#include <pcl/common/io.h>
#include <pcl/common/point_tests.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/point_types.h>

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

bool parsePositiveFloat(const char* value, float& output)
{
    char* end = nullptr;
    output = std::strtof(value, &end);
    return end != value && *end == '\0' && output > 0.0f;
}

bool parsePositiveInt(const char* value, int& output)
{
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0)
    {
        return false;
    }

    output = static_cast<int>(parsed);
    return true;
}
}

int main(int argc, char** argv)
{
    if (argc != 8)
    {
        std::cerr
            << "Usage: sift3d input.{ply,pcd} output.{ply,pcd} "
            << "normal_radius min_scale octaves scales_per_octave "
            << "min_contrast\n";
        return 2;
    }

    float normal_radius = 0.0f;
    float min_scale = 0.0f;
    float min_contrast = 0.0f;
    int octaves = 0;
    int scales_per_octave = 0;

    if (!parsePositiveFloat(argv[3], normal_radius) ||
        !parsePositiveFloat(argv[4], min_scale) ||
        !parsePositiveInt(argv[5], octaves) ||
        !parsePositiveInt(argv[6], scales_per_octave) ||
        !parsePositiveFloat(argv[7], min_contrast))
    {
        std::cerr << "All SIFT3D parameters must be positive numbers\n";
        return 2;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    if (loadPointCloud(argv[1], *cloud) == -1)
    {
        std::cerr << "Failed to load input point cloud: " << argv[1] << '\n';
        return 1;
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
        return 1;
    }

    pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> normal_estimation;
    pcl::PointCloud<pcl::PointNormal>::Ptr point_normals(
        new pcl::PointCloud<pcl::PointNormal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr normal_tree(
        new pcl::search::KdTree<pcl::PointXYZ>);

    normal_estimation.setInputCloud(cloud);
    normal_estimation.setSearchMethod(normal_tree);
    normal_estimation.setRadiusSearch(normal_radius);
    normal_estimation.compute(*point_normals);

    for (std::size_t index = 0; index < point_normals->size(); ++index)
    {
        (*point_normals)[index].x = (*cloud)[index].x;
        (*point_normals)[index].y = (*cloud)[index].y;
        (*point_normals)[index].z = (*cloud)[index].z;
    }

    pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
    pcl::search::KdTree<pcl::PointNormal>::Ptr sift_tree(
        new pcl::search::KdTree<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointWithScale> keypoints_with_scale;

    sift.setSearchMethod(sift_tree);
    sift.setScales(min_scale, octaves, scales_per_octave);
    sift.setMinimumContrast(min_contrast);
    sift.setInputCloud(point_normals);
    sift.compute(keypoints_with_scale);

    pcl::PointCloud<pcl::PointXYZ> keypoints;
    pcl::copyPointCloud(keypoints_with_scale, keypoints);

    if (keypoints.empty())
    {
        std::cerr << "SIFT3D detected no keypoints\n";
        return 1;
    }

    if (savePointCloud(argv[2], keypoints) == -1)
    {
        std::cerr << "Failed to save SIFT3D keypoints: " << argv[2] << '\n';
        return 1;
    }

    std::cout << "Detected " << keypoints.size() << " SIFT3D keypoints\n";
    return 0;
}
