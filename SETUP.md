# LiDAR Point Cloud Registration Setup

This guide explains how to clone, install, build, and run the point cloud
registration project on Linux and Windows.

## 1. What Needs To Be Built

Most of the Python code uses Open3D and installs through `requirements.txt`.
However, some algorithm choices call local C++ executables:

| Setting | Needs extra build? | Executable expected by Python |
| --- | --- | --- |
| `FEATURE_METHOD = "NONE"` | No | None |
| `FEATURE_METHOD = "ISS"` | No | None |
| `FEATURE_METHOD = "FPFH"` | No | None |
| `FEATURE_METHOD = "SHOT"` | Yes, PCL | `pcl_tools/shot/build/shot` |
| `FEATURE_METHOD = "HARRIS3D"` | Yes, PCL | `pcl_tools/harris_3d/build/harris3d` |
| `FEATURE_METHOD = "3DSIFT"` | Yes, PCL | `source_codes/sift3d/build/sift3d` |
| `REGISTRATION_METHOD = "ICP"` | No | None |
| `REGISTRATION_METHOD = "ROBUST_ICP"` | No | None |
| `REGISTRATION_METHOD = "FGR"` | No | None |
| `REGISTRATION_METHOD = "GO_ICP"` | Yes, CMake/C++ | `source_codes/Go-ICP-master/Go-ICP-master/GoICP` |

The default config currently uses `REGISTRATION_METHOD = "GO_ICP"`, so a new
user must either build Go-ICP or change the method to `ICP`, `ROBUST_ICP`, or
`FGR` in `task1/config.py`.

## 2. Clone The Repository

```bash
git clone <repository_url>
cd lidar-point-cloud
```

## 3. Python Setup

### Linux

```bash
python3 -m venv myenv
source myenv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -m venv myenv
.\myenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\myenv\Scripts\Activate.ps1
```

## 4. Linux C++/PCL Setup

Install the native build dependencies:

```bash
sudo apt update
sudo apt install build-essential cmake libpcl-dev
```

Build the PCL-based feature tools:

```bash
cmake -S pcl_tools/shot -B pcl_tools/shot/build
cmake --build pcl_tools/shot/build

cmake -S pcl_tools/harris_3d -B pcl_tools/harris_3d/build
cmake --build pcl_tools/harris_3d/build

cmake -S source_codes/sift3d -B source_codes/sift3d/build
cmake --build source_codes/sift3d/build
```

Build Go-ICP:

```bash
cmake -S source_codes/Go-ICP-master/Go-ICP-master -B source_codes/Go-ICP-master/Go-ICP-master/build
cmake --build source_codes/Go-ICP-master/Go-ICP-master/build
cp source_codes/Go-ICP-master/Go-ICP-master/build/GoICP source_codes/Go-ICP-master/Go-ICP-master/GoICP
```

## 5. Windows C++/PCL Setup

Install these first:

- Visual Studio 2022 Build Tools with the "Desktop development with C++"
  workload.
- CMake.
- Git.
- vcpkg, for PCL.

Install PCL through vcpkg:

```powershell
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:\vcpkg\vcpkg install pcl:x64-windows
```

Set the vcpkg toolchain path for the current PowerShell session:

```powershell
$env:VCPKG_ROOT = "C:\vcpkg"
$toolchain = "$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake"
```

Build the PCL-based feature tools:

```powershell
cmake -S pcl_tools/shot -B pcl_tools/shot/build -A x64 -DCMAKE_TOOLCHAIN_FILE="$toolchain"
cmake --build pcl_tools/shot/build --config Release

cmake -S pcl_tools/harris_3d -B pcl_tools/harris_3d/build -A x64 -DCMAKE_TOOLCHAIN_FILE="$toolchain"
cmake --build pcl_tools/harris_3d/build --config Release

cmake -S source_codes/sift3d -B source_codes/sift3d/build -A x64 -DCMAKE_TOOLCHAIN_FILE="$toolchain"
cmake --build source_codes/sift3d/build --config Release
```

The Python code expects the Windows executables directly inside each `build`
folder. If CMake places them under `build/Release`, copy them once:

```powershell
Copy-Item pcl_tools/shot/build/Release/shot.exe pcl_tools/shot/build/shot.exe
Copy-Item pcl_tools/harris_3d/build/Release/harris3d.exe pcl_tools/harris_3d/build/harris3d.exe
Copy-Item source_codes/sift3d/build/Release/sift3d.exe source_codes/sift3d/build/sift3d.exe
```

Build Go-ICP:

```powershell
cmake -S source_codes/Go-ICP-master/Go-ICP-master -B source_codes/Go-ICP-master/Go-ICP-master/build -A x64
cmake --build source_codes/Go-ICP-master/Go-ICP-master/build --config Release
Copy-Item source_codes/Go-ICP-master/Go-ICP-master/build/Release/GoICP.exe source_codes/Go-ICP-master/Go-ICP-master/GoICP.exe
```

## 6. Configure The Pipeline

Edit `task1/config.py` before running the project.

Important settings:

- `DATA_DIR`: folder containing the point clouds.
- `OUTPUT_DIR`: folder where results are written.
- `SOURCE_NAME` and `TARGET_NAME`: point cloud names for `main.py`.
- `FEATURE_METHOD`: one of `NONE`, `ISS`, `3DSIFT`, `HARRIS3D`, `FPFH`, or
  `SHOT`.
- `REGISTRATION_METHOD`: one of `ICP`, `ROBUST_ICP`, `FGR`, or `GO_ICP`.
- `VISUALIZE`: set to `False` for batch runs or headless machines.
- `SAVE_RESULTS`: set to `True` to save metrics and aligned point clouds.

For the simplest first test, use:

```python
FEATURE_METHOD = "NONE"
REGISTRATION_METHOD = "ICP"
VISUALIZE = False
```

This path does not require PCL or Go-ICP.

## 7. Run The Project

Run a single pair:

```bash
python task1/main.py
```

Run the batch evaluation:

```bash
python task1/batch.py
```

Keep `VISUALIZE = False` when running `batch.py`; otherwise the script can
pause while waiting for Open3D viewer windows to close.

## 8. Troubleshooting

### `SHOT executable not found`, `Harris3D executable not found`, or `SIFT3D executable not found`

The selected `FEATURE_METHOD` needs a C++ tool that has not been built yet.
Build the matching tool from the Linux or Windows section above, or choose
`NONE`, `ISS`, or `FPFH`.

### `GoICP executable not found`

`REGISTRATION_METHOD = "GO_ICP"` needs the Go-ICP executable. Build Go-ICP, or
choose `ICP`, `ROBUST_ICP`, or `FGR`.

### CMake cannot find PCL

On Linux, confirm `libpcl-dev` is installed.

On Windows, confirm PCL was installed with vcpkg and that the CMake command
includes:

```powershell
-DCMAKE_TOOLCHAIN_FILE="$toolchain"
```

### CMake uses old paths from another machine

Delete the affected `build` folder and configure again:

```bash
rm -rf pcl_tools/shot/build
cmake -S pcl_tools/shot -B pcl_tools/shot/build
```

On Windows PowerShell:

```powershell
Remove-Item -Recurse -Force pcl_tools/shot/build
cmake -S pcl_tools/shot -B pcl_tools/shot/build -A x64 -DCMAKE_TOOLCHAIN_FILE="$toolchain"
```

Build folders and CMake cache files are machine-specific and should not be
treated as portable source files.
