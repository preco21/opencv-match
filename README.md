# opencv-match

> A Rust library that simplifies template matching with OpenCV

## Prerequisites

This library is built on top of the [OpenCV](https://opencv.org/) library. You will need to have OpenCV installed on your system.

You can find the instructions [here](https://github.com/twistedfall/opencv-rust/blob/master/INSTALL.md).

Currently supports `v0.92.2` of the `opencv` crate.

[TODO: Make version selection more flexible]

## Install

```shell
cargo add opencv-match

# Alternatively, pull from the repository directly:
cargo add --git https://github.com/preco21/opencv-match.git
```

## Features

- **Multiple template matching:** Detects and matches multiple instances of a template within a single image.
- **Non-maximum suppression:** Integrates [non-maximum suppression](https://builtin.com/machine-learning/non-maximum-suppression) to eliminate duplicate matches.
- **Directional clustering:** Clusters matching points vertically or horizontally into distinct sections, organizing them into bounding boxes for segmentation.

[TODO: Scale-invariant matching, Angle-invariant matching]

## Usage

[WIP]

## FAQ

### How can I link OpenCV statically in Windows?

Assuming you are using [`vcpkg`](https://vcpkg.io/), you install the static version of OpenCV with the following commands:

```powershell
vcpkg add zlib:x64-windows-static opencv4[contrib,nonfree]:x64-windows-static
vcpkg install zlib:x64-windows-static opencv4[contrib,nonfree]:x64-windows-static
```

This will take a while to build. After that, you can link the static libraries in your `Cargo.toml`:

```toml
[env]
OPENCV_LINK_LIBS = "zlib,opencv_core4,opencv_imgproc4"
OPENCV_INCLUDE_PATHS = "C:\\[PATH_TO_VCPKG]\\installed\\x64-windows-static\\include"
OPENCV_LINK_PATHS = "C:\\[PATH_TO_VCPKG]\\installed\\x64-windows-static\\lib"
OPENCV_MSVC_CRT = "static"

[target.x86_64-pc-windows-msvc]
rustflags = ["-C", "target-feature=+crt-static"]
```

[WIP: Do we need OPENCV_LINK_LIBS explicitly specified?]
