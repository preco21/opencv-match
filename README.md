# opencv-match

> A Rust library that simplifies template matching with OpenCV

## Install

This library depends on the [OpenCV](https://opencv.org/) library.

You will need to follow the instruction to install OpenCV on your system. You can find the instructions [here](https://github.com/twistedfall/opencv-rust/blob/master/INSTALL.md).

```shell
cargo add opencv-match

# Alternatively, pull from the repository directly:
cargo add --git https://github.com/preco21/opencv-match.git
```

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
