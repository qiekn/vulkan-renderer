# Vulkan Engine

跟随 Khronos 官方 [Building a Simple Engine](https://docs.vulkan.org/tutorial/latest/Building_a_Simple_Engine/introduction.html) 教程实现一个可复用的 Vulkan 渲染引擎。

前置学习仓库：[qiekn/vulkan](https://github.com/qiekn/vulkan)（Khronos Vulkan Tutorial 基础章节笔记）。

## 开发环境

- **平台**: Windows (MSYS2 UCRT64)
- **编译器**: Clang 21 (C++23 Modules & LLVM libc++)
- **构建**: CMake 3.30+ / Ninja
- **Vulkan SDK**: 1.4.341.1
- **渲染风格**: `vk::raii` + Dynamic Rendering + C++23 Modules

## 三方库依赖

| 第三方库 (Git Submodule)                                         | 用途       |
| ---                                                              | ---        |
| [GLFW](https://github.com/glfw/glfw)                             | 窗口管理   |
| [GLM](https://github.com/g-truc/glm)                             | 数学库     |

后续章节会按需新增 tinyobjloader / stb / Dear ImGui 等。

## 构建运行

```bash
git clone --recursive <this-repo-url>
```

下载安装 [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)，确保 `VULKAN_SDK` 环境变量已设置；把 VulkanSDK 的 `Bin` 目录加入 `PATH`（CMake 自定义命令需要调用 `slangc.exe`）：

```bash
export PATH=$PATH:"/c/VulkanSDK/1.4.341.1/Bin"
```

然后：

```bash
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++
cmake --build build
cd build
./vulkan_engine
```

或直接运行 `./run.sh`（调试模式：`./run.sh debug`）。

## 教程章节进度

- [ ] Engine Architecture
- [ ] Camera Transformations
- [ ] Lighting & Materials
- [ ] GUI (Dear ImGui)
- [ ] Loading Models
- [ ] Subsystems
- [ ] Tooling
- [ ] Mobile Development
- [ ] Advanced Topics
