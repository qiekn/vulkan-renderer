cmake_minimum_required(VERSION 3.30)

# import std: must be set before project() for toolchain detection
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")

include(cmake/EnableCxxImportStd.cmake)
project(vulkan_engine LANGUAGES CXX)

set(CMAKE_CXX_MODULE_STD ON)

# compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -----------------------------------------------------------------------------: Config
set(ASSETS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assets)
set(ASSETS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/assets)
set(SHADERS_BINARY_DIR ${ASSETS_BINARY_DIR}/shaders)

# -----------------------------------------------------------------------------: Dependencies
find_package(Vulkan REQUIRED)

add_subdirectory(${CMAKE_SOURCE_DIR}/deps/glfw)
add_subdirectory(${CMAKE_SOURCE_DIR}/deps/glm)

# -----------------------------------------------------------------------------: Vulkan C++ Module
add_library(VulkanCppModule)
add_library(Vulkan::cppm ALIAS VulkanCppModule)

target_compile_definitions(VulkanCppModule PUBLIC
  VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
  VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
  VULKAN_HPP_CXX_MODULE_EXPERIMENTAL_WARNING=1
)

target_include_directories(VulkanCppModule PRIVATE "${Vulkan_INCLUDE_DIRS}")
target_link_libraries(VulkanCppModule PUBLIC Vulkan::Vulkan)

target_sources(VulkanCppModule
  PUBLIC
  FILE_SET cxx_modules TYPE CXX_MODULES
  BASE_DIRS "${Vulkan_INCLUDE_DIRS}"
  FILES "${Vulkan_INCLUDE_DIRS}/vulkan/vulkan.cppm"
        "${Vulkan_INCLUDE_DIRS}/vulkan/vulkan_video.cppm"
)

# -----------------------------------------------------------------------------: Slang
find_program(SLANGC_EXECUTABLE
  NAMES slangc
  HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
  DOC "Path to slangc shader compiler"
)
if (NOT SLANGC_EXECUTABLE)
  message(FATAL_ERROR "slangc not found. Set VULKAN_SDK or add slangc to PATH.")
endif()
message(STATUS "Found slangc: ${SLANGC_EXECUTABLE}")

function (add_slang_shader_target TARGET)
  cmake_parse_arguments ("SHADER" "" "" "SOURCES" ${ARGN})

  if (NOT SHADER_SOURCES)
    message(FATAL_ERROR "add_slang_shader_target(${TARGET}) requires SOURCES")
  endif()

  set (ENTRY_POINTS -entry vertMain -entry fragMain)
  set (SLANG_OUTPUT ${SHADERS_BINARY_DIR}/slang.spv)

  add_custom_command (
          OUTPUT ${SLANG_OUTPUT}
          COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_BINARY_DIR}
          COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o ${SLANG_OUTPUT}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          DEPENDS ${SHADER_SOURCES}
          COMMENT "Compiling Slang Shaders"
          VERBATIM
  )
  add_custom_target (${TARGET} DEPENDS ${SLANG_OUTPUT})
endfunction()

# -----------------------------------------------------------------------------: Targets
file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB_RECURSE MODULE_FILES CONFIGURE_DEPENDS "src/*.cppm")
file(GLOB_RECURSE SHADER_SLANG_SOURCES CONFIGURE_DEPENDS "assets/shaders/*.slang")

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE ${SOURCE_FILES})
target_sources(${PROJECT_NAME} PRIVATE FILE_SET CXX_MODULES FILES ${MODULE_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE
  ./deps/head-only/
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  Vulkan::cppm
  glfw
  glm
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
  GLM_FORCE_DEPTH_ZERO_TO_ONE  # Vulkan clip-space depth is [0, 1], not OpenGL's [-1, 1]
  GLM_FORCE_RADIANS
)

add_custom_target(sync_assets ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${ASSETS_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${ASSETS_SOURCE_DIR} ${ASSETS_BINARY_DIR}
  COMMENT "Syncing runtime assets to build/assets"
)

if (SHADER_SLANG_SOURCES)
  add_slang_shader_target(shader SOURCES ${SHADER_SLANG_SOURCES})
  add_dependencies(${PROJECT_NAME} shader)
endif()

add_dependencies(${PROJECT_NAME} sync_assets)
