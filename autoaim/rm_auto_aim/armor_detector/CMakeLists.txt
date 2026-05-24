cmake_minimum_required(VERSION 3.10)
project(armor_detector)

## Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
## enforcing cleaner code.
add_definitions(-Wall -Werror)

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#######################
## Find dependencies ##
#######################

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP)
find_package(hik_camera REQUIRED)
ament_auto_find_build_dependencies()

find_path(HOBOT_CV_INCLUDE_DIR
  NAMES hobot_cv/hobotcv_imgproc.h
  PATHS /opt/tros/humble/include /usr/include /usr/local/include)
find_library(HOBOT_CV_LIBRARY
  NAMES hobot_cv
  PATHS /opt/tros/humble/lib /usr/lib /usr/local/lib)

###########
## Build ##
###########

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/detector.cpp
  src/detector_node.cpp
  src/pnp_solver.cpp
  src/yolo.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

if(HOBOT_CV_INCLUDE_DIR AND HOBOT_CV_LIBRARY)
  target_include_directories(${PROJECT_NAME} PRIVATE ${HOBOT_CV_INCLUDE_DIR})
  target_link_libraries(${PROJECT_NAME} ${HOBOT_CV_LIBRARY})
  target_compile_definitions(${PROJECT_NAME} PRIVATE ARMOR_DETECTOR_USE_HOBOT_CV=1)
  message(STATUS "armor_detector: hobot_cv enabled")
else()
  message(STATUS "armor_detector: hobot_cv not found, fallback to OpenCV preprocess")
endif()

if(OpenMP_CXX_FOUND)
  target_compile_options(${PROJECT_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_libraries(${PROJECT_NAME} ${OpenMP_CXX_LIBRARIES})
endif()

# RDK libdnn headers/libraries
target_include_directories(${PROJECT_NAME} PRIVATE /usr/include/dnn)
target_link_libraries(${PROJECT_NAME} dnn pthread rt dl)
target_link_options(${PROJECT_NAME} PRIVATE "-Wl,-unresolved-symbols=ignore-in-shared-libs")

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN rm_auto_aim::ArmorDetectorNode
  EXECUTABLE armor_detector_node
)

#############
## Testing ##
#############

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_copyright
    ament_cmake_uncrustify
    ament_cmake_cpplint
  )
  ament_lint_auto_find_test_dependencies()

  # find_package(ament_cmake_gtest)
  # ament_add_gtest(test_node_startup test/test_node_startup.cpp)
  # target_link_libraries(test_node_startup ${PROJECT_NAME})

  # ament_add_gtest(test_number_cls test/test_number_cls.cpp)
  # target_link_libraries(test_number_cls ${PROJECT_NAME})

endif()

#############
## Install ##
#############

ament_auto_package(
  # YOLO model is provided externally (e.g. autoaim/model) via parameters.
)
