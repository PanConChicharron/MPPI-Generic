# add_executable(cartpole_example cartpole_example.cpp)
# target_include_directories(cartpole_example PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
# target_link_libraries(cartpole_example
#                       cnpy
#                       cartpole_mppi)

file(GLOB CUDA_TARGET_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cu)

find_package(OpenCV 4.5.4 REQUIRED)

foreach(T_FILE IN LISTS CUDA_TARGET_SRCS)
  # Get filename without extension to use as target name
  get_filename_component(T_NAME ${T_FILE} NAME_WE)
  add_executable(${T_NAME} ${T_FILE})
  target_include_directories(${T_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${T_NAME}
                        ${MPPI_HEADER_LIBRARY_NAME}
                        ${OpenCV_LIBS}
                        )
  # set_target_properties(${T_NAME} PROPERTIES FOLDER test)
endforeach()
