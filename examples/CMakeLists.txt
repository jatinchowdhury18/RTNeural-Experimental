set(EXAMPLES_OUT_DIR ${PROJECT_BINARY_DIR}/examples_out)
file(MAKE_DIRECTORY ${EXAMPLES_OUT_DIR})

function(create_example example_name)
    message(STATUS "RTNeural-experimental -- Configuring example: ${example_name}")
    add_executable(${example_name} ${example_name}.cpp)
    target_link_libraries(${example_name}
        PUBLIC
            RTNeural
            RTNeural-experimental
    )
    target_compile_features(${example_name} PRIVATE cxx_std_17)
    target_include_directories(${example_name} PRIVATE ${CPM_RTNeural_SOURCE})
    target_compile_definitions(${example_name}
        PRIVATE
            RTNEURAL_SOURCE_DIR="${RTNeural_SOURCE_DIR}"
            RTNEURAL_EXPERIMENTS_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    add_custom_command(TARGET ${example_name}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "copying $<TARGET_FILE:${example_name}> to ${EXAMPLES_OUT_DIR}/${example_name}"
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${example_name}> ${EXAMPLES_OUT_DIR}/${example_name})
endfunction()

if(APPLE)
    add_subdirectory(accelerate)
endif()
add_subdirectory(residual_connection)
