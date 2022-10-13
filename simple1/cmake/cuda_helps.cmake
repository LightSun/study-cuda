

function(cuda_add_demo)
        set(options OPT opt)
        set(oneValueArgs ONE exe_name)
        set(multiValueArgs MULTI exe_files)
        cmake_parse_arguments(Gen "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
        cuda_add_executable(${Gen_exe_name}
            ${Gen_exe_files}
        )
        target_link_libraries(${Gen_exe_name} ${CUDA_LIBRARIES})
endfunction()

