# 定义函数
function(deploy)
    set(options opt1 opt2 opt3)
    set(oneValueArgs oneV1 oneV2 oneV3)
    set(multiValueArgs multV1 multV2)
    
    message(STATUS "ARGN: ${ARGN}")
    message(STATUS "options: ${options}")
    message(STATUS "oneValueArgs: ${oneValueArgs}")
    message(STATUS "multiValueArgs: ${multiValueArgs}")
    
    cmake_parse_arguments(Gen "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
 
    message(STATUS "Gen_opt1: ${Gen_opt1}")
    message(STATUS "Gen_opt2: ${Gen_opt2}")
    message(STATUS "Gen_opt3: ${Gen_opt3}")
    
    message(STATUS "Gen_oneV1: ${Gen_oneV1}")
    message(STATUS "Gen_oneV2: ${Gen_oneV2}")
    message(STATUS "Gen_oneV3: ${Gen_oneV3}")
 
    message(STATUS "Gen_multV1: ${Gen_multV1}")
    message(STATUS "Gen_multV2: ${Gen_multV2}")
    
    message(STATUS "Gen_KEYWORDS_MISSING_VALUES: ${Gen_KEYWORDS_MISSING_VALUES}")
    message(STATUS "Gen_UNPARSED_ARGUMENTS: ${Gen_UNPARSED_ARGUMENTS}")
    
    foreach (item ${Gen_multV2})
        message(STATUS "item: ${item}")
    endforeach()
endfunction()
 
# 调用形式一(标准)
message(STATUS "----begin----调用形式一")
deploy(opt1 opt2 opt3 oneV1 abc oneV2 def oneV3 xyz multV1 kaizen baidu git multV2 C++ Java Python)
message(STATUS "----end----调用形式一")
 
# 调用形式二(缺少opt1、opt2、oneV3)
message(STATUS "\n")
message(STATUS "----begin----调用形式二")
deploy(opt3 oneV1 abc oneV2 def multV1 kaizen baidu git multV2 C++ Java Python)
message(STATUS "----end----调用形式二")
 
# 调用形式三(缺少opt2、opt3; 多余 opt4、hig)
message(STATUS "\n")
message(STATUS "----begin----调用形式三")
deploy(opt1 opt2 opt4 oneV1 abc oneV2 def hig oneV3 multV1 kaizen baidu git multV2 C++)
message(STATUS "----end----调用形式三")
 
# 调用形式四(标准但顺序不同)
message(STATUS "\n")
message(STATUS "----begin----调用形式四")
deploy(multV1 kaizen baidu git oneV1 abc opt1 oneV2 baidu opt2 oneV3 beijing opt3 multV2 C++ Java)
message(STATUS "----end----调用形式四")
 
 
# 打印结束日志
message(STATUS "##########END_TEST")
