# Set the name of the project and target:
SET(TARGET "main")
SET(CMAKE_CXX_FLAGS "-pthread")


########################################
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.3 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()


DEAL_II_INITIALIZE_CACHED_VARIABLES()

########################################
FILE(GLOB TARGET_SRC "src/*.cc")
#FILE(GLOB TARGET_INC "include/*h")
#SET(TARGET_SRC ${TARGET_SRC} ${TARGET_INC})
INCLUDE_DIRECTORIES("include/")

PROJECT(DE_IMPLICIT)
#ADD_EXECUTABLE(${TARGET} ${TARGET_SRC})
#DEAL_II_SETUP_TARGET(${TARGET})



########################################
#ADD_CUSTOM_TARGET(debug
  #COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  #COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  #COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  #)

#ADD_CUSTOM_TARGET(release
  #COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  #COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  #COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  #)

DEAL_II_INVOKE_AUTOPILOT()
