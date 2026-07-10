#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "royale::royale" for configuration "Release"
set_property(TARGET royale::royale APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(royale::royale PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/libroyale.so.5.16.1"
  IMPORTED_SONAME_RELEASE "libroyale.so.5.16.1"
  )

list(APPEND _cmake_import_check_targets royale::royale )
list(APPEND _cmake_import_check_files_for_royale::royale "${_IMPORT_PREFIX}/bin/libroyale.so.5.16.1" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
