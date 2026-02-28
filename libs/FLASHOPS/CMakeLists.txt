add_library(FLASHOPS
	flash_ops.c
	flash_ops_helper.c
	custom_fgets.c
)

target_include_directories(FLASHOPS PUBLIC
	.
)

target_link_libraries(FLASHOPS pico_stdlib hardware_flash hardware_sync)