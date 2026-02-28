#ifndef FLASH_OPS_HELPER_H
#define FLASH_OPS_HELPER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "flash_ops.h"

uint32_t get_flash_write_count(uint32_t offset);
uint32_t get_flash_data_length(uint32_t offset);
void serialize_flash_data(const flash_data *data, uint8_t *buffer, size_t buffer_size);

#endif // FLASH_OPS_HELPER_H
