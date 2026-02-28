#ifndef FLASH_OPS_H
#define FLASH_OPS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef struct {
    bool valid;
    uint32_t write_count;
    size_t data_len;
    uint8_t *data_ptr;
} flash_data;

void flash_write_safe(uint32_t offset, const uint8_t *data, size_t data_len);
void flash_read_safe(uint32_t offset, uint8_t *buffer, size_t buffer_len);
void flash_erase_safe(uint32_t offset);

#endif // FLASH_OPS_H
