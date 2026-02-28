#include "flash_ops_helper.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>

#include "hardware/flash.h"
#include "hardware/sync.h"
#include "pico/stdlib.h"

#define FLASH_TARGET_OFFSET (256 * 1024)
#define FLASH_SIZE          PICO_FLASH_SIZE_BYTES
#define METADATA_SIZE       sizeof(flash_data)

/**
 * Retorna write_count do header.
 */
uint32_t get_flash_write_count(uint32_t offset) {
    uint32_t foff = FLASH_TARGET_OFFSET + offset;
    if (foff % FLASH_SECTOR_SIZE) return 0;
    if (foff + METADATA_SIZE > FLASH_TARGET_OFFSET + FLASH_SIZE) return 0;
    flash_data tmp;
    memcpy(&tmp, (const void*)(XIP_BASE + foff), METADATA_SIZE);
    return tmp.write_count;
}

/**
 * Serializa header + data_ptr.
 */
void serialize_flash_data(const flash_data *d, uint8_t *buf, size_t buf_size) {
    size_t req = sizeof(d->valid) + sizeof(d->write_count)
               + sizeof(d->data_len) + d->data_len;
    if (buf_size < req) {
        printf("Buffer too small (%zu < %zu)\n", buf_size, req);
        return;
    }
    memcpy(buf, &d->valid, sizeof(d->valid));
    buf += sizeof(d->valid);
    memcpy(buf, &d->write_count, sizeof(d->write_count));
    buf += sizeof(d->write_count);
    memcpy(buf, &d->data_len, sizeof(d->data_len));
    buf += sizeof(d->data_len);
    if (d->data_ptr && d->data_len)
        memcpy(buf, d->data_ptr, d->data_len);
}
