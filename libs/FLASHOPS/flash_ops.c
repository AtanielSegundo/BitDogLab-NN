// flash_ops.c
#include "flash_ops.h"
#include "flash_ops_helper.h"
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>

#include "pico/stdlib.h"
#include "hardware/flash.h"
#include "hardware/sync.h"

#define FLASH_TARGET_OFFSET (256 * 1024)
#define FLASH_SIZE          PICO_FLASH_SIZE_BYTES
#define METADATA_SIZE       sizeof(flash_data)

/**
 * Gravação segura.
 */
void flash_write_safe(uint32_t offset, const uint8_t *data, size_t data_len) {
    uint32_t flash_offset = FLASH_TARGET_OFFSET + offset;
    printf("flash_offset: %u\n", flash_offset);

    if (!data || data_len == 0) {
        printf("Error: No data or zero length.\n");
        return;
    }
    if (flash_offset % FLASH_SECTOR_SIZE) {
        printf("Error: Offset not sector-aligned.\n");
        return;
    }
    if (data_len > FLASH_SECTOR_SIZE - METADATA_SIZE) {
        printf("Error: Data too big for sector.\n");
        return;
    }
    if (flash_offset + METADATA_SIZE > FLASH_TARGET_OFFSET + FLASH_SIZE) {
        printf("Error: Beyond flash limits.\n");
        return;
    }

    uint32_t cnt = get_flash_write_count(offset) + 1;
    flash_data meta = {
        .valid       = true,
        .write_count = cnt,
        .data_len    = data_len,
        .data_ptr    = (uint8_t*)data
    };

    size_t raw_size = sizeof(flash_data) + data_len;
    const uint32_t page = FLASH_PAGE_SIZE; // 256
    size_t prog_size = (raw_size + page - 1) & ~(page - 1);

    uint8_t *buf = malloc(prog_size);
    if (!buf) {
        printf("Malloc failed.\n");
        return;
    }

    // serializa header + data
    serialize_flash_data(&meta, buf, prog_size);
    // padding
    if (prog_size > raw_size) {
        memset(buf + raw_size, 0xFF, prog_size - raw_size);
    }

    uint32_t ints = save_and_disable_interrupts();
    flash_range_erase(flash_offset, FLASH_SECTOR_SIZE);
    flash_range_program(flash_offset, buf, prog_size);
    restore_interrupts(ints);

    free(buf);
}


/**
 * Leitura segura (ajustada).
 */
void flash_read_safe(uint32_t offset, uint8_t *buffer, size_t buffer_len) {
    uint32_t flash_offset = FLASH_TARGET_OFFSET + offset;
    printf("Calculated flash offset: %u\n", flash_offset);

    if (flash_offset % FLASH_SECTOR_SIZE) {
        printf("Error: Offset not sector-aligned.\n");
        return;
    }
    if (flash_offset + METADATA_SIZE > FLASH_TARGET_OFFSET + FLASH_SIZE) {
        printf("Error: Beyond flash limits.\n");
        return;
    }

    // 1) lê o byte raw de valid
    uint8_t raw_valid = *(const uint8_t *)(XIP_BASE + flash_offset);
    if (raw_valid != 1) {
        // nunca gravado
        printf("Error: No valid data present.\n");
        return;
    }

    // 2) lê data_len (offset: 1 byte valid + 4 bytes write_count)
    size_t data_len;
    memcpy(&data_len,
           (const void *)(XIP_BASE + flash_offset + 1 + sizeof(uint32_t)),
           sizeof(size_t));

    if (data_len == 0 || buffer_len < data_len) {
        printf("Error: Buffer too small or zero-length data.\n");
        return;
    }

    // 3) copia direto da flash (imediatamente após header fields)
    size_t header_size = 1 + sizeof(uint32_t) + sizeof(size_t);
    memcpy(buffer,
           (const void *)(XIP_BASE + flash_offset + header_size),
           data_len);
}


/**
 * Erase seguro (sem mudanças).
 */
void flash_erase_safe(uint32_t offset) {
    uint32_t flash_offset = FLASH_TARGET_OFFSET + offset;
    if (flash_offset % FLASH_SECTOR_SIZE) {
        printf("Error: Offset not sector-aligned.\n");
        return;
    }
    if (flash_offset + METADATA_SIZE > FLASH_TARGET_OFFSET + FLASH_SIZE) {
        printf("Error: Beyond flash limits.\n");
        return;
    }

    uint32_t cnt = get_flash_write_count(offset) + 1;
    uint32_t ints = save_and_disable_interrupts();
    uint32_t sector = flash_offset & ~(FLASH_SECTOR_SIZE - 1);

    flash_range_erase(sector, FLASH_SECTOR_SIZE);

    flash_data m = {
        .valid       = false,
        .write_count = cnt,
        .data_len    = 0,
        .data_ptr    = NULL
    };
    flash_range_program(sector, (uint8_t*)&m, sizeof(flash_data));
    restore_interrupts(ints);
}
