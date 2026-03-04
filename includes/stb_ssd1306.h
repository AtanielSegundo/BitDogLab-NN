#ifndef STB_SSD1306_H
    #define STB_SSD1306_H

    #include <stdlib.h>
    #include "pico/stdlib.h"
    #include "hardware/i2c.h"

    #define ssd1306_height 64 // Define a altura do display (32 pixels)
    #define ssd1306_width 128 // Define a largura do display (128 pixels)

    #define ssd1306_i2c_address _u(0x3C) // Define o endereço do i2c do display

    #define ssd1306_i2c_clock 400 // Define o tempo do clock (pode ser aumentado)

    // Comandos de configuração (endereços)
    #define ssd1306_set_memory_mode _u(0x20)
    #define ssd1306_set_column_address _u(0x21)
    #define ssd1306_set_page_address _u(0x22)
    #define ssd1306_set_horizontal_scroll _u(0x26)
    #define ssd1306_set_scroll _u(0x2E)

    #define ssd1306_set_display_start_line _u(0x40)

    #define ssd1306_set_contrast _u(0x81)
    #define ssd1306_set_charge_pump _u(0x8D)

    #define ssd1306_set_segment_remap _u(0xA0)
    #define ssd1306_set_entire_on _u(0xA4)
    #define ssd1306_set_all_on _u(0xA5)
    #define ssd1306_set_normal_display _u(0xA6)
    #define ssd1306_set_inverse_display _u(0xA7)
    #define ssd1306_set_mux_ratio _u(0xA8)
    #define ssd1306_set_display _u(0xAE)
    #define ssd1306_set_common_output_direction _u(0xC0)
    #define ssd1306_set_common_output_direction_flip _u(0xC0)

    #define ssd1306_set_display_offset _u(0xD3)
    #define ssd1306_set_display_clock_divide_ratio _u(0xD5)
    #define ssd1306_set_precharge _u(0xD9)
    #define ssd1306_set_common_pin_configuration _u(0xDA)
    #define ssd1306_set_vcomh_deselect_level _u(0xDB)

    #define ssd1306_page_height _u(8)
    #define ssd1306_n_pages (ssd1306_height / ssd1306_page_height)
    #define ssd1306_buffer_length (ssd1306_n_pages * ssd1306_width)

    #define ssd1306_write_mode _u(0xFE)
    #define ssd1306_read_mode _u(0xFF)

    static uint8_t font[] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Nothing
        0x78, 0x14, 0x12, 0x11, 0x12, 0x14, 0x78, 0x00, // A
        0x7f, 0x49, 0x49, 0x49, 0x49, 0x49, 0x7f, 0x00, // B
        0x7e, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x00, // C
        0x7f, 0x41, 0x41, 0x41, 0x41, 0x41, 0x7e, 0x00, // D
        0x7f, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x00, // E
        0x7f, 0x09, 0x09, 0x09, 0x09, 0x01, 0x01, 0x00, // F
        0x7f, 0x41, 0x41, 0x41, 0x51, 0x51, 0x73, 0x00, // G
        0x7f, 0x08, 0x08, 0x08, 0x08, 0x08, 0x7f, 0x00, // H
        0x00, 0x00, 0x00, 0x7f, 0x00, 0x00, 0x00, 0x00, // I
        0x21, 0x41, 0x41, 0x3f, 0x01, 0x01, 0x01, 0x00, // J
        0x00, 0x7f, 0x08, 0x08, 0x14, 0x22, 0x41, 0x00, // K
        0x7f, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x00, // L
        0x7f, 0x02, 0x04, 0x08, 0x04, 0x02, 0x7f, 0x00, // M
        0x7f, 0x02, 0x04, 0x08, 0x10, 0x20, 0x7f, 0x00, // N
        0x3e, 0x41, 0x41, 0x41, 0x41, 0x41, 0x3e, 0x00, // O
        0x7f, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e, 0x00, // P
        0x3e, 0x41, 0x41, 0x49, 0x51, 0x61, 0x7e, 0x00, // Q
        0x7f, 0x11, 0x11, 0x11, 0x31, 0x51, 0x0e, 0x00, // R
        0x46, 0x49, 0x49, 0x49, 0x49, 0x30, 0x00, 0x00, // S
        0x01, 0x01, 0x01, 0x7f, 0x01, 0x01, 0x01, 0x00, // T
        0x3f, 0x40, 0x40, 0x40, 0x40, 0x40, 0x3f, 0x00, // U
        0x0f, 0x10, 0x20, 0x40, 0x20, 0x10, 0x0f, 0x00, // V
        0x7f, 0x20, 0x10, 0x08, 0x10, 0x20, 0x7f, 0x00, // W
        0x00, 0x41, 0x22, 0x14, 0x14, 0x22, 0x41, 0x00, // X
        0x01, 0x02, 0x04, 0x78, 0x04, 0x02, 0x01, 0x00, // Y
        0x41, 0x61, 0x59, 0x45, 0x43, 0x41, 0x00, 0x00, // Z
        0x3e, 0x41, 0x41, 0x49, 0x41, 0x41, 0x3e, 0x00, // 0
        0x00, 0x00, 0x42, 0x7f, 0x40, 0x00, 0x00, 0x00, // 1
        0x30, 0x49, 0x49, 0x49, 0x49, 0x46, 0x00, 0x00, // 2
        0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x36, 0x00, // 3
        0x3f, 0x20, 0x20, 0x78, 0x20, 0x20, 0x00, 0x00, // 4
        0x4f, 0x49, 0x49, 0x49, 0x49, 0x30, 0x00, 0x00, // 5
        0x3f, 0x48, 0x48, 0x48, 0x48, 0x48, 0x30, 0x00, // 6
        0x01, 0x01, 0x01, 0x61, 0x31, 0x0d, 0x03, 0x00, // 7
        0x36, 0x49, 0x49, 0x49, 0x49, 0x49, 0x36, 0x00, // 8
        0x06, 0x09, 0x09, 0x09, 0x09, 0x09, 0x7f, 0x00, // 9
        
        // NOVOS CARACTERES ADICIONADOS:
        0x00, 0x60, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, // . (Ponto) - Índice 37
        0x26, 0x16, 0x08, 0x64, 0x32, 0x00, 0x00, 0x00, // % (Porcentagem) - Índice 38
        0x08, 0x08, 0x08, 0x08, 0x08, 0x00, 0x00, 0x00, // - (Hífen/Menos) - Índice 39
        0x00, 0x66, 0x66, 0x00, 0x00, 0x00, 0x00, 0x00, // : (Dois pontos) - Índice 40
    };
    
    struct render_area {
        uint8_t start_column;
        uint8_t end_column;
        uint8_t start_page;
        uint8_t end_page;
        int buffer_length;
    };

    typedef struct {
        uint8_t width, height, pages, address;
        i2c_inst_t * i2c_port;
        bool external_vcc;
        uint8_t *ram_buffer;
        size_t bufsize;
        uint8_t port_buffer[2];
    } ssd1306_t;


    void calculate_render_area_buffer_length(struct render_area *area);
    void ssd1306_send_command(uint8_t cmd);
    void ssd1306_send_command_list(uint8_t *ssd, int number);
    void ssd1306_send_buffer(uint8_t ssd[], int buffer_length);
    void ssd1306_init();
    void ssd1306_scroll(bool set);
    void render_on_display(uint8_t *ssd, struct render_area *area);
    void ssd1306_set_pixel(uint8_t *ssd, int x, int y, bool set);
    void ssd1306_draw_line(uint8_t *ssd, int x_0, int y_0, int x_1, int y_1, bool set);
    void ssd1306_draw_char(uint8_t *ssd, int16_t x, int16_t y, uint8_t character);
    void ssd1306_draw_string(uint8_t *ssd, int16_t x, int16_t y, char *string);
    void ssd1306_command(ssd1306_t *ssd, uint8_t command);
    void ssd1306_config(ssd1306_t *ssd);
    void ssd1306_init_bm(ssd1306_t *ssd, uint8_t width, uint8_t height, bool external_vcc, uint8_t address, i2c_inst_t *i2c);
    void ssd1306_send_data(ssd1306_t *ssd);
    void ssd1306_draw_bitmap(ssd1306_t *ssd, const uint8_t *bitmap);

    #ifdef SSD1306_IMPLEMENTATION
        #include <stdio.h>
        #include <string.h>
        #include <stdlib.h>
        #include <ctype.h>
        #include "pico/stdlib.h"
        #include "pico/binary_info.h"
        #include "hardware/i2c.h"
        
        // Calcular quanto do buffer será destinado à área de renderização
        void calculate_render_area_buffer_length(struct render_area *area) {
            area->buffer_length = (area->end_column - area->start_column + 1) * (area->end_page - area->start_page + 1);
        }

        // Processo de escrita do i2c espera um byte de controle, seguido por dados
        void ssd1306_send_command(uint8_t command) {
            uint8_t buffer[2] = {0x80, command};
            i2c_write_blocking(i2c1, ssd1306_i2c_address, buffer, 2, false);
        }

        // Envia uma lista de comandos ao hardware
        void ssd1306_send_command_list(uint8_t *ssd, int number) {
            for (int i = 0; i < number; i++) {
                ssd1306_send_command(ssd[i]);
            }
        }

        // Copia buffer de referência num novo buffer, a fim de adicionar o byte de controle desde o início
        void ssd1306_send_buffer(uint8_t ssd[], int buffer_length) {
            uint8_t *temp_buffer = malloc(buffer_length + 1);

            temp_buffer[0] = 0x40;
            memcpy(temp_buffer + 1, ssd, buffer_length);

            i2c_write_blocking(i2c1, ssd1306_i2c_address, temp_buffer, buffer_length + 1, false);

            free(temp_buffer);
        }

        // Cria a lista de comandos (com base nos endereços definidos em ssd1306_i2c.h) para a inicialização do display
        void ssd1306_init() {
            uint8_t commands[] = {
                ssd1306_set_display, ssd1306_set_memory_mode, 0x00,
                ssd1306_set_display_start_line, ssd1306_set_segment_remap | 0x01, 
                ssd1306_set_mux_ratio, ssd1306_height - 1,
                ssd1306_set_common_output_direction | 0x08, ssd1306_set_display_offset,
                0x00, ssd1306_set_common_pin_configuration,
            
        #if ((ssd1306_width == 128) && (ssd1306_height == 32))
            0x02,
        #elif ((ssd1306_width == 128) && (ssd1306_height == 64))
            0x12,
        #else
            0x02,
        #endif
                ssd1306_set_display_clock_divide_ratio, 0x80, ssd1306_set_precharge,
                0xF1, ssd1306_set_vcomh_deselect_level, 0x30, ssd1306_set_contrast,
                0xFF, ssd1306_set_entire_on, ssd1306_set_normal_display,
                ssd1306_set_charge_pump, 0x14, ssd1306_set_scroll | 0x00,
                ssd1306_set_display | 0x01,
            };

            ssd1306_send_command_list(commands, count_of(commands));
        }

        // Cria a lista de comandos para configurar o scrolling
        void ssd1306_scroll(bool set) {
            uint8_t commands[] = {
                ssd1306_set_horizontal_scroll | 0x00, 0x00, 0x00, 0x00, 0x03,
                0x00, 0xFF, ssd1306_set_scroll | (set ? 0x01 : 0)
            };

            ssd1306_send_command_list(commands, count_of(commands));
        }

        // Atualiza uma parte do display com uma área de renderização
        void render_on_display(uint8_t *ssd, struct render_area *area) {
            uint8_t commands[] = {
                ssd1306_set_column_address, area->start_column, area->end_column,
                ssd1306_set_page_address, area->start_page, area->end_page
            };

            ssd1306_send_command_list(commands, count_of(commands));
            ssd1306_send_buffer(ssd, area->buffer_length);
        }

        // Determina o pixel a ser aceso (no display) de acordo com a coordenada fornecida
        void ssd1306_set_pixel(uint8_t *ssd, int x, int y, bool set) {
            assert(x >= 0 && x < ssd1306_width && y >= 0 && y < ssd1306_height);

            const int bytes_per_row = ssd1306_width;

            int byte_idx = (y / 8) * bytes_per_row + x;
            uint8_t byte = ssd[byte_idx];

            if (set) {
                byte |= 1 << (y % 8);
            }
            else {
                byte &= ~(1 << (y % 8));
            }

            ssd[byte_idx] = byte;
        }

        // Algoritmo de Bresenham básico
        void ssd1306_draw_line(uint8_t *ssd, int x_0, int y_0, int x_1, int y_1, bool set) {
            int dx = abs(x_1 - x_0); // Deslocamentos
            int dy = -abs(y_1 - y_0);
            int sx = x_0 < x_1 ? 1 : -1; // Direção de avanço
            int sy = y_0 < y_1 ? 1 : -1;
            int error = dx + dy; // Erro acumulado
            int error_2;

            while (true) {
                ssd1306_set_pixel(ssd, x_0, y_0, set); // Acende pixel no ponto atual
                if (x_0 == x_1 && y_0 == y_1) {
                    break; // Verifica se o ponto final foi alcançado
                }

                error_2 = 2 * error; // Ajusta o erro acumulado

                if (error_2 >= dy) {
                    error += dy;
                    x_0 += sx; // Avança na direção x
                }
                if (error_2 <= dx) {
                    error += dx;
                    y_0 += sy; // Avança na direção y
                }
            }
        }

        // Adquire os pixels para um caractere (de acordo com ssd1306_font.h)
        inline int ssd1306_get_font(uint8_t character){
            if (character >= 'A' && character <= 'Z') {
                return character - 'A' + 1;
            }
            else if (character >= '0' && character <= '9') {
                return character - '0' + 27;
            }
            else if (character == '.') return 37;
            else if (character == '%') return 38;
            else if (character == '-') return 39;
            else if (character == ':') return 40;

            else return 0;
        }

        // Desenha um único caractere no display
        void ssd1306_draw_char(uint8_t *ssd, int16_t x, int16_t y, uint8_t character) {
            if (x > ssd1306_width - 8 || y > ssd1306_height - 8) {
                return;
            }

            y = y / 8;

            character = toupper(character);
            int idx = ssd1306_get_font(character);
            int fb_idx = y * 128 + x;

            for (int i = 0; i < 8; i++) {
                ssd[fb_idx++] = font[idx * 8 + i];
            }
        }

        // Desenha uma string, chamando a função de desenhar caractere várias vezes
        void ssd1306_draw_string(uint8_t *ssd, int16_t x, int16_t y, char *string) {
            if (x > ssd1306_width - 8 || y > ssd1306_height - 8) {
                return;
            }

            while (*string) {
                ssd1306_draw_char(ssd, x, y, *string++);
                x += 8;
            }
        }

        // Comando de configuração com base na estrutura ssd1306_t
        void ssd1306_command(ssd1306_t *ssd, uint8_t command) {
        ssd->port_buffer[1] = command;
        i2c_write_blocking(
            ssd->i2c_port, ssd->address, ssd->port_buffer, 2, false );
        }

        // Função de configuração do display para o caso do bitmap
        void ssd1306_config(ssd1306_t *ssd) {
            ssd1306_command(ssd, ssd1306_set_display | 0x00);
            ssd1306_command(ssd, ssd1306_set_memory_mode);
            ssd1306_command(ssd, 0x01);
            ssd1306_command(ssd, ssd1306_set_display_start_line | 0x00);
            ssd1306_command(ssd, ssd1306_set_segment_remap | 0x01);
            ssd1306_command(ssd, ssd1306_set_mux_ratio);
            ssd1306_command(ssd, ssd1306_height - 1);
            ssd1306_command(ssd, ssd1306_set_common_output_direction | 0x08);
            ssd1306_command(ssd, ssd1306_set_display_offset);
            ssd1306_command(ssd, 0x00);
            ssd1306_command(ssd, ssd1306_set_common_pin_configuration);
            ssd1306_command(ssd, 0x12);
            ssd1306_command(ssd, ssd1306_set_display_clock_divide_ratio);
            ssd1306_command(ssd, 0x80);
            ssd1306_command(ssd, ssd1306_set_precharge);
            ssd1306_command(ssd, 0xF1);
            ssd1306_command(ssd, ssd1306_set_vcomh_deselect_level);
            ssd1306_command(ssd, 0x30);
            ssd1306_command(ssd, ssd1306_set_contrast);
            ssd1306_command(ssd, 0xFF);
            ssd1306_command(ssd, ssd1306_set_entire_on);
            ssd1306_command(ssd, ssd1306_set_normal_display);
            ssd1306_command(ssd, ssd1306_set_charge_pump);
            ssd1306_command(ssd, 0x14);
            ssd1306_command(ssd, ssd1306_set_display | 0x01);
        }

        // Inicializa o display para o caso de exibição de bitmap
        void ssd1306_init_bm(ssd1306_t *ssd, uint8_t width, uint8_t height, bool external_vcc, uint8_t address, i2c_inst_t *i2c) {
            ssd->width = width;
            ssd->height = height;
            ssd->pages = height / 8U;
            ssd->address = address;
            ssd->i2c_port = i2c;
            ssd->bufsize = ssd->pages * ssd->width + 1;
            ssd->ram_buffer = calloc(ssd->bufsize, sizeof(uint8_t));
            ssd->ram_buffer[0] = 0x40;
            ssd->port_buffer[0] = 0x80;
        }

        // Envia os dados ao display
        void ssd1306_send_data(ssd1306_t *ssd) {
            ssd1306_command(ssd, ssd1306_set_column_address);
            ssd1306_command(ssd, 0);
            ssd1306_command(ssd, ssd->width - 1);
            ssd1306_command(ssd, ssd1306_set_page_address);
            ssd1306_command(ssd, 0);
            ssd1306_command(ssd, ssd->pages - 1);
            i2c_write_blocking(
            ssd->i2c_port, ssd->address, ssd->ram_buffer, ssd->bufsize, false );
        }

        // Desenha o bitmap (a ser fornecido em display_oled.c) no display
        void ssd1306_draw_bitmap(ssd1306_t *ssd, const uint8_t *bitmap) {
            for (int i = 0; i < ssd->bufsize - 1; i++) {
                ssd->ram_buffer[i + 1] = bitmap[i];

                ssd1306_send_data(ssd);
            }
        }

        #define FONT_PIXEL_WIDTH  8
        #define FONT_PIXEL_HEIGHT 8

        const uint SSD1306_I2C_SDA = 14;
        const uint SSD1306_I2C_SCL = 15;

        volatile struct render_area frame_area = {
            start_column : 0,
            end_column : ssd1306_width - 1,
            start_page : 0,
            end_page : ssd1306_n_pages - 1
        };

        volatile uint8_t ssd[ssd1306_buffer_length];

        void ssd1306_i2c_setup(){
            i2c_init(i2c1, ssd1306_i2c_clock * 1000);
            gpio_set_function(SSD1306_I2C_SDA, GPIO_FUNC_I2C);
            gpio_set_function(SSD1306_I2C_SCL, GPIO_FUNC_I2C);
            gpio_pull_up(SSD1306_I2C_SDA);
            gpio_pull_up(SSD1306_I2C_SCL);
            ssd1306_init();
            calculate_render_area_buffer_length(&frame_area);
        }

        void fill_screen(int color){
            memset(ssd, color, ssd1306_buffer_length);
            render_on_display(ssd, &frame_area);
        }    

        void clear_screen(void){
            fill_screen(0U);
        }    

        #define STRINGS(...) (char*[]){__VA_ARGS__}
        void render_strings(char* strings[], uint count){
            int y = 0;
            for (uint i = 0; i < count; i++)
            {
                ssd1306_draw_string(ssd, 0, y, strings[i]);
                y += FONT_PIXEL_HEIGHT;
            }
            render_on_display(ssd, &frame_area);
        }


        void ssd1306_clear_rect_area(int x_start, int y_start, int width, int height) {
            for (int x = x_start; x < x_start + width; x++) {
                for (int y = y_start; y < y_start + height; y++) {
                    if (x >= 0 && x < ssd1306_width && y >= 0 && y < ssd1306_height) {
                        ssd1306_set_pixel((uint8_t*)ssd, x, y, false);
                    }
                }
            }
        }


    #endif

#endif