/*
 * evaluate.c — BitDogLab Model Evaluation Pipeline
 *
 * Receives a trained model over USB CDC (COM port), then runs live
 * joystick → LED matrix inference.  Designed to work with .bin files
 * produced by metric_listener.py.
 *
 * ── Modes (B button toggles) ─────────────────────────────────────────────
 *  WAITING_FOR_MODEL   Listens on USB stdin for a model upload
 *  INFERENCE           Joystick position → NN → 5×5 LED matrix (blue)
 *
 * ── Upload protocol (host → device, over USB stdio) ──────────────────────
 *  E:MODEL,<hidden_layers>,<hidden>,<act_hidden>,<act_output>,<total_bytes>
 *      act_hidden / act_output: 0=sigmoid  1=relu  2=linear
 *      Inputs are always 2 (joystick x,y), outputs always 25 (5×5 matrix).
 *
 *  E:DATA,<chunk_idx>,<base64_payload>       (one per chunk, 45 raw bytes)
 *  E:END,<total_chunks>
 *
 * ── Device responses ─────────────────────────────────────────────────────
 *  E:READY                                   (emitted when entering WAITING)
 *  E:ACK                                     (after each E:DATA)
 *  E:LOAD_OK,<total_weights>                 (model loaded successfully)
 *  E:LOAD_FAIL,<reason>                      (error during load)
 *
 * ── Companion script ─────────────────────────────────────────────────────
 *  Use model_sender.py to push a .bin file:
 *      python model_sender.py --port COM3 --model reluosig_l2x8_lr0.1.bin
 */

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>

#include "pico/stdlib.h"
#include "pico/rand.h"

#define BOARD_PWM_HELPERS
#define BOARD_BUTTON_HELPERS
#define BOARD_ADC_HELPERS
#include "bitdoglab.h"

#define BDLED_IMPLEMENTATION
#include "bdl_led_matrix.h"

#define SSD1306_IMPLEMENTATION
#include "stb_ssd1306.h"

#include "rtos.h"
#include "gennan.h"

/* ── RNG (required by genann) ─────────────────────────────────────────────── */
int internal_rand(void)   { return (int)(get_rand_32() >> 1); }
int INTERNAL_RAND_MAX     = 0x7FFFFFFF;

/* ═══════════════════════════════════════════════════════════════════════════
 * Constants
 * ═══════════════════════════════════════════════════════════════════════════ */
#define NN_INPUTS     2
#define NN_OUTPUTS    25
#define MATRIX_ROWS   5
#define MATRIX_COLS   5
#define MATRIX_CELLS  (MATRIX_ROWS * MATRIX_COLS)
#define INFER_THRESH  0.2
#define LCD_LINE_H    12
#define LCD_BUF_SZ    64

#define MODEL_CHUNK_BYTES  45
#define MAX_LINE_LEN       128
#define RECV_TIMEOUT_MS    30000   /* 30 s total timeout per upload */

/* ═══════════════════════════════════════════════════════════════════════════
 * Modes & display
 * ═══════════════════════════════════════════════════════════════════════════ */
typedef enum { EVAL_MODE_WAITING, EVAL_MODE_INFERENCE } EvalMode;

typedef struct {
    EvalMode mode;
    bool     has_model;
    char     model_name[24];
    int      hidden_layers;
    int      hidden;
    int      total_weights;
} LCDPayload;

static QueueHandle_t xLCDQ;
static QueueHandle_t xBtnQ;
static void push_lcd(const LCDPayload *p) { xQueueOverwrite(xLCDQ, p); }

/* ═══════════════════════════════════════════════════════════════════════════
 * Global state
 * ═══════════════════════════════════════════════════════════════════════════ */
static volatile EvalMode g_mode      = EVAL_MODE_WAITING;
static volatile bool     g_has_model = false;

static genann   *g_ann = NULL;
static LedMatrix g_matrix = {0};

static char g_model_name[24] = {0};
static int  g_hidden_layers  = 0;
static int  g_hidden         = 0;

/* ═══════════════════════════════════════════════════════════════════════════
 * Activation function helpers
 * ═══════════════════════════════════════════════════════════════════════════ */
typedef double (*ActFn)(const genann *, double);

static ActFn act_from_id(int id) {
    switch (id) {
        case 0:  return genann_act_sigmoid;
        case 1:  return genann_act_relu;
        case 2:  return genann_act_linear;
        default: return genann_act_sigmoid;
    }
}

static const char *act_tag(int id) {
    switch (id) {
        case 0:  return "sig";
        case 1:  return "relu";
        case 2:  return "lin";
        default: return "?";
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Base64 decoder
 * ═══════════════════════════════════════════════════════════════════════════ */
static const int8_t B64_DEC[128] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
    52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
    15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
    -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
    41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
};

/* Decodes base64 in-place. Returns number of decoded bytes, or -1 on error. */
static int b64_decode(const char *src, uint8_t *dst, int dst_cap) {
    int len = (int)strlen(src);
    /* strip trailing padding for counting */
    int pad = 0;
    if (len > 0 && src[len - 1] == '=') pad++;
    if (len > 1 && src[len - 2] == '=') pad++;

    int out_len = (len / 4) * 3 - pad;
    if (out_len > dst_cap) return -1;

    int j = 0;
    for (int i = 0; i < len; i += 4) {
        uint8_t a = (i   < len && src[i]   != '=') ? (uint8_t)B64_DEC[(unsigned char)src[i]]   : 0;
        uint8_t b = (i+1 < len && src[i+1] != '=') ? (uint8_t)B64_DEC[(unsigned char)src[i+1]] : 0;
        uint8_t c = (i+2 < len && src[i+2] != '=') ? (uint8_t)B64_DEC[(unsigned char)src[i+2]] : 0;
        uint8_t d = (i+3 < len && src[i+3] != '=') ? (uint8_t)B64_DEC[(unsigned char)src[i+3]] : 0;

        if (j < dst_cap) dst[j++] = (a << 2) | (b >> 4);
        if (j < dst_cap && i + 2 < len && src[i+2] != '=') dst[j++] = (b << 4) | (c >> 2);
        if (j < dst_cap && i + 3 < len && src[i+3] != '=') dst[j++] = (c << 6) | d;
    }
    return out_len;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Serial line reader (non-blocking, FreeRTOS-friendly)
 * ═══════════════════════════════════════════════════════════════════════════ */
static bool read_line(char *buf, int buf_sz, uint32_t timeout_ms) {
    int pos = 0;
    uint32_t deadline = to_ms_since_boot(get_absolute_time()) + timeout_ms;
    while (to_ms_since_boot(get_absolute_time()) < deadline) {
        int c = getchar_timeout_us(500);
        if (c == PICO_ERROR_TIMEOUT) {
            vTaskDelay(pdMS_TO_TICKS(1));
            continue;
        }
        if (c == '\r') continue;
        if (c == '\n') {
            if (pos == 0) continue;
            buf[pos] = '\0';
            return true;
        }
        if (pos < buf_sz - 1)
            buf[pos++] = (char)c;
    }
    return false;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Model receive state machine
 *
 * Returns a new genann* on success, NULL on failure.
 * Caller must genann_free() the result.
 * ═══════════════════════════════════════════════════════════════════════════ */
static genann *receive_model(void) {
    char line[MAX_LINE_LEN];

    /* ── Wait for E:MODEL header ────────────────────────────────────────── */
    printf("E:READY\r\n");

    while (g_mode == EVAL_MODE_WAITING) {
        if (!read_line(line, sizeof(line), 2000)) continue;

        if (strncmp(line, "E:MODEL,", 8) != 0) {
            printf("[EVAL] (ignored) %s\n", line);
            continue;
        }

        int hl, h, act_h, act_o, total_bytes;
        if (sscanf(line + 8, "%d,%d,%d,%d,%d", &hl, &h, &act_h, &act_o, &total_bytes) != 5) {
            printf("E:LOAD_FAIL,bad_header\r\n");
            continue;
        }

        int expected_weights = total_bytes / (int)sizeof(double);
        printf("[EVAL] Header: hl=%d h=%d act_h=%s act_o=%s bytes=%d weights=%d\n",
               hl, h, act_tag(act_h), act_tag(act_o), total_bytes, expected_weights);

        /* Free previous model if any */
        if (g_ann) { genann_free(g_ann); g_ann = NULL; }

        genann *ann = genann_init(NN_INPUTS, hl, h, NN_OUTPUTS);
        if (!ann) {
            printf("E:LOAD_FAIL,oom\r\n");
            continue;
        }

        if (ann->total_weights != expected_weights) {
            printf("E:LOAD_FAIL,weight_mismatch_%d_vs_%d\r\n",
                   ann->total_weights, expected_weights);
            genann_free(ann);
            continue;
        }

        ann->activation_hidden = act_from_id(act_h);
        ann->activation_output = act_from_id(act_o);

        printf("E:ACK\r\n");

        /* ── Receive base64 data chunks ─────────────────────────────────── */
        uint8_t *weight_buf = (uint8_t *)ann->weight;
        int bytes_received = 0;
        int chunks_received = 0;
        bool error = false;

        while (g_mode == EVAL_MODE_WAITING) {
            if (!read_line(line, sizeof(line), RECV_TIMEOUT_MS)) {
                printf("E:LOAD_FAIL,timeout\r\n");
                error = true;
                break;
            }

            if (strncmp(line, "E:DATA,", 7) == 0) {
                /* Parse: E:DATA,<chunk_idx>,<base64> */
                char *p = line + 7;
                /* skip chunk_idx */
                char *comma = strchr(p, ',');
                if (!comma) {
                    printf("E:LOAD_FAIL,bad_data_line\r\n");
                    error = true;
                    break;
                }
                char *b64 = comma + 1;

                /* Decode into weight buffer */
                uint8_t tmp[MODEL_CHUNK_BYTES + 4];
                int decoded = b64_decode(b64, tmp, sizeof(tmp));
                if (decoded < 0) {
                    printf("E:LOAD_FAIL,b64_decode\r\n");
                    error = true;
                    break;
                }

                if (bytes_received + decoded > total_bytes) {
                    printf("E:LOAD_FAIL,overflow\r\n");
                    error = true;
                    break;
                }

                memcpy(weight_buf + bytes_received, tmp, (size_t)decoded);
                bytes_received += decoded;
                chunks_received++;

                printf("E:ACK\r\n");

            } else if (strncmp(line, "E:END,", 6) == 0) {
                int expected_chunks = 0;
                sscanf(line + 6, "%d", &expected_chunks);

                if (chunks_received != expected_chunks) {
                    printf("E:LOAD_FAIL,chunks_%d_vs_%d\r\n",
                           chunks_received, expected_chunks);
                    error = true;
                }
                if (bytes_received != total_bytes) {
                    printf("E:LOAD_FAIL,bytes_%d_vs_%d\r\n",
                           bytes_received, total_bytes);
                    error = true;
                }
                break;

            } else {
                printf("[EVAL] (ignored) %s\n", line);
            }
        }

        if (error || g_mode != EVAL_MODE_WAITING) {
            genann_free(ann);
            if (g_mode != EVAL_MODE_WAITING)
                printf("[EVAL] Mode changed during upload, aborting\n");
            return NULL;
        }

        /* Build a human-readable name */
        snprintf(g_model_name, sizeof(g_model_name), "%so%s_l%dx%d",
                 act_tag(act_h), act_tag(act_o), hl, h);
        g_hidden_layers = hl;
        g_hidden        = h;

        printf("E:LOAD_OK,%d\r\n", ann->total_weights);
        printf("[EVAL] Model loaded: %s  (%d weights, %d bytes)\n",
               g_model_name, ann->total_weights, total_bytes);

        return ann;
    }

    return NULL;  /* mode changed before model received */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * LED matrix helpers
 * ═══════════════════════════════════════════════════════════════════════════ */
static void matrix_show_inference(const double *out) {
    if (xSemaphoreTake(xMatrixMutex, pdMS_TO_TICKS(100)) != pdPASS) return;
    bdl_matrixClear(&g_matrix);
    for (int row = 0; row < MATRIX_ROWS; row++) {
        for (int col = 0; col < MATRIX_COLS; col++) {
            double v = out[row * MATRIX_COLS + col];
            if (v >= INFER_THRESH) {
                float w = 0.1f * (float)((v - INFER_THRESH) / (1.0 - INFER_THRESH));
                bdl_matrixSetPixel(&g_matrix, row, col, 0, 0, 255, w);
            }
        }
    }
    xSemaphoreGive(xMatrixMutex);
}

static void matrix_clear(void) {
    if (xSemaphoreTake(xMatrixMutex, pdMS_TO_TICKS(100)) != pdPASS) return;
    bdl_matrixClear(&g_matrix);
    xSemaphoreGive(xMatrixMutex);
}

/* Flash green for successful model load */
static void matrix_flash_ok(void) {
    for (int i = 0; i < 3; i++) {
        if (xSemaphoreTake(xMatrixMutex, pdMS_TO_TICKS(100)) == pdPASS) {
            bdl_matrixClear(&g_matrix);
            for (int r = 0; r < MATRIX_ROWS; r++)
                for (int c = 0; c < MATRIX_COLS; c++)
                    bdl_matrixSetPixel(&g_matrix, r, c, 0, 255, 0, 0.05f);
            xSemaphoreGive(xMatrixMutex);
        }
        vTaskDelay(pdMS_TO_TICKS(150));
        matrix_clear();
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ISR
 * ═══════════════════════════════════════════════════════════════════════════ */
static volatile uint32_t g_last_isr_ms = 0;

void ISR_HandleButtons(uint gpio, uint32_t events) {
    (void)events;
    uint32_t now = to_ms_since_boot(get_absolute_time());
    BaseType_t hp = pdFALSE;
    if (now - g_last_isr_ms > 100 &&
        (gpio == BOARD_BUTTON_B || gpio == BOARD_BUTTON_JOYSTICK)) {
        xQueueOverwriteFromISR(xBtnQ, &gpio, &hp);
        g_last_isr_ms = now;
    }
    portYIELD_FROM_ISR(hp);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Tasks
 * ═══════════════════════════════════════════════════════════════════════════ */

void deviceAliveTask(void *p) {
    (void)p;
    setup_rgb_led_with_bright();
    bool st = false;
    const Color white = {255, 255, 255};
    for (;;) {
        rgb_led_put_with_bright(white, st ? 3.0f : 0.0f);
        st = !st;
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

void buttonsTask(void *p) {
    (void)p;
    uint32_t btn;
    for (;;) {
        if (xQueueReceive(xBtnQ, &btn, portMAX_DELAY) != pdPASS) continue;
        if (gpio_get((uint)btn) != BOARD_BUTTON_ON) continue;

        if (btn == BOARD_BUTTON_B) {
            if (g_mode == EVAL_MODE_WAITING) {
                if (g_has_model) {
                    g_mode = EVAL_MODE_INFERENCE;
                    printf("[BTN] Mode -> INFERENCE\n");
                } else {
                    printf("[BTN] No model loaded, staying in WAITING\n");
                }
            } else {
                g_mode = EVAL_MODE_WAITING;
                printf("[BTN] Mode -> WAITING_FOR_MODEL\n");
            }
        }
    }
}

void readJoystickTask(void *p) {
    (void)p;
    setup_joystick();
    const uint16_t DZ = 150;
    float alpha = 0.4f;
    JoystickState s = {0};
    for (;;) {
        s = read_joystick();
        s = apply_low_pass_filter(s, &alpha);
        s.x = apply_joystick_deadzone(s.x, DZ) == 0 ? 2048 : s.x;
        s.y = apply_joystick_deadzone(s.y, DZ) == 0 ? 2048 : s.y;
        xQueueOverwrite(xJoystickQueue, &s);
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

void matrixUpdateTask(void *p) {
    (void)p;
    bdl_matrixInit(&g_matrix, LED_PIN, MATRIX_ROWS, MATRIX_COLS);
    xSemaphoreGive(xMatrixMutex);
    for (;;) {
        if (xSemaphoreTake(xMatrixMutex, portMAX_DELAY) == pdPASS) {
            bdl_matrixWrite(&g_matrix);
            xSemaphoreGive(xMatrixMutex);
        }
        vTaskDelay(pdMS_TO_TICKS(33));
    }
}

void lcdUpdateTask(void *p) {
    (void)p;
    ssd1306_i2c_setup();
    clear_screen();
    LCDPayload info = {0};
    static char buf[LCD_BUF_SZ];
    uint8_t y;
    for (;;) {
        xQueueReceive(xLCDQ, &info, pdMS_TO_TICKS(200));
        ssd1306_clear_rect_area(0, 0, ssd1306_width, ssd1306_height);
        y = 2;

        if (info.mode == EVAL_MODE_WAITING) {
            ssd1306_draw_string((uint8_t *)ssd, 0, y, "[WAITING]");
            y += LCD_LINE_H;
            if (info.has_model) {
                ssd1306_draw_string((uint8_t *)ssd, 0, y, (char *)info.model_name);
                y += LCD_LINE_H;
                ssd1306_draw_string((uint8_t *)ssd, 0, y, "B -> inference");
                y += LCD_LINE_H;
                ssd1306_draw_string((uint8_t *)ssd, 0, y, "or send new model");
            } else {
                ssd1306_draw_string((uint8_t *)ssd, 0, y, "Send model via COM");
                y += LCD_LINE_H;
                ssd1306_draw_string((uint8_t *)ssd, 0, y, "Listening...");
            }
        } else {
            ssd1306_draw_string((uint8_t *)ssd, 0, y,
                info.has_model ? "[INFERENCE]" : "[INFER] No model");
            y += LCD_LINE_H;
            if (info.has_model) {
                ssd1306_draw_string((uint8_t *)ssd, 0, y, (char *)info.model_name);
                y += LCD_LINE_H;
                snprintf(buf, sizeof(buf), "hl=%d h=%d w=%d",
                         info.hidden_layers, info.hidden, info.total_weights);
                ssd1306_draw_string((uint8_t *)ssd, 0, y, buf);
                y += LCD_LINE_H;
                ssd1306_draw_string((uint8_t *)ssd, 0, y, "Joystick->LEDs");
            }
        }
        render_on_display((uint8_t *)ssd, &frame_area);
        vTaskDelay(pdMS_TO_TICKS(200));
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * evaluateTask — main loop
 * ═══════════════════════════════════════════════════════════════════════════ */
void evaluateTask(void *p) {
    (void)p;
    vTaskDelay(pdMS_TO_TICKS(600));

    LCDPayload lcd = {0};

    for (;;) {

        /* ══ WAITING_FOR_MODEL ═════════════════════════════════════════════ */
        if (g_mode == EVAL_MODE_WAITING) {
            lcd.mode         = EVAL_MODE_WAITING;
            lcd.has_model    = g_has_model;
            strncpy(lcd.model_name, g_model_name, sizeof(lcd.model_name) - 1);
            lcd.hidden_layers = g_hidden_layers;
            lcd.hidden        = g_hidden;
            lcd.total_weights = g_ann ? g_ann->total_weights : 0;
            push_lcd(&lcd);

            genann *new_ann = receive_model();
            if (new_ann) {
                if (g_ann) genann_free(g_ann);
                g_ann       = new_ann;
                g_has_model = true;

                lcd.has_model    = true;
                lcd.total_weights = g_ann->total_weights;
                lcd.hidden_layers = g_hidden_layers;
                lcd.hidden        = g_hidden;
                strncpy(lcd.model_name, g_model_name, sizeof(lcd.model_name) - 1);
                push_lcd(&lcd);

                matrix_flash_ok();

                /* Auto-switch to inference after successful load */
                g_mode = EVAL_MODE_INFERENCE;
                printf("[EVAL] Auto-switching to INFERENCE mode\n");
            }
            continue;
        }

        /* ══ INFERENCE ════════════════════════════════════════════════════ */
        lcd.mode          = EVAL_MODE_INFERENCE;
        lcd.has_model     = g_has_model;
        lcd.hidden_layers = g_hidden_layers;
        lcd.hidden        = g_hidden;
        lcd.total_weights = g_ann ? g_ann->total_weights : 0;
        strncpy(lcd.model_name, g_model_name, sizeof(lcd.model_name) - 1);
        push_lcd(&lcd);

        if (g_ann) {
            JoystickState joy = {0};
            if (xQueuePeek(xJoystickQueue, &joy, pdMS_TO_TICKS(100)) == pdPASS) {
                double nn_in[2] = {
                    (double)map_joystick_value(joy.x, -1.0f, 1.0f),
                    (double)map_joystick_value(joy.y, -1.0f, 1.0f),
                };
                matrix_show_inference(genann_run(g_ann, nn_in));
            }
        }
        vTaskDelay(pdMS_TO_TICKS(50));
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(void) {
    stdio_init_all();
    sleep_ms(2000);
    printf("[SYS] evaluate.c — Model Evaluation Pipeline v1\n");

    init_buttons(&ISR_HandleButtons);
    init_rtos_handlers();

    xLCDQ = xQueueCreate(1, sizeof(LCDPayload));
    xBtnQ = xQueueCreate(1, sizeof(uint32_t));

    /* Core 0 */
    xTaskCreateAffinitySet(deviceAliveTask, "Alive",    512,  NULL,  1, RP2040_CORE_0, NULL);
    xTaskCreateAffinitySet(buttonsTask,     "Buttons",  512,  NULL,  6, RP2040_CORE_0, NULL);
    xTaskCreateAffinitySet(evaluateTask,    "Evaluate", 8192, NULL,  5, RP2040_CORE_0, NULL);

    /* Core 1 */
    xTaskCreateAffinitySet(readJoystickTask, "Joystick", 1024, NULL, 10, RP2040_CORE_1, NULL);
    xTaskCreateAffinitySet(matrixUpdateTask, "Matrix",   1024, NULL, 10, RP2040_CORE_1, NULL);
    xTaskCreateAffinitySet(lcdUpdateTask,    "LCD",      2048, NULL,  7, RP2040_CORE_1, NULL);

    vTaskStartScheduler();
    while (true) tight_loop_contents();
}
