/*
 * BitDogLab — Coleta de dados com triggered logic (v3)
 *
 * Fluxo por amostra:
 *   1. Gera padrão de linha na matriz (verde)
 *   2. Captura snapshot early_joy
 *   3. A cada 750 ms verifica:
 *        - joystick saiu do centro?
 *        - está apontando na direção correta (dot product >= 0)?
 *        - parou de mover (dx²+dy² < tolerância)?
 *      → quando as três condições batem: triggered = true
 *   4. Envia: DATA,joy_x,joy_y,p0,...,p24\n
 *   5. Repete
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "pico/stdlib.h"
#include "pico/rand.h"

#define BOARD_ADC_HELPERS
#define BOARD_PWM_HELPERS
#define BOARD_BUTTON_HELPERS
#include "bitdoglab.h"

#define BDLED_IMPLEMENTATION
#include "bdl_led_matrix.h"

#include "rtos.h"

/* ── Constantes ───────────────────────────────────────────────────────────── */
#define MATRIX_ROWS   5
#define MATRIX_COLS   5
#define MATRIX_CELLS  25

#define POLL_MS       750    /* intervalo de checagem do joystick              */
#define TOLERANCE     (64*64) /* dx²+dy² mínimo para considerar "moveu"       */

/* ── Estado global ────────────────────────────────────────────────────────── */
static LedMatrix g_matrix   = {0};
static double    g_target_pattern[MATRIX_CELLS] = {0};
static int       g_current_endpoint = 0;

/* ── Padrões round-robin (Fisher-Yates) ──────────────────────────────────── */
static const int LINE_ENDPOINTS[][2] = {
    {0,0},{0,1},{0,2},{0,3},{0,4},
    {1,3},{2,3},{3,3},{3,2},{3,1},{2,1},{1,1},{1,2},
    {1,4},{2,4},{3,4},
    {4,4},{4,3},{4,2},{4,1},{4,0},
    {3,0},{2,0},{1,0}
};
#define TOTAL_DIRECTIONS 24

static int g_dir_order[TOTAL_DIRECTIONS];
static int g_dir_idx = TOTAL_DIRECTIONS;

static void shuffle_directions(void) {
    for (int i = 0; i < TOTAL_DIRECTIONS; i++) g_dir_order[i] = i;
    for (int i = TOTAL_DIRECTIONS - 1; i > 0; i--) {
        int j = (int)(get_rand_32() % (i + 1));
        int tmp = g_dir_order[i]; g_dir_order[i] = g_dir_order[j]; g_dir_order[j] = tmp;
    }
    g_dir_idx = 0;
}

static void gen_line_pattern(double *pat) {
    if (g_dir_idx >= TOTAL_DIRECTIONS) shuffle_directions();

    g_current_endpoint = g_dir_order[g_dir_idx];

    int ep = g_dir_order[g_dir_idx++];
    int r0=2, c0=2, r1=LINE_ENDPOINTS[ep][0], c1=LINE_ENDPOINTS[ep][1];

    for (int i = 0; i < MATRIX_CELLS; i++) pat[i] = 0.1;

    int dx=abs(c1-c0), sx=(c0<c1)?1:-1;
    int dy=-abs(r1-r0), sy=(r0<r1)?1:-1;
    int err=dx+dy, r=r0, c=c0;
    for (;;) {
        if (r>=0&&r<MATRIX_ROWS&&c>=0&&c<MATRIX_COLS)
            pat[r*MATRIX_COLS+c] = 0.9;
        if (r==r1&&c==c1) break;
        int e2=2*err;
        if (e2>=dy){err+=dy;c+=sx;}
        if (e2<=dx){err+=dx;r+=sy;}
    }
}

/* ── Matriz ───────────────────────────────────────────────────────────────── */
static void matrix_show_pattern(const double *pat) {
    if (xSemaphoreTake(xMatrixMutex, pdMS_TO_TICKS(50)) != pdPASS) return;
    bdl_matrixClear(&g_matrix);
    for (int row=0; row<MATRIX_ROWS; row++)
        for (int col=0; col<MATRIX_COLS; col++)
            if (pat[row*MATRIX_COLS+col] >= 0.5)
                bdl_matrixSetPixel(&g_matrix, row, col, 0, 255, 0, 0.1f);
    xSemaphoreGive(xMatrixMutex);
}

static void matrix_clear(void) {
    if (xSemaphoreTake(xMatrixMutex, pdMS_TO_TICKS(50)) != pdPASS) return;
    bdl_matrixClear(&g_matrix);
    xSemaphoreGive(xMatrixMutex);
}

/* ── Serial ───────────────────────────────────────────────────────────────── */
static void serial_send(double joy_x, double joy_y, const double *pat) {
    printf("DATA,%.4f,%.4f", joy_x, joy_y);
    for (int i = 0; i < MATRIX_CELLS; i++)
        printf(",%.4f", pat[i]);
    printf("\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Tasks
 * ═══════════════════════════════════════════════════════════════════════════ */

void readJoystickTask(void *params) {
    setup_joystick();
    const uint16_t DEADZONE = 150;
    float ema = 0.4f;
    JoystickState s = {0};
    for (;;) {
        s = read_joystick();
        s = apply_low_pass_filter(s, &ema);
        int xc = apply_joystick_deadzone(s.x, DEADZONE);
        int yc = apply_joystick_deadzone(s.y, DEADZONE);
        s.x = (xc==0) ? 2048 : s.x;
        s.y = (yc==0) ? 2048 : s.y;
        xQueueOverwrite(xJoystickQueue, &s);
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

void matrixUpdateTask(void *params) {
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

void dataCollectTask(void *params) {
    vTaskDelay(pdMS_TO_TICKS(400));

    JoystickState early_joy = {0};
    JoystickState joy       = {0};
    uint32_t count = 0;

    printf("[SYS] Pronto. Mova o joystick para a direcao indicada e segure.\n");

    for (;;) {
        /* 1. Gera e exibe padrão */
        gen_line_pattern(g_target_pattern);
        matrix_show_pattern(g_target_pattern);
        printf("[NN] Nova linha gerada (ep=%d)\n", g_current_endpoint);

        /* 2. Snapshot inicial do joystick */
        if (xQueuePeek(xJoystickQueue, &early_joy, pdMS_TO_TICKS(50)) != pdPASS)
            early_joy.x = early_joy.y = 2048;

        /* 3. Loop de trigger — igual ao v3 */
        bool triggered = false;
        while (!triggered) {
            vTaskDelay(pdMS_TO_TICKS(POLL_MS));

            if (xQueuePeek(xJoystickQueue, &joy, pdMS_TO_TICKS(50)) != pdPASS)
                continue;

            /* Ignora se ainda no centro */
            if (joy.x == 2048 && joy.y == 2048)
                continue;

            int32_t dx = (int32_t)joy.x - (int32_t)early_joy.x;
            int32_t dy = (int32_t)joy.y - (int32_t)early_joy.y;

            /* Direção esperada pelo endpoint (col → x, row → y) */
            float ep_dx = (LINE_ENDPOINTS[g_current_endpoint][1] - 2) / 2.0f;
            float ep_dy = (LINE_ENDPOINTS[g_current_endpoint][0] - 2) / 2.0f;

            float joy_dx = map_joystick_value(joy.x, -1.0f, 1.0f);
            float joy_dy = map_joystick_value(joy.y, -1.0f, 1.0f);

            /* Dot product: joystick aponta contra a direção esperada? → reseta */
            if (joy_dx * ep_dx + joy_dy * ep_dy < 0.0f) {
                early_joy = joy;
                continue;
            }

            /* Parou de mover dentro da tolerância → captura */
            if ((dx*dx + dy*dy) < (int32_t)TOLERANCE) {
                triggered = true;
            } else {
                early_joy = joy;
            }
        }

        /* 4. Captura e envia */
        double joy_x = (double)map_joystick_value(joy.x, -1.0f, 1.0f);
        double joy_y = (double)map_joystick_value(joy.y, -1.0f, 1.0f);

        serial_send(joy_x, joy_y, g_target_pattern);
        count++;
        printf("[OK] #%lu  x=%.3f y=%.3f\n", (unsigned long)count, joy_x, joy_y);

        /* Feedback visual: apaga brevemente */
        matrix_clear();
        vTaskDelay(pdMS_TO_TICKS(300));
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  main
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(void) {
    stdio_init_all();
    sleep_ms(1500);
    printf("[SYS] BitDogLab Data Collector v3\n");
    printf("[SYS] Formato: DATA,joy_x,joy_y,p0,...,p24\n");

    init_rtos_handlers();

    xTaskCreateAffinitySet(dataCollectTask,  "Collect",  2048, NULL,  5, RP2040_CORE_0, NULL);

    xTaskCreateAffinitySet(readJoystickTask, "Joystick", 1024, NULL, 10, RP2040_CORE_1, NULL);
    xTaskCreateAffinitySet(matrixUpdateTask, "Matrix",   1024, NULL, 10, RP2040_CORE_1, NULL);

    vTaskStartScheduler();
    while (true) tight_loop_contents();
}