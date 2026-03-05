/*
 * BitDogLab — Joystick → LED Matrix Neural Network  (v3)
 *
 * Arquitectura NN: 2 entradas (x,y) → 50×2 ocultos (sigmoid) → 25 saídas (sigmoid)
 *
 * BOTÃO B          → alterna Modo Treino / Inferência
 * BOTÃO A          → (treino) confirma joystick e dispara backprop
 * BOTÃO JOYSTICK   → salva pesos na flash (qualquer modo)
 *
 * Na inicialização o sistema tenta carregar um modelo previamente salvo.
 *
 * ── Por que sigmoid e não ReLU na camada oculta? ───────────────────────────
 * O genann calcula os deltas ocultos com  d = o * (1 - o) * Σ(w * d_next),
 * que é exatamente a derivada da sigmoid aplicada à saída o.
 * Com ReLU no forward pass, o pode ser 0 para entradas negativas.
 * Quando o = 0  →  d = 0  →  esse neurônio nunca mais recebe gradiente
 * (problema do "neurônio morto" / dying ReLU).
 * Usar sigmoid nas camadas ocultas torna o forward e o backprop consistentes.
 * ──────────────────────────────────────────────────────────────────────────
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "pico/stdlib.h"
#include "pico/rand.h"
#include "pico/multicore.h"
#include "hardware/flash.h"
#include "hardware/sync.h"

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


/* ── RNG ──────────────────────────────────────────────────────────────────── */
int internal_rand(void)  { return (int)(get_rand_32() >> 1); }
int INTERNAL_RAND_MAX    = 0x7FFFFFFF;

/* ── Flash: área reservada para o modelo ──────────────────────────────────── */

#define NN_FLASH_SECTORS  (16U)                                     
#define NN_FLASH_OFFSET   (PICO_FLASH_SIZE_BYTES - (NN_FLASH_SECTORS) * 4 * 1024)
#define NN_FLASH_MAGIC    0xBD42A001u 

typedef struct {
    uint32_t magic;
    int      inputs;
    int      hidden_layers;
    int      hidden;
    int      outputs;
    int      total_weights;
    uint32_t n_samples;
    uint32_t _pad;          /* alinha a 32 bytes para facilitar debug         */
} NNFlashHeader;            

/* ── Constantes ───────────────────────────────────────────────────────────── */
#define MATRIX_ROWS     5
#define MATRIX_COLS     5
#define MATRIX_CELLS    (MATRIX_ROWS * MATRIX_COLS)

#define LEARNING_RATE   0.05
#define TRAIN_EPOCHS    5
#define INFER_THRESH    0.2

#define LCD_LINE_H      12
#define LCD_BUF_SZ      64

/* ── Estado global ────────────────────────────────────────────────────────── */
static volatile AppMode       g_mode        = MODE_TRAINING;
static volatile TrainingState g_train_state = TRAIN_SHOW_LINE;
static volatile float         g_last_loss   = 1.0f;
static volatile uint32_t      g_train_count = 0;
static volatile float         g_last_acc    = 0.0f;
static volatile bool          g_save_req    = false;  /* joystick → save      */
static volatile bool          g_loaded_ok   = false;  /* modelo carregado?    */

static LedMatrix  g_matrix = {0};
static genann    *g_ann    = NULL;
static double     g_target_pattern[MATRIX_CELLS] = {0};

/* ── Geração de padrões round-robin ───────────────────────────────────────── */
static const int LINE_ENDPOINTS[][2] = {
    {0,0}, {0,1}, {0,2}, {0,3}, {0,4},
    {1,3}, {2,3}, {3,3}, {3,2}, {3,1}, {2,1}, {1,1}, {1,2}, 
    {1,4}, {2,4}, {3,4},
    {4,4}, {4,3}, {4,2}, {4,1}, {4,0},
    {3,0}, {2,0}, {1,0}
};
#define TOTAL_DIRECTIONS 24
static int g_dir_order[TOTAL_DIRECTIONS] = {0,1,2,3,
                                            4,5,6,7,
                                            8,9,10,11,
                                            12,13,14,
                                            15,16,17,
                                            18,19,20,
                                            21,22,23
                                        };
static int g_dir_idx = TOTAL_DIRECTIONS;

// fisher-yeates
static void shuffle_directions(void) {
    for (int i = TOTAL_DIRECTIONS - 1; i > 0; i--) {
        int j = (int)(get_rand_32() % (i + 1));
        int tmp = g_dir_order[i]; g_dir_order[i] = g_dir_order[j]; g_dir_order[j] = tmp;
    }
    g_dir_idx = 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Flash: salva e carrega pesos da rede neural
 *
 * Formato no flash:
 *   [ NNFlashHeader (32 bytes) | double weight[total_weights] ]
 *
 * nn_save_to_flash() aloca um buffer temporário na RAM, serializa o modelo,
 * apaga os setores necessários e reprograma.  É seguro chamar de qualquer
 * task porque save_and_disable_interrupts() garante acesso exclusivo.
 * ═══════════════════════════════════════════════════════════════════════════ */

static bool nn_save_to_flash(const genann *ann, uint32_t n_samples) {
    if (!ann) return false;

    size_t weights_bytes = (size_t)ann->total_weights * sizeof(double);
    size_t total_bytes   = sizeof(NNFlashHeader) + weights_bytes;

    /* Arredonda para múltiplo de FLASH_PAGE_SIZE (256 bytes) */
    size_t prog_size = (total_bytes + FLASH_PAGE_SIZE - 1) & ~(FLASH_PAGE_SIZE - 1u);
    size_t erase_size = (size_t)NN_FLASH_SECTORS * FLASH_SECTOR_SIZE;

    if (prog_size > erase_size) {
        printf("[FLASH] ERRO: modelo excede area reservada (%u > %u)\n",
               (unsigned)prog_size, (unsigned)erase_size);
        return false;
    }

    uint8_t *buf = malloc(prog_size);
    if (!buf) { printf("[FLASH] ERRO: malloc falhou\n"); return false; }
    memset(buf, 0xFF, prog_size);

    /* Escreve cabeçalho */
    NNFlashHeader hdr = {
        .magic         = NN_FLASH_MAGIC,
        .inputs        = ann->inputs,
        .hidden_layers = ann->hidden_layers,
        .hidden        = ann->hidden,
        .outputs       = ann->outputs,
        .n_samples     = n_samples, 
        .total_weights = ann->total_weights,
        ._pad          = 0
    };
    memcpy(buf, &hdr, sizeof(hdr));

    /* Escreve pesos */
    memcpy(buf + sizeof(hdr), ann->weight, weights_bytes);
    
    xSemaphoreGive(xFlashStartSem);
    xSemaphoreTake(xFlashReadySem, portMAX_DELAY);

    vTaskSuspendAll();

    uint32_t ints = save_and_disable_interrupts();

    flash_range_erase(NN_FLASH_OFFSET, erase_size);
    flash_range_program(NN_FLASH_OFFSET, buf, prog_size);

    restore_interrupts(ints);

    xTaskResumeAll();

    xSemaphoreGive(xFlashDoneSem);

    free(buf);
    printf("[FLASH] Modelo salvo: %u pesos, %u bytes, offset=0x%X\n",
           (unsigned)ann->total_weights, (unsigned)total_bytes, (unsigned)NN_FLASH_OFFSET);
    return true;
}

/*
 * Retorna true se encontrou um modelo válido na flash E a topologia
 * coincide com ann (a rede já deve ter sido criada com genann_init).
 * Apenas sobrescreve ann->weight; funções de ativação não são tocadas.
 */
static bool nn_load_from_flash(genann *ann, uint32_t* n_samples) {
    if (!ann) return false;

    const uint8_t *flash_ptr = (const uint8_t *)(XIP_BASE + NN_FLASH_OFFSET);
    NNFlashHeader hdr;
    memcpy(&hdr, flash_ptr, sizeof(hdr));

    if (hdr.magic != NN_FLASH_MAGIC) {
        printf("[FLASH] Nenhum modelo salvo (magic=0x%08X)\n", (unsigned)hdr.magic);
        return false;
    }
    if (hdr.inputs        != ann->inputs        ||
        hdr.hidden_layers != ann->hidden_layers  ||
        hdr.hidden        != ann->hidden         ||
        hdr.outputs       != ann->outputs        ||
        hdr.total_weights != ann->total_weights)
    {
        printf("[FLASH] Topologia incompativel, ignorando modelo salvo\n");
        return false;
    }
    *n_samples = hdr.n_samples;
    memcpy(ann->weight, flash_ptr + sizeof(hdr),
           (size_t)ann->total_weights * sizeof(double));
    printf("[FLASH] Modelo carregado: %d pesos\n", ann->total_weights);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Funções auxiliares — métricas e padrões
 * ═══════════════════════════════════════════════════════════════════════════ */

static float compute_mse(const double *out, const double *tgt, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) { float d = (float)(out[i]-tgt[i]); s += d*d; }
    return s / (float)n;
}

static float compute_accuracy(const double *out, const double *tgt, int n) {
    int ok = 0;
    for (int i = 0; i < n; i++)
        if (((out[i] >= INFER_THRESH)?1:0) == ((tgt[i] >= 0.5)?1:0)) ok++;
    return (float)ok / (float)n;
}

static void gen_line_pattern(double *pat) {
    if (g_dir_idx >= TOTAL_DIRECTIONS) shuffle_directions();
    int ep = g_dir_order[g_dir_idx++];
    int r0=2, c0=2, r1=LINE_ENDPOINTS[ep][0], c1=LINE_ENDPOINTS[ep][1];

    for (int i = 0; i < MATRIX_CELLS; i++) pat[i] = 0.1;

    int dx=abs(c1-c0), sx=(c0<c1)?1:-1;
    int dy=-abs(r1-r0), sy=(r0<r1)?1:-1;
    int err=dx+dy, r=r0, c=c0;
    for(;;) {
        if (r>=0&&r<MATRIX_ROWS&&c>=0&&c<MATRIX_COLS) pat[r*MATRIX_COLS+c]=0.9;
        if (r==r1&&c==c1) break;
        int e2=2*err;
        if(e2>=dy){err+=dy;c+=sx;}
        if(e2<=dx){err+=dx;r+=sy;}
    }
}

static void matrix_show_pattern(const double *pat, uint8_t r, uint8_t g, uint8_t b, float w) {
    if (xSemaphoreTake(xMatrixMutex, pdMS_TO_TICKS(100)) != pdPASS) return;
    bdl_matrixClear(&g_matrix);
    for (int row=0; row<MATRIX_ROWS; row++)
        for (int col=0; col<MATRIX_COLS; col++)
            if (pat[row*MATRIX_COLS+col] >= 0.5)
                bdl_matrixSetPixel(&g_matrix, row, col, r, g, b, w);
    xSemaphoreGive(xMatrixMutex);
}

static void matrix_show_inference(const double *out) {
    if (xSemaphoreTake(xMatrixMutex, pdMS_TO_TICKS(100)) != pdPASS) return;
    bdl_matrixClear(&g_matrix);
    for (int row=0; row<MATRIX_ROWS; row++) {
        for (int col=0; col<MATRIX_COLS; col++) {
            double v = out[row*MATRIX_COLS+col];
            if (v >= INFER_THRESH) {
                float w = 0.2f*(float)(v-INFER_THRESH)/(1.0f-INFER_THRESH);
                bdl_matrixSetPixel(&g_matrix, row, col, 0, 0, 255, w);
            }
        }
    }
    xSemaphoreGive(xMatrixMutex);
}

/* Pisca os LEDs em amarelo para confirmar save concluído */
static void matrix_flash_feedback(void) {
    for (int i = 0; i < 3; i++) {
        if (xSemaphoreTake(xMatrixMutex, pdMS_TO_TICKS(100)) == pdPASS) {
            bdl_matrixClear(&g_matrix);
            for (int r=0; r<MATRIX_ROWS; r++)
                for (int c=0; c<MATRIX_COLS; c++)
                    bdl_matrixSetPixel(&g_matrix, r, c, 255, 200, 0, 0.05f);
            xSemaphoreGive(xMatrixMutex);
        }
        vTaskDelay(pdMS_TO_TICKS(150));
        if (xSemaphoreTake(xMatrixMutex, pdMS_TO_TICKS(100)) == pdPASS) {
            bdl_matrixClear(&g_matrix);
            xSemaphoreGive(xMatrixMutex);
        }
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

static inline void push_lcd(AppMode mode, TrainingState ts,
                              float loss, uint32_t n, float acc, bool flash_ok) {
    LCDInfo_t info = { mode, ts, loss, n, acc, flash_ok };
    xQueueOverwrite(xLCDQueue, &info);
}

static void joy_to_nn_input(JoystickState joy, double *nn_in) {
    nn_in[0] = (double)map_joystick_value(joy.x, -1.0f, 1.0f);
    nn_in[1] = (double)map_joystick_value(joy.y, -1.0f, 1.0f);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ISR de Botões
 * ═══════════════════════════════════════════════════════════════════════════ */
static const uint32_t    DEBOUNCE_MS    = 100;
static volatile uint32_t last_isr_time  = 0;

void ISR_HandleButtons(uint gpio, uint32_t events) {
    uint32_t now = to_ms_since_boot(get_absolute_time());
    BaseType_t hp = pdFALSE;
    if (now - last_isr_time > DEBOUNCE_MS) {
        if (gpio==BOARD_BUTTON_A || gpio==BOARD_BUTTON_B || gpio==BOARD_BUTTON_JOYSTICK) {
            xQueueOverwriteFromISR(xButtonsQueue, &gpio, &hp);
            last_isr_time = now;
        }
    }
    portYIELD_FROM_ISR(hp);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Tasks
 * ═══════════════════════════════════════════════════════════════════════════ */

void deviceAliveTask(void *params) {
    setup_rgb_led_with_bright();
    bool state = false;
    const Color white = {255,255,255};
    for (;;) {
        rgb_led_put_with_bright(white, state ? 5.0f : 0.0f);
        state = !state;
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

/* ── Botões ──────────────────────────────────────────────────────────────────
 * A          → confirma joystick no treino
 * B          → alterna modo
 * Joystick   → agenda save na flash (sinaliza neuralNetTask via g_save_req)
 *              O save é feito na neuralNetTask pois ela é dona de g_ann.
 * ─────────────────────────────────────────────────────────────────────────── */
void buttonsTask(void *args) {
    BOARD_BUTTON btn = 0;
    for (;;) {
        if (xQueueReceive(xButtonsQueue, &btn, portMAX_DELAY) != pdPASS) continue;
        if (gpio_get(btn) != BOARD_BUTTON_ON)                             continue;

        if (btn == BOARD_BUTTON_B) {
            if (xSemaphoreTake(xStateMutex, pdMS_TO_TICKS(100)) == pdPASS) {
                g_mode = (g_mode == MODE_TRAINING) ? MODE_INFERENCE : MODE_TRAINING;
                if (g_mode == MODE_TRAINING) {
                    g_train_state = TRAIN_SHOW_LINE;
                    xSemaphoreTake(xTrainTriggerSem, 0);
                }
                xSemaphoreGive(xStateMutex);
            }
            printf("[BTN] Modo: %s\n", g_mode == MODE_TRAINING ? "TREINO" : "INFERENCIA");

        } else if (btn == BOARD_BUTTON_A) {
            if (g_mode == MODE_TRAINING &&
                (g_train_state == TRAIN_SHOW_LINE || g_train_state == TRAIN_WAIT_CONFIRM)) {
                xSemaphoreGive(xTrainTriggerSem);
                printf("[BTN] Treino acionado!\n");
            }

        } else if (btn == BOARD_BUTTON_JOYSTICK) {
            /* Agenda save; neuralNetTask executa quando seguro */
            g_save_req = true;
            printf("[BTN] Save solicitado\n");
        }
    }
}

void readJoystickTask(void *params) {
    setup_joystick();
    const uint16_t DEADZONE = 150;
    float ema_alpha = 0.4f;
    JoystickState s = {0};
    for (;;) {
        s = read_joystick();
        s = apply_low_pass_filter(s, &ema_alpha);
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

void core1LockoutInitTask(void *params) {
    /* Registra o IRQ handler de lockout no Core 1.
     * A partir daqui, multicore_lockout_start_blocking() pode pausar
     * este core com segurança durante escritas na flash. 
    * */
    multicore_lockout_victim_init();
    vTaskDelete(NULL);
}

void lcdUpdateTask(void *params) {
    ssd1306_i2c_setup();
    clear_screen();

    static char buf[LCD_BUF_SZ];
    LCDInfo_t info = { MODE_TRAINING, TRAIN_SHOW_LINE, 1.0f, 0, 0.0f, false };
    uint8_t y;

    for (;;) {
        xQueueReceive(xLCDQueue, &info, 0);
        ssd1306_clear_rect_area(0, 0, ssd1306_width, ssd1306_height);
        y = 2;

        /* Linha 1: modo + ícone flash */
        snprintf(buf, sizeof(buf), "[%s] %s",
                 info.mode == MODE_TRAINING ? "TREINO" : "INFER.",
                 info.flash_status ? "SAVED" : "");
        ssd1306_draw_string((uint8_t *)ssd, 0, y, buf); y += LCD_LINE_H;

        if (info.mode == MODE_TRAINING) {
            const char *msg = "";
            switch (info.train_state) {
                case TRAIN_SHOW_LINE:    msg = "Mova o joystick"; break;
                case TRAIN_WAIT_CONFIRM: msg = "Pressione A";     break;
                case TRAIN_RUNNING:      msg = "Treinando...";    break;
                case TRAIN_SHOW_RESULT:  msg = "Saida da rede";   break;
            }
            ssd1306_draw_string((uint8_t *)ssd, 0, y, msg); y += LCD_LINE_H;
            snprintf(buf, sizeof(buf), "Loss: %.5f", (double)info.loss);
            ssd1306_draw_string((uint8_t *)ssd, 0, y, buf); y += LCD_LINE_H;
            snprintf(buf, sizeof(buf), "#%-4lu Acc: %.0f%%",
                     (unsigned long)info.train_count, (double)(info.accuracy*100.0f));
            ssd1306_draw_string((uint8_t *)ssd, 0, y, buf);
        } else {
            ssd1306_draw_string((uint8_t *)ssd, 0, y, "Joystick->LEDs"); y += LCD_LINE_H;
            snprintf(buf, sizeof(buf), "Loss: %.5f", (double)info.loss);
            ssd1306_draw_string((uint8_t *)ssd, 0, y, buf); y += LCD_LINE_H;
            snprintf(buf, sizeof(buf), "Amostras: %lu", (unsigned long)info.train_count);
            ssd1306_draw_string((uint8_t *)ssd, 0, y, buf);
        }

        render_on_display((uint8_t *)ssd, &frame_area);
        vTaskDelay(pdMS_TO_TICKS(200));
    }
}

// ── Neural Net Task ─────────────────────────────────────────────────────────

#define REPLAY_SIZE 16
typedef struct { double in[2]; double target[MATRIX_CELLS]; } ReplaySample;
static ReplaySample replay[REPLAY_SIZE];
static int replay_count = 0;

void neuralNetTask(void *params) {
    vTaskDelay(pdMS_TO_TICKS(400));

    g_ann = genann_init(2, 2, 16, 25);
    if (!g_ann) { printf("[NN] ERRO: genann_init!\n"); vTaskDelete(NULL); return; }

    // XAVIER GLOROT INIT
    double scale = sqrt(1.0 / 2.0);
    for (int i = 0; i < g_ann->total_weights; i++) {
        double u1 = (get_rand_32() + 1.0) / 0x100000000ULL;
        double u2 = (get_rand_32() + 1.0) / 0x100000000ULL;
        g_ann->weight[i] = sqrt(-2.0*log(u1)) * cos(2.0*M_PI*u2) * scale;
    }

    g_ann->activation_hidden = genann_act_relu;
    g_ann->activation_output = genann_act_sigmoid;

    /* Tenta carregar modelo da flash */
    if (gpio_get(JOYSTICK_SW) == BOARD_BUTTON_ON){
        g_loaded_ok = nn_load_from_flash(g_ann,&g_train_count);
        if (g_loaded_ok) {
        printf("[NN] Modelo restaurado da flash\n");
        } else {
            printf("[NN] Iniciando com pesos aleatorios\n");
        }
    }

    static double target[MATRIX_CELLS];
    static double nn_in[2];
    JoystickState joy = {0};
    bool triggered;
    bool last_save_ok = g_loaded_ok;
    float lr = LEARNING_RATE;

    push_lcd(g_mode, g_train_state, g_last_loss, g_train_count, g_last_acc, last_save_ok);

    for (;;) {

        /* ── Verifica pedido de save (qualquer modo) ─────────────────────── */
        if (g_save_req) {
            g_save_req = false;
            printf("[NN] Salvando na flash...\n");
            last_save_ok = nn_save_to_flash(g_ann,g_train_count);
            matrix_flash_feedback();   /* pisca amarelo para confirmar        */
            push_lcd(g_mode, g_train_state, g_last_loss,
                     g_train_count, g_last_acc, last_save_ok);
        }

        /* ══ MODO TREINO ══════════════════════════════════════════════════ */
        if (g_mode == MODE_TRAINING) {

            g_train_state = TRAIN_SHOW_LINE;
            gen_line_pattern(target);
            memcpy(g_target_pattern, target, sizeof(target));
            matrix_show_pattern(target, 0, 255, 0, 0.1f);
            push_lcd(MODE_TRAINING, TRAIN_SHOW_LINE,
                     g_last_loss, g_train_count, g_last_acc, last_save_ok);
            printf("[NN] Nova linha gerada\n");

            g_train_state = TRAIN_WAIT_CONFIRM;
            push_lcd(MODE_TRAINING, TRAIN_WAIT_CONFIRM,
                     g_last_loss, g_train_count, g_last_acc, last_save_ok);

            triggered = false;
            while (g_mode == MODE_TRAINING && !triggered)
                triggered = (xSemaphoreTake(xTrainTriggerSem, pdMS_TO_TICKS(100)) == pdPASS);
            if (g_mode != MODE_TRAINING) continue;

            if (xQueuePeek(xJoystickQueue, &joy, pdMS_TO_TICKS(50)) != pdPASS)
                { joy.x = 2048; joy.y = 2048; }
            joy_to_nn_input(joy, nn_in);
            
            replay[replay_count % REPLAY_SIZE] = (ReplaySample){{nn_in[0], nn_in[1]}};
            memcpy(replay[replay_count % REPLAY_SIZE].target, target, sizeof(target));
            replay_count++;
            
            printf("[NN] Entrada: x=%.3f y=%.3f\n", nn_in[0], nn_in[1]);

            g_train_state = TRAIN_RUNNING;
            push_lcd(MODE_TRAINING, TRAIN_RUNNING,
                     g_last_loss, g_train_count, g_last_acc, last_save_ok);
            
            int n = (replay_count < REPLAY_SIZE) ? replay_count : REPLAY_SIZE;
            lr = LEARNING_RATE / (1.0f + 0.01f * g_train_count);
            for (int ep = 0; ep < TRAIN_EPOCHS; ep++)
                for (int s = 0; s < n; s++)
                    genann_train(g_ann, replay[s].in, replay[s].target, lr);

            const double *out = genann_run(g_ann, nn_in);
            g_last_loss = compute_mse(out, target, MATRIX_CELLS);
            g_last_acc  = compute_accuracy(out, target, MATRIX_CELLS);
            g_train_count++;

            printf("[NN] #%lu  loss=%.5f  acc=%.0f%%\n",
                   (unsigned long)g_train_count,
                   (double)g_last_loss, (double)(g_last_acc*100.0f));

            g_train_state = TRAIN_SHOW_RESULT;
            matrix_show_inference(out);
            push_lcd(MODE_TRAINING, TRAIN_SHOW_RESULT,
                     g_last_loss, g_train_count, g_last_acc, last_save_ok);

            vTaskDelay(pdMS_TO_TICKS(1000));

        /* ══ MODO INFERÊNCIA ═════════════════════════════════════════════ */
        } else {
            if (xQueuePeek(xJoystickQueue, &joy, pdMS_TO_TICKS(100)) == pdPASS) {
                joy_to_nn_input(joy, nn_in);
                const double *out = genann_run(g_ann, nn_in);
                matrix_show_inference(out);
                push_lcd(MODE_INFERENCE, TRAIN_SHOW_LINE,
                         g_last_loss, g_train_count, g_last_acc, last_save_ok);
            }
            vTaskDelay(pdMS_TO_TICKS(50));
        }
    }
}

// Função em RAM: não acessa flash durante a espera
static void __not_in_flash_func(flash_guard_wait)(void) {
    xSemaphoreGive(xFlashReadySem);           // "estou em RAM, pode gravar"
    while (xSemaphoreTake(xFlashDoneSem, 0) != pdPASS) {
        __wfi();                               // espera em RAM
    }
}

void flashGuardTask(void *params) {
    for (;;) {
        // Bloqueia até Core 0 pedir save — custo zero
        xSemaphoreTake(xFlashStartSem, portMAX_DELAY);
        // Entra no loop em RAM até o save terminar
        flash_guard_wait();
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(void) {
    stdio_init_all();
    sleep_ms(2000);
    printf("[SYS] BitDogLab NN Demo v3\n");

    init_buttons(&ISR_HandleButtons);
    init_rtos_handlers();

    // CORE 0
    xTaskCreateAffinitySet(deviceAliveTask,  "Alive",    512,  NULL,  1, RP2040_CORE_0, NULL);
    xTaskCreateAffinitySet(buttonsTask,      "Buttons",  512,  NULL,  6, RP2040_CORE_0, NULL);
    xTaskCreateAffinitySet(neuralNetTask,    "NeuralNet",4096, NULL,  5, RP2040_CORE_0, NULL);

    // CORE 1
    xTaskCreateAffinitySet(flashGuardTask,   "FlashGrd", 256,  NULL, configMAX_PRIORITIES-1, RP2040_CORE_1, NULL);
    xTaskCreateAffinitySet(readJoystickTask, "Joystick", 1024, NULL, 10, RP2040_CORE_1, NULL);
    xTaskCreateAffinitySet(matrixUpdateTask, "Matrix",   1024, NULL, 10, RP2040_CORE_1, NULL);
    xTaskCreateAffinitySet(lcdUpdateTask,    "LCD",      2048, NULL,  7, RP2040_CORE_1, NULL);

    vTaskStartScheduler();
    while (true) tight_loop_contents();
}