#ifndef PRJ_RTOS_H
    #define PRJ_RTOS_H

    #include "FreeRTOS.h"
    #include "task.h"
    #include "queue.h"
    #include "semphr.h"
    #include "pico/stdlib.h"

    #define BOARD_BUTTON_HELPERS
    #define BOARD_ADC_HELPERS
    #include "bitdoglab.h"

    #define BUTTONS_QUEUE_SIZE  (1U)
    QueueHandle_t xButtonsQueue;

    #define JOYSTICK_QUEUE_SIZE (1U)
    QueueHandle_t xJoystickQueue;

    SemaphoreHandle_t xMatrixMutex;

    SemaphoreHandle_t xTrainTriggerSem;
    SemaphoreHandle_t xStateMutex;

    typedef enum { 
        MODE_TRAINING = 0, 
        MODE_INFERENCE 
    } AppMode;
    typedef enum {
        TRAIN_SHOW_LINE    = 0,
        TRAIN_WAIT_CONFIRM,
        TRAIN_RUNNING,
        TRAIN_SHOW_RESULT,
    } TrainingState;

    typedef struct {
        AppMode       mode;
        TrainingState train_state;
        float         loss;
        uint32_t      train_count;
        float         accuracy;
        bool          flash_status; /* true = último save ok / load ok             */
    } LCDInfo_t;

    QueueHandle_t xLCDQueue;

    static SemaphoreHandle_t xFlashStartSem;  // Core 0 → Core 1: "para agora"
    static SemaphoreHandle_t xFlashReadySem;  // Core 1 → Core 0: "pode gravar"
    static SemaphoreHandle_t xFlashDoneSem;   // Core 0 → Core 1: "terminei"

    void init_buttons(gpio_irq_callback_t btn_handler){
        setup_button_with_isr(BOARD_BUTTON_A,btn_handler);
        setup_button_with_isr(BOARD_BUTTON_B,btn_handler);
        setup_button_with_isr(BOARD_BUTTON_JOYSTICK,btn_handler);
    }
    
    void init_rtos_handlers(void){
        xButtonsQueue  = xQueueCreate(BUTTONS_QUEUE_SIZE,  sizeof(BOARD_BUTTON));
        xJoystickQueue = xQueueCreate(JOYSTICK_QUEUE_SIZE, sizeof(JoystickState));
        xTrainTriggerSem = xSemaphoreCreateBinary();
        xStateMutex      = xSemaphoreCreateMutex();
        xLCDQueue        = xQueueCreate(1, sizeof(LCDInfo_t));
        xMatrixMutex   = xSemaphoreCreateMutex();
        if (xMatrixMutex != NULL) {
            xSemaphoreTake(xMatrixMutex, 0); 
        }

        xFlashStartSem = xSemaphoreCreateBinary();
        xFlashReadySem = xSemaphoreCreateBinary();
        xFlashDoneSem  = xSemaphoreCreateBinary();
    }
    
    

#endif