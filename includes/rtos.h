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


    void init_buttons(gpio_irq_callback_t btn_handler){
        setup_button_with_isr(BOARD_BUTTON_A,btn_handler);
        setup_button_with_isr(BOARD_BUTTON_B,btn_handler);
        setup_button_with_isr(BOARD_BUTTON_JOYSTICK,btn_handler);
    }
    
    void init_rtos_handlers(void){
        xButtonsQueue  = xQueueCreate(BUTTONS_QUEUE_SIZE,  sizeof(BOARD_BUTTON));
        xJoystickQueue = xQueueCreate(JOYSTICK_QUEUE_SIZE, sizeof(JoystickState));
    }
    
#endif