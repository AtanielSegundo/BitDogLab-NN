#include <stdio.h>
#include "pico/stdlib.h"

#define BOARD_PWM_HELPERS 
#define BOARD_BUTTON_HELPERS
#define BOARD_ADC_HELPERS
#include "bitdoglab.h"

#define BDLED_IMPLEMENTATION
#include "bdl_led_matrix.h"

#include "rtos.h"


//----- Tasks Functions ------------------------------------------------------------

void deviceAliveTask(void* params){
    setup_rgb_led_with_bright();
    bool blink_state = 0;
    const Color blink_color  = {255,255,255};
    const float blink_bright = 10.0f;
    for (;;)
    {   
        if (blink_state) {
            rgb_led_put_with_bright(blink_color,blink_bright);
        } else{
            rgb_led_put_with_bright(blink_color,0.0);
        }

        blink_state = !blink_state;
        vTaskDelay(pdMS_TO_TICKS(500));    
    }
}

const uint32_t DEBOUNCE_MS = 100; 
static uint32_t last_interrupt_time = 0;

void ISR_HandleButtons(uint gpio, uint32_t event_mask) {
    uint32_t current_time = to_ms_since_boot(get_absolute_time());
    BaseType_t hpTaskWoken = pdFALSE;

    if (current_time - last_interrupt_time > DEBOUNCE_MS) {
        if (gpio == BOARD_BUTTON_A || gpio == BOARD_BUTTON_B || gpio == BOARD_BUTTON_JOYSTICK) {
            xQueueOverwriteFromISR(xButtonsQueue, &gpio, &hpTaskWoken);
            last_interrupt_time = current_time;
        }
    }
    
    portYIELD_FROM_ISR(hpTaskWoken);
}

void buttonsTask(void* args){
    BOARD_BUTTON local_btn = {0};
    
    BaseType_t result = pdFALSE;
    for(;;){
        result = xQueueReceive(xButtonsQueue,&local_btn,portMAX_DELAY);
        if (result == pdPASS && gpio_get(local_btn) == BOARD_BUTTON_ON){
            printf("[INFO] Button Pressed: %s\n", board_button_str[local_btn]);
        }
    }
}

void readJoystickTask(void* params) {
    setup_joystick();
    const uint16_t DEADZONE = 200;
    // CUT FREQUENCY = (alpha * fs) / (1  - alpha)
    const float EMA_ALPHA = 0.4; 
    JoystickState local_state = {0};

    for (;;) {
        local_state = read_joystick();

        local_state = apply_low_pass_filter(local_state,&EMA_ALPHA);
        
        int x_clean = apply_joystick_deadzone(local_state.x, DEADZONE);
        int y_clean = apply_joystick_deadzone(local_state.y, DEADZONE);
        
        local_state.x = (x_clean == 0) ? 2048 : local_state.x;
        local_state.y = (y_clean == 0) ? 2048 : local_state.y;

        xQueueOverwrite(xJoystickQueue,&local_state);
        
        vTaskDelay(pdMS_TO_TICKS(20)); 
    }
}

void printJoystickTask(void* params) {
    JoystickState local_state = {0};
    BaseType_t res = pdFAIL;
    float norm_x = 0.0f;
    float norm_y = 0.0f; 
    for (;;) {
        res = xQueueReceive(xJoystickQueue,&local_state,portMAX_DELAY);
        if (res == pdPASS){
            norm_x = map_joystick_value(local_state.x,-1.0,1.0);
            norm_y = map_joystick_value(local_state.y,-1.0,1.0);
            printf("[INFO] X = %.2f , Y = %.2f \n",norm_x,norm_y);
        }
        vTaskDelay(pdMS_TO_TICKS(500)); 
    }
}

void matrixTask(void* params){
    LedMatrix matrix = {0}; 
    bdl_matrixInit(&matrix, LED_PIN, 5, 5);
    for(;;){
        
        bdl_matrixClear(&matrix);
        bdl_matrixDrawLine(&matrix,0,0,4,4,255,255,0,0.6);
        
        bdl_matrixWrite(&matrix);
        vTaskDelay(pdMS_TO_TICKS(33));
    }
}

//----------------------------------------------------------------------------------

int main()
{
    stdio_init_all();
    
    sleep_ms(1000);
    
        printf("[INFO] System Booting!\n");
    
    sleep_ms(1000);

    init_buttons(&ISR_HandleButtons);
    init_rtos_handlers();

    xTaskCreateAffinitySet(deviceAliveTask,
                           "Device Alive Task",
                           512 ,
                           NULL,
                           1,
                           RP2040_CORE_0,
                           NULL);

    xTaskCreateAffinitySet(buttonsTask,
                           "Buttons Task",
                           512,
                           NULL,
                           5,
                           RP2040_CORE_0,
                           NULL);
    
    xTaskCreateAffinitySet(readJoystickTask,
                           "Read Joystick Task",
                           1024,
                           NULL,
                           10,
                           RP2040_CORE_0,
                           NULL);

    xTaskCreateAffinitySet(matrixTask,
                           "Matrix Update Task",
                           1024,
                           NULL,
                           10,
                           RP2040_CORE_0,
                           NULL);
    
    xTaskCreateAffinitySet(printJoystickTask,
                           "Print Joystick Task",
                           1024,
                           NULL,
                           5,
                           RP2040_CORE_0,
                           NULL);
                               

    vTaskStartScheduler();

    while (true) {
        tight_loop_contents();
    }

}
