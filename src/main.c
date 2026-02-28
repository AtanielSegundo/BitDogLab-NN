#include <stdio.h>
#include "pico/stdlib.h"

#define BOARD_PWM_HELPERS 
#define BOARD_BUTTON_HELPERS
#include "bitdoglab.h"

#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "semphr.h"



void deviceAliveTask(void* params){
    setup_rgb_led_with_bright();
    bool blink_state = 0;
    const Color blink_color  = {255,255,255};
    const float blink_bright = 60.0f;
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

int main()
{
    stdio_init_all();

    printf("Hello, world!\n");
    
    xTaskCreateAffinitySet(deviceAliveTask,
                           "Device Alive Task",
                           512 ,
                           NULL,
                           1,
                           RP2040_CORE_0,NULL);
    
    vTaskStartScheduler();

    while (true) {
        tight_loop_contents();
    }

}
