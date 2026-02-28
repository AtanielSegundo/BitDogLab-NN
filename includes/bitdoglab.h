#ifndef BITDOGLAB_H

#define BITDOGLAB_H

typedef enum {
    BOARD_RED_CH   = 13,
    BOARD_GREEN_CH = 11,
    BOARD_BLUE_CH  = 12
} BOARD_RGB_LED;

typedef struct{
    uint8_t r;
    uint8_t g;
    uint8_t b;
} Color;

typedef enum {
    BOARD_BUZZER_A = 21,
    BOARD_BUZZER_B = 10
} BOARD_BUZZER;

// Pull Up Buttons
#define BOARD_BUTTON_ON  (0U)
#define BOARD_BUTTON_OFF (0U)
typedef enum {
    BOARD_BUTTON_A = 5,
    BOARD_BUTTON_B = 6,
    BOARD_BUTTON_JOYSTICK = 22 
} BOARD_BUTTON;

// KY023
typedef enum {
    JOYSTICK_VRX = 27,
    JOYSTICK_VRY = 26,
    JOYSTICK_SW  = 22
} BOARD_JOYSTICK;

#define RP2040_CORE_0 (1 << 0)
#define RP2040_CORE_1 (1 << 1)

inline void setup_pin(uint8_t pin,enum gpio_dir dir){
    gpio_init(pin);
    gpio_set_dir(pin,(bool)dir);
    gpio_put(pin, 0);           
}

inline void setup_rgb_led(void){
    setup_pin(BOARD_RED_CH  ,GPIO_OUT);
    setup_pin(BOARD_GREEN_CH,GPIO_OUT);
    setup_pin(BOARD_BLUE_CH ,GPIO_OUT);
}

inline void rgb_led_put(Color c){
    gpio_put(BOARD_RED_CH  , c.r);
    gpio_put(BOARD_GREEN_CH, c.g);
    gpio_put(BOARD_BLUE_CH , c.b);
};

#ifdef BOARD_PWM_HELPERS

#include "hardware/pwm.h"
#include "hardware/clocks.h"

void setup_pwm(uint8_t pin, float divider, uint16_t wrap)
{
    uint slice;
    gpio_set_function(pin, GPIO_FUNC_PWM); 
    slice = pwm_gpio_to_slice_num(pin);    
    pwm_set_clkdiv(slice, divider);    
    pwm_set_wrap(slice, wrap);           
    pwm_set_gpio_level(pin, 0);            
    pwm_set_enabled(slice, false);         
}

void setup_pwm_to_frequency(uint8_t pin, float frequency) {
    uint slice = pwm_gpio_to_slice_num(pin);
    uint32_t f_sys = clock_get_hz(clk_sys);

    // f_pwm = f_sys / (div * (wrap + 1))  =>  div * (wrap + 1) = f_sys / f_pwm
    float total_div = (float)f_sys / frequency;

    float divider = 1.0f;
    uint16_t wrap = 65535;

    if (total_div > 65536.0f) {
        divider = total_div / 65536.0f;
        if (divider > 255.0f) divider = 255.0f;     
        wrap = (uint16_t)(total_div / divider) - 1;
    } else {
        wrap = (uint16_t)total_div - 1;
    }

    // 3. Apply to hardware
    gpio_set_function(pin, GPIO_FUNC_PWM);
    pwm_set_clkdiv(slice, divider);
    pwm_set_wrap(slice, wrap);
    pwm_set_gpio_level(pin, 0);
    pwm_set_enabled(slice, true);

}

void set_pwm_duty(uint8_t pin, float duty) {
    if (duty < 0.0f) duty = 0.0f;
    if (duty > 100.0f) duty = 100.0f;
    
    uint slice = pwm_gpio_to_slice_num(pin);
    uint16_t wrap  = pwm_hw->slice[slice].top;
    uint32_t level = (uint32_t)((duty / 100.0f) * (wrap + 1));
    
    pwm_set_gpio_level(pin, (uint16_t)level);

    if (!pwm_hw->slice[slice].csr & PWM_CH0_CSR_EN_BITS) {
        pwm_set_enabled(slice, true);
    }
}

void setup_rgb_led_with_bright(void) {
    const float LED_FREQ = 1000.0f; 
    setup_pwm_to_frequency(BOARD_RED_CH,   LED_FREQ);
    setup_pwm_to_frequency(BOARD_GREEN_CH, LED_FREQ);
    setup_pwm_to_frequency(BOARD_BLUE_CH,  LED_FREQ);
}

// Controls color and global brightness
// c: Color struct (0-255 per channel)
// brightness: 0.0f to 100.0f
void rgb_led_put_with_bright(Color c, float brightness) {
    if (brightness < 0.0f) brightness = 0.0f;
    if (brightness > 100.0f) brightness = 100.0f;

    // Convert 0-255 color range to 0.0-1.0 ratio
    float r_ratio = (float)c.r / 255.0f;
    float g_ratio = (float)c.g / 255.0f;
    float b_ratio = (float)c.b / 255.0f;

    // Multiply channel ratio by global brightness to get final duty cycle
    set_pwm_duty(BOARD_RED_CH,   r_ratio * brightness);
    set_pwm_duty(BOARD_GREEN_CH, g_ratio * brightness);
    set_pwm_duty(BOARD_BLUE_CH,  b_ratio * brightness);
}

#ifdef BOARD_BUTTON_HELPERS

inline void setup_button(uint8_t pin) {
    gpio_init(pin);
    gpio_set_dir(pin, GPIO_IN);
    gpio_pull_up(pin);
}

inline void setup_button_with_isr(uint8_t pin, gpio_irq_callback_t handler) {
    setup_button(pin);
    gpio_set_irq_enabled_with_callback(pin, GPIO_IRQ_EDGE_FALL, true, handler);
}

inline bool button_is_pressed(uint8_t pin) {
    return gpio_get(pin) == BOARD_BUTTON_ON;
}

inline bool button_is_pressed_poll(uint8_t pin, uint32_t debounce_ms) {
    if (button_is_pressed(pin)) {
        sleep_ms(debounce_ms);
        return button_is_pressed(pin);
    }
    return false;
}


/*

Example 01: Button Usage
    void botao_callback(uint gpio, uint32_t events) {
        if (gpio == BOARD_BUTTON_A) {
            // Lógica para o Botão A
        } else if (gpio == BOARD_BUTTON_B) {
            // Lógica para o Botão B
        }
    }

    int main() {
        stdio_init_all();
        setup_button_with_isr(BOARD_BUTTON_A,&botao_callback)
        setup_button_with_isr(BOARD_BUTTON_B,&botao_callback)
        while(1) {
            tight_loop_contents();
        }
    }

*/

#endif

#ifdef BOARD_ADC_HELPERS

#endif

#endif

#endif