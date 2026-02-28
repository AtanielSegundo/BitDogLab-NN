#include "custom_fgets.h"
#include <stdio.h>

char* custom_fgets(char* str, int n, FILE* stream) {
    int i = 0;
    while (i < n - 1) {
        int ch = getchar();

        if (ch == '\n' || ch == '\r') {
            str[i] = '\0';
            return str;
        } else if (ch == '\b' || ch == 0x7F) {
            if (i > 0) {
                i--;
                printf("\b \b");
            }
        } else if (ch >= 32 && ch <= 126) {
            str[i++] = (char)ch;
            printf("%c", ch);
        }
        // Non-printable characters are ignored
    }
    str[i] = '\0';
    return str;
}
