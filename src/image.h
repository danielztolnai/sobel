#ifndef IMAGE_H
#define IMAGE_H

#define BIT_DEPTH 8
#define COLOR_TYPE PNG_COLOR_TYPE_RGB

typedef struct t_pixel {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} t_pixel;

typedef struct t_pixel_f {
    float r;
    float g;
    float b;
} t_pixel_f;

typedef struct t_image {
    int width;
    int height;
    t_pixel** data;
} t_image;

void mallocImage(t_image* image);
void freeImage(t_image image);
int readPngFile(char* fileName, t_image* image);
int writePngFile(char* fileName, t_image image);

#endif /* IMAGE_H */
