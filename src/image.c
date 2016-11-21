#include "image.h"
#include <stdlib.h>
#include <png.h>

void mallocImage(t_image* image)
{
    image->data = (t_pixel**) malloc(sizeof(t_pixel*) * image->height);
    for (int y=0; y<image->height; y++) {
        image->data[y] = (t_pixel*) malloc(sizeof(t_pixel) * image->width);
    }
}

void freeImage(t_image image)
{
    for (int y=0; y<image.height; y++) {
        if(image.data[y] != NULL) {
            free(image.data[y]);
            image.data[y] = NULL;
        }
    }
    if(image.data != NULL) {
        free(image.data);
        image.data = NULL;
    }
}

int readPngFile(char* fileName, t_image* image)
{
    /* open file and test for it being a png */
    FILE *fp = fopen(fileName, "rb");
    if (!fp) {
        printf("[readPngFile] File %s could not be opened for reading\n", fileName);
        return 1;
    }

    unsigned char header[8]; // 8 is the maximum size that can be checked
    if(fread(header, 1, 8, fp) != 8) {
        printf("[readPngFile] File %s is not recognized as a PNG file\n", fileName);
        fclose(fp);
        return 1;
    }
    if (png_sig_cmp(header, 0, 8)) {
        printf("[readPngFile] File %s is not recognized as a PNG file\n", fileName);
        fclose(fp);
        return 1;
    }

    /* initialize stuff */
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        printf("[readPngFile] png_create_read_struct failed\n");
        fclose(fp);
        return 1;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        printf("[readPngFile] png_create_info_struct failed\n");
        fclose(fp);
        return 1;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        printf("[readPngFile] Error during init_io\n");
        fclose(fp);
        return 1;
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    image->width  = png_get_image_width(png_ptr, info_ptr);
    image->height = png_get_image_height(png_ptr, info_ptr);
    if(png_get_color_type(png_ptr, info_ptr) != COLOR_TYPE) {
        printf("[readPngFile] only RGB color type is supported\n");
        fclose(fp);
        return 1;
    }

    if(png_get_bit_depth(png_ptr, info_ptr) != BIT_DEPTH) {
        printf("[readPngFile] bit depth different from 8 is not supported\n");
        fclose(fp);
        return 1;
    }

    if(png_set_interlace_handling(png_ptr) > 1) {
        printf("[readPngFile] interlaced images are not supported\n");
        fclose(fp);
        return 1;
    }
    png_read_update_info(png_ptr, info_ptr);

    /* read file */
    if (setjmp(png_jmpbuf(png_ptr))) {
        printf("[readPngFile] Error during read_image\n");
        fclose(fp);
        return 1;
    }

    mallocImage(image);

    png_read_image(png_ptr, (png_bytepp)image->data);

    fclose(fp);
    return 0;
}

int writePngFile(char* fileName, t_image image)
{
    /* create file */
    FILE *fp = fopen(fileName, "wb");
    if (!fp) {
        printf("[writePngFile] File %s could not be opened for writing\n", fileName);
        return 1;
    }

    /* initialize stuff */
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr) {
        printf("[writePngFile] png_create_write_struct failed\n");
        fclose(fp);
        return 1;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        printf("[writePngFile] png_create_info_struct failed\n");
        fclose(fp);
        return 1;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        printf("[writePngFile] Error during init_io\n");
        fclose(fp);
        return 1;
    }

    png_init_io(png_ptr, fp);

    /* write header */
    if (setjmp(png_jmpbuf(png_ptr))) {
        printf("[writePngFile] Error during writing header\n");
        fclose(fp);
        return 1;
    }

    png_set_IHDR(png_ptr, info_ptr, image.width, image.height,
                 BIT_DEPTH, COLOR_TYPE, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    /* write bytes */
    if (setjmp(png_jmpbuf(png_ptr))) {
        printf("[writePngFile] Error during writing bytes\n");
        fclose(fp);
        return 1;
    }

    png_write_image(png_ptr, (png_bytepp)image.data);

    /* end write */
    if (setjmp(png_jmpbuf(png_ptr))) {
        printf("[writePngFile] Error during end of write\n");
        fclose(fp);
        return 1;
    }

    png_write_end(png_ptr, NULL);

    fclose(fp);
    return 0;
}
