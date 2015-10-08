#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bmpReader.h"

#define numColor 3
#define N 8


// gpu function call wrapper 
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// go through this - http://cuda-programming.blogspot.in/2013/01/handling-cuda-error-messages.html

__global__ void dct(unsigned char *channel, float *dct, float *idct, float *quantizationTable, int numRowsint, int numCols, float *DCTv8matrix, float *DCTv8matrixT )
{
        const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                                blockIdx.y * blockDim.y + threadIdx.y);

        const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

        
        int i = 0;
        __shared__ float ATX[N][N];

        __shared__ float sQuantizationTable[N][N+1]; // avoids bank conflict
        __shared__ float sDCTv8matrix[N][N+1];
        __shared__ float sDCTv8matrixT[N][N+1];
        __shared__ float channelData[N][N+1];
        __shared__ float Sdct[N][N+1];

        float temp = 0;
        //for (i = 0; i < N*N; ++i)
        //{
        ATX[threadIdx.y][threadIdx.x] = 0;
        Sdct[threadIdx.y][threadIdx.x] = 0;

        sQuantizationTable[threadIdx.y][threadIdx.x] = quantizationTable[threadIdx.y * blockDim.x + threadIdx.x];
        sDCTv8matrix[threadIdx.y][threadIdx.x] = DCTv8matrix[threadIdx.y * blockDim.x + threadIdx.x];
        sDCTv8matrixT[threadIdx.y][threadIdx.x] = DCTv8matrixT[threadIdx.y * blockDim.x + threadIdx.x];
        channelData[threadIdx.y][threadIdx.x] = channel[thread_1D_pos];
        
        //}

        __syncthreads();

        for (i = 0; i < N; ++i) // looping over shared mem is a prob - bank conflict ?
        {           
               temp += sDCTv8matrix[i][threadIdx.y] * channelData[i][threadIdx.x];
        }      
         ATX[threadIdx.x][threadIdx.y] = temp;
         __syncthreads(); 


         temp = 0;
        for (i = 0; i < N; ++i) 
        {
                
               temp  += ATX[i][threadIdx.y] * sDCTv8matrix[i][threadIdx.x];

    
        }
        Sdct[threadIdx.y][threadIdx.x] = temp * sQuantizationTable[threadIdx.y][threadIdx.x];;
        ATX[threadIdx.y][threadIdx.x] = 0;
        temp = 0;
       __syncthreads();



        for (i = 0; i < N; ++i) // looping over shared mem is a prob - bank conflict ?
        {
                temp += sDCTv8matrixT[i][threadIdx.y] * Sdct[i][threadIdx.x];
        }       
        ATX[threadIdx.x][threadIdx.y] = temp;
        temp = 0;   
        __syncthreads();

        for (i = 0; i < N; ++i) // looping over shared mem is a prob - bank conflict ?
        {
                 temp += ATX[i][threadIdx.y] * sDCTv8matrixT[i][threadIdx.x];
                 //idct[thread_1D_pos] += ATX[threadIdx.y * blockDim.x + i] * sDCTv8matrix[threadIdx.y * blockDim.x + i];
        }
        idct[thread_1D_pos] = temp;

}


int main(int argc, char *argv[])
{


        BMPData *image1 = NULL;
        FILE *fp;
        int i = 0,j = 0;
        int size = 0;
        unsigned char *red = NULL,*green = NULL,*blue = NULL;
        unsigned char *d_red = NULL,*d_green = NULL,*d_blue = NULL;

        float *d_red_dct = NULL;
        float *d_blue_dct = NULL;
        float *d_green_dct = NULL;

        float *d_red_idct = NULL;
        float *d_blue_idct = NULL;
        float *d_green_idct = NULL;

        float *red_dct = NULL;
        float *blue_dct = NULL;
        float *green_dct = NULL;


        float *red_idct = NULL;
        float *blue_idct = NULL;
        float *green_idct = NULL;

        float *d_quantizationTable = NULL;


        float *d_DCTv8matrix = NULL;
        float *d_DCTv8matrixT = NULL;

        float time;

        cudaEvent_t start, stop;

        const int value = 0;

        const float DCTv8matrix[N*N] = {
        0.3535533905932738f,  0.4903926402016152f,  0.4619397662556434f,  0.4157348061512726f,  0.3535533905932738f,  0.2777851165098011f,  0.1913417161825449f,  0.0975451610080642f, 
        0.3535533905932738f,  0.4157348061512726f,  0.1913417161825449f, -0.0975451610080641f, -0.3535533905932737f, -0.4903926402016152f, -0.4619397662556434f, -0.2777851165098011f, 
        0.3535533905932738f,  0.2777851165098011f, -0.1913417161825449f, -0.4903926402016152f, -0.3535533905932738f,  0.0975451610080642f,  0.4619397662556433f,  0.4157348061512727f, 
        0.3535533905932738f,  0.0975451610080642f, -0.4619397662556434f, -0.2777851165098011f,  0.3535533905932737f,  0.4157348061512727f, -0.1913417161825450f, -0.4903926402016153f, 
        0.3535533905932738f, -0.0975451610080641f, -0.4619397662556434f,  0.2777851165098009f,  0.3535533905932738f, -0.4157348061512726f, -0.1913417161825453f,  0.4903926402016152f, 
        0.3535533905932738f, -0.2777851165098010f, -0.1913417161825452f,  0.4903926402016153f, -0.3535533905932733f, -0.0975451610080649f,  0.4619397662556437f, -0.4157348061512720f, 
        0.3535533905932738f, -0.4157348061512727f,  0.1913417161825450f,  0.0975451610080640f, -0.3535533905932736f,  0.4903926402016152f, -0.4619397662556435f,  0.2777851165098022f, 
        0.3535533905932738f, -0.4903926402016152f,  0.4619397662556433f, -0.4157348061512721f,  0.3535533905932733f, -0.2777851165098008f,  0.1913417161825431f, -0.0975451610080625f
        };

        const float DCTv8matrixT[N*N] = {
        0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f, 
        0.4903926402016152f,  0.4157348061512726f,  0.2777851165098011f,  0.0975451610080642f, -0.0975451610080641f, -0.2777851165098010f, -0.4157348061512727f, -0.4903926402016152f, 
        0.4619397662556434f,  0.1913417161825449f, -0.1913417161825449f, -0.4619397662556434f, -0.4619397662556434f, -0.1913417161825452f,  0.1913417161825450f,  0.4619397662556433f, 
        0.4157348061512726f, -0.0975451610080641f, -0.4903926402016152f, -0.2777851165098011f,  0.2777851165098009f,  0.4903926402016153f,  0.0975451610080640f, -0.4157348061512721f, 
        0.3535533905932738f, -0.3535533905932737f, -0.3535533905932738f,  0.3535533905932737f,  0.3535533905932738f, -0.3535533905932733f, -0.3535533905932736f,  0.3535533905932733f, 
        0.2777851165098011f, -0.4903926402016152f,  0.0975451610080642f,  0.4157348061512727f, -0.4157348061512726f, -0.0975451610080649f,  0.4903926402016152f, -0.2777851165098008f, 
        0.1913417161825449f, -0.4619397662556434f,  0.4619397662556433f, -0.1913417161825450f, -0.1913417161825453f,  0.4619397662556437f, -0.4619397662556435f,  0.1913417161825431f, 
        0.0975451610080642f, -0.2777851165098011f,  0.4157348061512727f, -0.4903926402016153f,  0.4903926402016152f, -0.4157348061512720f,  0.2777851165098022f, -0.0975451610080625f
        };

        const float quantizationTable[N*N] = 
        { 
        0.500000,   1.000000,   1.000000,   1.000000,   1.000000,   1.000000,   0.500000,   1.000000,
        1.000000,   1.000000,   0.500000,   0.500000,   0.500000,   0.500000,   0.500000,   0.250000,
        0.333333,   0.500000,   0.500000,   0.500000,   0.500000,   0.200000,   0.250000,   0.250000,
        0.333333,   0.250000,   0.166667,   0.200000,   0.166667,   0.166667,   0.166667,   0.200000,
        0.166667,   0.166667,   0.166667,   0.142857,   0.111111,   0.125000,   0.166667,   0.142857,
        0.111111,   0.142857,   0.166667,   0.166667,   0.125000,   0.090909,   0.125000,   0.111111,
        0.100000,   0.100000,   0.100000,   0.100000,   0.100000,   0.166667,   0.125000,   0.090909,
        0.083333,   0.090909,   0.100000,   0.083333,   0.111111,   0.100000,   0.100000,   0.100000
        };

        const float quantizationTableUV[N*N] =       
       {
           0.058824,   0.055556,   0.041667,   0.021277,   0.010101,   0.010101,   0.010101,   0.010101,
           0.055556,   0.047619,   0.038462,   0.015152,   0.010101,   0.010101,   0.010101,   0.010101,
           0.041667,   0.038462,   0.017857,   0.010101,   0.010101,   0.010101,   0.010101,   0.010101,
           0.021277,   0.015152,   0.010101,   0.010101,   0.010101,   0.010101,   0.010101,   0.010101,
           0.010101,   0.010101,   0.010101,   0.010101,   0.010101,   0.010101,   0.010101,   0.010101,
           0.010101,   0.010101,   0.010101,   0.010101,   0.010101,   0.010101,   0.010101,   0.010101,
           0.010101,   0.010101,   0.010101,   0.010101,   0.010101,   0.010101,   0.010101,   0.010101,
           0.010101,   0.010101,   0.010101,   0.010101,   0.010101,   0.010101,   0.010101,   0.010101
        }; // add this to UV channel 

        /*
            use YUV color space

        */
        cudaStream_t stream1,stream2,stream3;

        //unsigned char  test[64] = {48,39,40,68,60,38,50,121,149,82,79,101,113,106,27,62,58,63,77,69,124,107,74,125,80,97,74,54,59,71,91,66,18,34,33,46,64,61,32,37,149,108,80,106,116,61,73,92,211,233,159,88,107,158,161,109,212,104,40,44,71,136,113,66};



        // increase occupancy - use bigger images 


        gpuErrchk(cudaStreamCreate(&stream1) );
        gpuErrchk(cudaStreamCreate(&stream2) );
        gpuErrchk(cudaStreamCreate(&stream3) );

        if((image1 = readBMPfile(argv[1])) == NULL)
        {
                printf("Error in File 1\n");
                return -1;
        }

        size = image1->infoHeader.height * image1->infoHeader.width;
        //size = 64;
        dim3 grid(image1->infoHeader.width/N,image1->infoHeader.height/N);            // defines a grid of 256 x 1 x 1 blocks
        dim3 block(N,N); 
        red = (unsigned char *) malloc(size * sizeof(char) );
        blue = (unsigned char *) malloc(size * sizeof(char) );
        green = (unsigned char *) malloc(size * sizeof(char) );

        red_dct = (float *) malloc(size * sizeof(float) );
        blue_dct = (float *) malloc(size * sizeof(float) );
        green_dct = (float *) malloc(size * sizeof(float) );

        red_idct = (float *) malloc(size * sizeof(float) );
        blue_idct = (float *) malloc(size * sizeof(float) );
        green_idct = (float *) malloc(size * sizeof(float) );



        for (i = 0,j = 0; i < size * numColor; i+=3,++j)
        {
                        blue[j] = image1->bitMapImage[i]; 
                        green[j] = image1->bitMapImage[i + 1];
                        red[j] = image1->bitMapImage[i + 2];
        }
        gpuErrchk( cudaEventCreate(&start) );
        gpuErrchk( cudaEventCreate(&stop) );
        gpuErrchk( cudaEventRecord(start, 0) );

        gpuErrchk( cudaMalloc( (void**)&d_red, size * sizeof(char) ));
        gpuErrchk( cudaMalloc( (void**)&d_green, size * sizeof(char) ));
        gpuErrchk( cudaMalloc( (void**)&d_blue, size * sizeof(char) ));


        gpuErrchk( cudaMalloc( (void**)&d_red_dct, size * sizeof(float) ));
        gpuErrchk( cudaMalloc( (void**)&d_blue_dct, size * sizeof(float) ));
        gpuErrchk( cudaMalloc( (void**)&d_green_dct, size * sizeof(float) ));
        
        gpuErrchk(cudaMemset(d_red_dct,value,size * sizeof(float)));
        gpuErrchk(cudaMemset(d_green_dct,value,size * sizeof(float)));
        gpuErrchk(cudaMemset(d_blue_dct,value,size * sizeof(float)));

        
        gpuErrchk( cudaMalloc( (void**)&d_red_idct, size * sizeof(float) ));
        gpuErrchk( cudaMalloc( (void**)&d_blue_idct, size * sizeof(float) ));
        gpuErrchk( cudaMalloc( (void**)&d_green_idct, size * sizeof(float) ));    

        gpuErrchk(cudaMemset(d_red_idct,value,size * sizeof(float)));
        gpuErrchk(cudaMemset(d_green_idct,value,size * sizeof(float)));
        gpuErrchk(cudaMemset(d_blue_idct,value,size * sizeof(float)));            

        gpuErrchk( cudaMalloc( (void**)&d_DCTv8matrix, N * N * sizeof(float) ));
        gpuErrchk( cudaMalloc( (void**)&d_DCTv8matrixT, N * N * sizeof(float) ));
        gpuErrchk( cudaMalloc( (void**)&d_quantizationTable, N * N * sizeof(float) ));


        gpuErrchk(cudaMemcpy( d_quantizationTable, &quantizationTable, N * N * sizeof(float), cudaMemcpyHostToDevice ));
        gpuErrchk(cudaMemcpy( d_DCTv8matrix, &DCTv8matrix, N * N * sizeof(float), cudaMemcpyHostToDevice ));
        gpuErrchk(cudaMemcpy( d_DCTv8matrixT, &DCTv8matrixT, N * N * sizeof(float), cudaMemcpyHostToDevice ));

        
        
        



        gpuErrchk(cudaMemcpyAsync( d_red, red, size * sizeof(char) , cudaMemcpyHostToDevice, stream1 ));
        dct<<<  grid, block, 0, stream1 >>>(d_red, d_red_dct, d_red_idct,d_quantizationTable, image1->infoHeader.height,image1->infoHeader.width, d_DCTv8matrix, d_DCTv8matrixT);
        gpuErrchk(cudaMemcpyAsync( d_green, green, size * sizeof(char), cudaMemcpyHostToDevice, stream2 ));
        dct<<<  grid, block, 0, stream2  >>>(d_green, d_green_dct, d_green_idct,d_quantizationTable, image1->infoHeader.height,image1->infoHeader.width, d_DCTv8matrix, d_DCTv8matrixT);
        gpuErrchk(cudaMemcpyAsync( d_blue, blue, size * sizeof(char), cudaMemcpyHostToDevice, stream3 ));
        dct<<<  grid, block, 0, stream3  >>>(d_blue, d_blue_dct, d_blue_idct,d_quantizationTable, image1->infoHeader.height,image1->infoHeader.width, d_DCTv8matrix, d_DCTv8matrixT);



        //gpuErrchk(cudaMemcpy(red_dct, d_red_dct , size * sizeof(float), cudaMemcpyDeviceToHost ));
        gpuErrchk(cudaMemcpyAsync(red_idct, d_red_idct, size * sizeof(float), cudaMemcpyDeviceToHost, stream1 ));

        //gpuErrchk(cudaMemcpy(green_dct, d_green_dct , size * sizeof(float), cudaMemcpyDeviceToHost ));
        gpuErrchk(cudaMemcpyAsync(green_idct, d_green_idct, size * sizeof(float), cudaMemcpyDeviceToHost, stream2 ));

        //gpuErrchk(cudaMemcpy(blue_dct, d_blue_dct , size * sizeof(float), cudaMemcpyDeviceToHost ));
        gpuErrchk(cudaMemcpyAsync(blue_idct, d_blue_idct, size * sizeof(float), cudaMemcpyDeviceToHost,stream3));
        gpuErrchk( cudaDeviceSynchronize());

        gpuErrchk( cudaEventRecord(stop, 0) );
        gpuErrchk( cudaEventSynchronize(stop) );
        gpuErrchk( cudaEventElapsedTime(&time, start, stop) );       


        for (i = 0,j = 0; j < size; i+=3,++j)
        {

               // printf("%f %f %f\n",blue_idct[i],red_idct[i],green_idct[i]);
                if(blue_idct[j] > 255)
                    image1->bitMapImage[i] = 255;
                else if(blue_idct[j] < 0)
                    image1->bitMapImage[i] = 0;
                else
                    image1->bitMapImage[i] = (unsigned char)blue_idct[j];

                if(green_idct[j] > 255)
                    image1->bitMapImage[i + 1] = 255;
                else if(green_idct[j] < 0)
                    image1->bitMapImage[i + 1] = 0;
                else
                    image1->bitMapImage[i + 1] = (unsigned char) green_idct[j]; 

                if(red_idct[j] > 255)
                    image1->bitMapImage[i + 2] = 255;
                else if(red_idct[j] < 0)
                    image1->bitMapImage[i + 2] = 0;
                else
                    image1->bitMapImage[i + 2] = (unsigned char) red_idct[j]; 
        }
        if(writeBMPfile(image1,"dctCompress.bmp") != 1)
        {
            printf("Error Writing File\n");
            gpuErrchk(cudaFree( d_red ));
            gpuErrchk(cudaFree( d_green ));
            gpuErrchk(cudaFree( d_blue ));
            gpuErrchk(cudaFree( d_red_dct ));
            gpuErrchk(cudaFree( d_blue_dct ));
            gpuErrchk(cudaFree( d_green_dct ));
            gpuErrchk(cudaFree( d_red_idct ));
            gpuErrchk(cudaFree( d_blue_idct ));
            gpuErrchk(cudaFree( d_green_idct ));
            gpuErrchk(cudaFree( d_DCTv8matrix ));
            gpuErrchk(cudaFree( d_DCTv8matrixT ));
            gpuErrchk(cudaFree( d_quantizationTable ));
            return -1;
        }
        gpuErrchk(cudaFree( d_red ));
        gpuErrchk(cudaFree( d_green ));
        gpuErrchk(cudaFree( d_blue ));
        gpuErrchk(cudaFree( d_red_dct ));
        gpuErrchk(cudaFree( d_blue_dct ));
        gpuErrchk(cudaFree( d_green_dct ));
        gpuErrchk(cudaFree( d_red_idct ));
        gpuErrchk(cudaFree( d_blue_idct ));
        gpuErrchk(cudaFree( d_green_idct ));
        gpuErrchk(cudaFree( d_DCTv8matrix ));
        gpuErrchk(cudaFree( d_DCTv8matrixT ));
        gpuErrchk(cudaFree( d_quantizationTable ));

        gpuErrchk(cudaStreamDestroy(stream1) );
        gpuErrchk(cudaStreamDestroy(stream2) );
        gpuErrchk(cudaStreamDestroy(stream3) );

        printf("Time to generate:  %3.1f ms \n", time);
        if((fp = fopen("timingInformation","a+")) == NULL)
        {
            printf("File opening has failed\n");
            return -1;
        }
        fprintf(fp, "Image Size: %d * %d Time:%f\n",image1->infoHeader.height,image1->infoHeader.width,time);
        fclose(fp);
        gpuErrchk(cudaDeviceReset());
        return 0;



}
