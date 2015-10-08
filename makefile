all:
	nvcc -g -G dct.cu bmpReader.cu -o run -lm -Xptxas -v 
 
