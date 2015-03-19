CC	= nvcc
gen = -gencode arch=compute_30,code=sm_30
flags = -Xcompiler -fPIC -g

all: mdf nscheme

rebuild: clean all

mdf:
	$(CC) $(flags) -shared mdf.cu -lm -g -G -o ../mdf.so -lcuda $(gen)

nscheme:
	$(CC) $(flags) -shared nscheme.cu -lm -g -G -o ../nscheme.so -lcuda $(gen)

clean:
	rm *.o *.so
