CC	= nvcc
gen = -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50
flags = -Xcompiler -fPIC -g -G -O0

all: nscheme escheme

rebuild: clean all

mdf:
	$(CC) $(flags) -shared mdf.cu -lm -o ../mdf.so -lcuda $(gen)

nscheme:
	$(CC) $(flags) -shared nscheme.cu -lm -o ../nscheme.so -lcuda $(gen)

escheme:
	$(CC) $(flags) -shared escheme.cu -lm -o ../escheme.so -lcuda $(gen)

clean:
	rm *.o *.so
