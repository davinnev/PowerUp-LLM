# PowerUp-LLM
An LLM optimizer that streamlines matrix vector multiplication process by implementing multithreading and synchronization techniques.
This optimizer is compatible for Linux machines.

### Improvements
(see folder "examples)
1. Running the matrix vector multiplication using a single thread. Time taken: 4.927 s
![alt text](https://github.com/davinnev/PowerUp-LLM/blob/main/examples/singlethread.jpg?raw=true)

2. Running the matrix vector multiplication using 4 threads. Time taken: 2.338 s
![alt text](https://github.com/davinnev/PowerUp-LLM/blob/main/examples/4threads.jpg?raw=true)

3. Running the matrix vector multiplication using 4 threads. Time taken: 1.548 s
![alt text](https://github.com/davinnev/PowerUp-LLM/blob/main/examples/16threads.jpg?raw=true)


### Running the script 
1. Clone the repository
```bash
git clone https://github.com/davinnev/PowerUp-LLM.git
```

2. Navigate to the source folder
```bash
cd src
```

3. Compile the code
```bash
gcc llama2.c -o llama2
```

4. Try running the optimizer
```bash
./llama2 {seed} {num of threads}
```
For example,
```bash
./llama2 42 4
```


