# PowerUp-LLM
An LLM optimizer that streamlines matrix vector multiplication process by implementing multithreading and synchronization techniques.

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


