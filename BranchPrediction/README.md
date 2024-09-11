# Getting Started

Download and compile GEM5.

```bash
sudo apt install build-essential git m4 scons zlib1g zlib1g-dev libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev python-dev python
git clone https://github.com/gem5/gem5
cd gem5
scons build/X86/gem5.opt -j 16  # ~ 15 min
```

Run the examples and observe the output in `m5out` directory.

```bash
build/X86/gem5.opt configs/learning_gem5/part1/simple.py
build/X86/gem5.opt configs/learning_gem5/part2/simple_cache.py
```

Download and compile the GAP Benchmark Suite.

```bash
git clone https://github.com/sbeamer/gapbs.git
cd gapbs
make
```

Modify the GEM5 source files and recompile them.

```bash
mv gem5/src/cpu/pred/BranchPredictor.py gem5/src/cpu/pred/BranchPredictor.py.backup
mv gem5/src/cpu/pred/2bit_local.cc gem5/src/cpu/pred/2bit_local.cc.backup
cp src/BranchPredictor.py gem5/src/cpu/pred/BranchPredictor.py
cp src/2bit_local.cc gem5/src/cpu/pred/2bit_local.cc
scons build/X86/gem5.opt -j 16
```

Run the experiment code and check the results in `log/log.txt`. The three numbers represent IPC (Instructions Per Cycle), branch prediction count, and branch prediction error count, respectively.

```bash
python autotest.py
```