# How to Run

## Matrix Multiplication Module Test

Install JDK.

```bash
sudo apt install openjdk-17-jdk
```

Install Scala CLI.

```bash
curl -sS "https://virtuslab.github.io/scala-cli-packages/KEY.gpg" | sudo gpg --dearmor  -o /etc/apt/trusted.gpg.d/scala-cli.gpg 2>/dev/null
sudo curl -s --compressed -o /etc/apt/sources.list.d/scala_cli_packages.list "https://virtuslab.github.io/scala-cli-packages/debian/scala_cli_packages.list"
sudo apt update
sudo apt install scala-cli
```

Install SBT.

```bash
echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | sudo tee /etc/apt/sources.list.d/sbt.list
echo "deb https://repo.scala-sbt.org/scalasbt/debian /" | sudo tee /etc/apt/sources.list.d/sbt_old.list
curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | sudo apt-key add
sudo apt-get update
sudo apt-get install sbt
```

Install Verilator.

```bash
sudo apt install verilator
```

Install GTKWave.

```bash
sudo apt install gtkwave
```

Download Chisel template.

```bash
git clone --recursive https://github.com/freechipsproject/chisel-template.git
cp build.sbt chisel-template/build.sbt
cp build.sc chisel-template/build.sc
cp src chisel-template/src
```

Run test.

```bash
sbt "testOnly GEMM.GEMMTest -- -DwriteVcd=1"
```

## Chipyard

Setup chipyard.

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
conda install -n base conda-lock==1.4.0
conda activate base
git clone https://github.com/ucb-bar/chipyard.git
cd chipyard
./build-setup.sh riscv-tools -s 6 -s 7 -s 8 -s 9
source env.sh
cd sims/verilator/
make CONFIG=RocketConfig VERILATOR_THREADS=12 -j 12
cd tests/
make
```