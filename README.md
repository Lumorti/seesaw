# seesaw

This program optimises a Bell scenario using a seesaw method in order to find Mutually Unbiased Bases (MUBs).

### Dependencies

In order to use this program, a working installation of [MOSEK](https://www.mosek.com/products/mosek/) is required, alongside a valid license installed to a valid location. See the [MOSEK documentation](https://docs.mosek.com/9.2/install/installation.html) for installation instructions. They offer a [free license](https://www.mosek.com/products/academic-licenses/) for academics. Once MOSEK is installed, the BASH variable $MSKHOME should be set to be the directory containing the "mosek" folder.

You also need make and g++ (in case they aren't installed already):
```bash
sudo apt-get install make g++
```

### Installing

To download this repo somewhere:
```bash
git clone https://github.com/Lumorti/seesaw
```

To then compile the binaries, first enter this new directory, then enter the code directory and run make:
```bash
cd seesaw/code
make
```

### Usage

To use the method to check for the existence of 2 MUBs in dimension 3:
```bash
./seesaw -d 3 -n 2
```

For other options, see the help:
```bash
./seesaw -h
```

