# star_lite_design

This repository holds the code used to design the Star_Lite class of configurations at Hampton University.

## Design Criteria
Star_Lite has the following design criteria,

- Device should be roughly quasiaxisymmetric.
- Magnetic field should have X-point divertor structure.
- `0.0875 T` mean field strength on axis.
- `0.15 m` minimum coil-to-coil distance.
- `0.5 m` major radius
- `0.054 m` coil-to-vessel distance (larger than minor radius of coil).
- `60kA` upper bound on coil currents.
- coils should be perfectly nfp2 and stellarator symmetric.
- Surface and X-point must fit inside vacuum vessel.
- Coil curvature and length should be roughly that of the original QUASR design (scaled).
- Magnetic forces on coils should be low `< 4000 N/m`
- Coils should be robust to manufacturing errors (`> 1mm` tolerances)

## Useful notes
- We assume a `0.054 m` coil minor radius


## Installation instructions on Mac M3
1. Install pyenv with brew
    ```
    brew install pyenv
    ```
    Then follow the setup instructions of pyenv on the pyenv github page. 
2. Install python 3.10.1
    ```
    pyenv install 3.10.1
    ```
3. Make a directory to keep everything in,
    ```
    mkdir project
    cd project
    ```
4. Set the python and make a virtualenv,
    ```
    pyenv local 3.10.1
    python -m venv env
    source env/bin/activate
    ```
   The venv can be activated with `source env/bin/activate`.
5. Install openmppi and some pip packages
    ```
    brew install open-mpi
    python -m pip install numpy scipy matplotlib pandas cmake scikit-build ninja f90wrap mpi4py pyyaml uuid pyevtk bentley-ottmann ground
    python -m pip install simsopt"[MPI]"
    ```
7. Add the repo to your PYTHONPATH so that the directory structure is recognized by python. Modify your
    `.zshrc` to contain the following line (replacing the path with the relevant one),
    ```
    export PYTHONPATH="${PYTHONPATH}:/Users/mpadidar/code/star_lite"
    ```