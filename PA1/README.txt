on PACE:
ssh scen9@coc-ice.pace.gatech.edu
mpicxx prog1.cpp -o prog1 -no-multibyte-chars
mpirun -np 4 ./prog1 4 3

on UBUNTU (debug):
mpicxx prog1.cpp -o prog1 -lm
mpirun -np 4 ./prog1 4 3 --mca orte_base_help_aggregate 0