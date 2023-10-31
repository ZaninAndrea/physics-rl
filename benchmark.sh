echo "------------------------------------"
echo "FIRST RUN"
echo "------------------------------------"
/usr/bin/time -ao times.txt mpirun -n 1 python example_dojo_fenics.py
/usr/bin/time -ao times.txt mpirun -n 2 python example_dojo_fenics.py
/usr/bin/time -ao times.txt mpirun -n 4 python example_dojo_fenics.py

echo "------------------------------------"
echo "SECOND RUN"
echo "------------------------------------"
/usr/bin/time -ao times.txt mpirun -n 1 python example_dojo_fenics.py
/usr/bin/time -ao times.txt mpirun -n 2 python example_dojo_fenics.py
/usr/bin/time -ao times.txt mpirun -n 4 python example_dojo_fenics.py

echo "------------------------------------"
echo "THIRD RUN"
echo "------------------------------------"
/usr/bin/time -ao times.txt mpirun -n 1 python example_dojo_fenics.py
/usr/bin/time -ao times.txt mpirun -n 2 python example_dojo_fenics.py
/usr/bin/time -ao times.txt mpirun -n 4 python example_dojo_fenics.py